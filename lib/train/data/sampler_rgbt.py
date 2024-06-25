import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
import traceback

import os
import cv2


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(
        self,
        datasets,
        p_datasets,
        samples_per_epoch,
        max_gap,
        num_search_frames,
        num_template_frames=1,
        processing=no_processing,
        frame_sample_mode="causal",
        train_cls=False,
        pos_prob=0.5,
    ):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None, allow_invisible=False, force_invisible=False):
        """Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
            # raise NotImplementedError
        else:
            return self.getitem()

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                H, W, _ = template_frames[0][0].shape
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(
                        1,
                    )
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                # negative samples
                else:
                    label = torch.zeros(
                        1,
                    )
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()

                data = {
                    "template_images": template_frames,
                    "template_anno": template_anno["bbox"],
                    "search_images": search_frames,
                    "search_anno": search_anno["bbox"],
                    "dataset": dataset.get_name(),
                    # "test_class": meta_obj_test.get("object_class_name"),
                }

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data["valid"]
                if not valid:
                    continue

                data["template_images_v"] = [img_vi[0] for img_vi in data["template_images"]]
                data["template_images_i"] = [img_vi[1] for img_vi in data["template_images"]]
                del data["template_images"]
                data["search_images_v"] = [img_vi[0] for img_vi in data["search_images"]]
                data["search_images_i"] = [img_vi[1] for img_vi in data["search_images"]]
                del data["search_images"]
                data["template_anno_v"] = [anno_vi[0] for anno_vi in data["template_anno"]]
                data["template_anno_i"] = [anno_vi[1] for anno_vi in data["template_anno"]]
                del data["template_anno"]
                data["search_anno_v"] = [anno_vi[0] for anno_vi in data["search_anno"]]
                data["search_anno_i"] = [anno_vi[1] for anno_vi in data["search_anno"]]
                del data["search_anno"]
            except Exception as e:
                valid = False
                traceback.print_exc()

        return data

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == "causal":
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(
                            visible, num_ids=1, min_id=self.num_template_frames - 1, max_id=len(visible) - self.num_search_frames
                        )
                        prev_frame_ids = self._sample_visible_ids(
                            visible,
                            num_ids=self.num_template_frames - 1,
                            min_id=base_frame_id[0] - self.max_gap - gap_increase,
                            max_id=base_frame_id[0],
                        )
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(
                            visible,
                            min_id=template_frame_ids[0] + 1,
                            max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                            num_ids=self.num_search_frames,
                        )
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # [N,2,(H,W,C)]
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                data = {
                    "template_images": template_frames,  # [N_t,2,(H,W,C)]
                    "template_anno": template_anno["bbox"],  # (N, 2, 4)
                    "search_images": search_frames,  # [N_s,2,(H,W,C)]
                    "search_anno": search_anno["bbox"],  # (N, 2, 4)
                    "dataset": dataset.get_name(),
                    # "test_class": meta_obj_test.get("object_class_name"),
                }
                # data = TensorDict(data)

                # make data augmentation

                data = self.processing(data)

                # check whether data is valid
                valid = data["valid"]
                if not valid:
                    # print("喵喵喵")
                    continue

                # vis for debug, valid False 时可能会没执行完所有过程导致numpy()报错
                # save_dir = "./visualization/train_data"
                # os.makedirs(save_dir, exist_ok=True)
                # mean_ = np.array([0.485, 0.456, 0.406])  # 硬敲吧, 太难传参进来了
                # std_ = np.array([0.229, 0.224, 0.225])
                # for idx, img_vi in enumerate(data["template_images"]):
                #     anno_vi = data["template_anno"][idx]

                #     img_v = 255 * (std_ * img_vi[0].numpy().transpose([1, 2, 0]) + mean_)
                #     img_i = 255 * (std_ * img_vi[1].numpy().transpose([1, 2, 0]) + mean_)
                #     img_v = img_v.copy().astype(np.uint8)
                #     img_i = img_i.copy().astype(np.uint8)
                #     bbox_xywh_v = anno_vi[0].numpy() * (img_v.shape[0] - 1)  # 0-1 to 0-size 长宽一样
                #     bbox_xywh_i = anno_vi[1].numpy() * (img_v.shape[0] - 1)
                #     bbox_xywh_v = bbox_xywh_v.astype(np.int32)
                #     bbox_xywh_i = bbox_xywh_i.astype(np.int32)
                #     cv2.rectangle(
                #         img_v,
                #         [bbox_xywh_v[0], bbox_xywh_v[1]],
                #         [bbox_xywh_v[0] + bbox_xywh_v[2], bbox_xywh_v[1] + bbox_xywh_v[3]],
                #         color=(255, 0, 0),
                #         thickness=1,
                #     )
                #     cv2.rectangle(
                #         img_i,
                #         [bbox_xywh_i[0], bbox_xywh_i[1]],
                #         [bbox_xywh_i[0] + bbox_xywh_i[2], bbox_xywh_i[1] + bbox_xywh_i[3]],
                #         color=(0, 0, 255),
                #         thickness=1,
                #     )
                #     img = np.concatenate([img_v, img_i], axis=1)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     cv2.imwrite(os.path.join(save_dir, "template_{}.png".format(idx)), img)
                # for idx, img_vi in enumerate(data["search_images"]):
                #     anno_vi = data["search_anno"][idx]

                #     img_v = 255 * (std_ * img_vi[0].numpy().transpose([1, 2, 0]) + mean_)
                #     img_i = 255 * (std_ * img_vi[1].numpy().transpose([1, 2, 0]) + mean_)
                #     img_v = img_v.copy().astype(np.uint8)
                #     img_i = img_i.copy().astype(np.uint8)
                #     bbox_xywh_v = anno_vi[0].numpy() * (img_v.shape[0] - 1)  # 0-1 to 0-size 长宽一样
                #     bbox_xywh_i = anno_vi[1].numpy() * (img_v.shape[0] - 1)
                #     bbox_xywh_v = bbox_xywh_v.astype(np.int32)
                #     bbox_xywh_i = bbox_xywh_i.astype(np.int32)
                #     cv2.rectangle(
                #         img_v,
                #         [bbox_xywh_v[0], bbox_xywh_v[1]],
                #         [bbox_xywh_v[0] + bbox_xywh_v[2], bbox_xywh_v[1] + bbox_xywh_v[3]],
                #         color=(255, 0, 0),
                #         thickness=1,
                #     )
                #     cv2.rectangle(
                #         img_i,
                #         [bbox_xywh_i[0], bbox_xywh_i[1]],
                #         [bbox_xywh_i[0] + bbox_xywh_i[2], bbox_xywh_i[1] + bbox_xywh_i[3]],
                #         color=(0, 0, 255),
                #         thickness=1,
                #     )
                #     img = np.concatenate([img_v, img_i], axis=1)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     cv2.imwrite(os.path.join(save_dir, "search_{}.png".format(idx)), img)

                # 出此下策......
                # template 应该是 [N_t,C,H,W] 推理再展开吧
                data["template_images_v"] = [img_vi[0] for img_vi in data["template_images"]]
                data["template_images_i"] = [img_vi[1] for img_vi in data["template_images"]]
                del data["template_images"]
                data["search_images_v"] = [img_vi[0] for img_vi in data["search_images"]]
                data["search_images_i"] = [img_vi[1] for img_vi in data["search_images"]]
                del data["search_images"]
                data["template_anno_v"] = [anno_vi[0] for anno_vi in data["template_anno"]]
                data["template_anno_i"] = [anno_vi[1] for anno_vi in data["template_anno"]]
                del data["template_anno"]
                data["search_anno_v"] = [anno_vi[0] for anno_vi in data["search_anno"]]
                data["search_anno_i"] = [anno_vi[1] for anno_vi in data["search_anno"]]
                del data["search_anno"]
            except Exception as e:
                valid = False
                traceback.print_exc()

        return data

    def get_center_box(self, H, W, ratio=1 / 8):
        cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
        tmp = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]
        return torch.tensor([tmp, tmp])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict["visible"]

            enough_visible_frames = (
                visible.type(torch.int64).sum().item() > 2 * (self.num_search_frames + self.num_template_frames) and len(visible) >= 20
            )

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id, allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids


if __name__ == "__main__":
    from lib.train.data import opencv_loader, LTRLoader  # , sampler, processing

    # for rgbt
    from lib.train.dataset import VTUAV, LasHeR, RGBT234, KAIST, LLVIPseq, M3FDseq
    from lib.train.data import sampler_rgbt as sampler
    from lib.train.data import processing_rgbt as processing
    import lib.train.data.transforms_rgbt as tfm
    from torch.utils.data.dataloader import DataLoader

    import lib.config.mixformer_vit_rgbt.config as config
    import lib.train.admin.settings as ws_settings

    from lib.train.base_functions import names2datasets

    import time

    cfg = config.cfg
    config.update_config_from_file("/home/lizheng/data4/MixFormer/experiments/mixformer_vit_rgbt/baseline_attention_alldata.yaml")

    def update_settings(settings, cfg):
        settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
        settings.search_area_factor = {"template": cfg.DATA.TEMPLATE.FACTOR, "search": cfg.DATA.SEARCH.FACTOR}
        settings.output_sz = {"template": cfg.DATA.TEMPLATE.SIZE, "search": cfg.DATA.SEARCH.SIZE}
        settings.center_jitter_factor = {"template": cfg.DATA.TEMPLATE.CENTER_JITTER, "search": cfg.DATA.SEARCH.CENTER_JITTER}
        settings.scale_jitter_factor = {"template": cfg.DATA.TEMPLATE.SCALE_JITTER, "search": cfg.DATA.SEARCH.SCALE_JITTER}
        settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
        settings.print_stats = None
        settings.batchsize = cfg.TRAIN.BATCH_SIZE
        settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE

    settings = ws_settings.Settings()
    update_settings(settings, cfg)
    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    settings.use_lmdb = False
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(
        tfm.ToTensorAndJitter(0.2), tfm.RandomHorizontalFlip_Norm(probability=0.5), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )

    data_processing_train = processing.MixformerProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode="sequence",
        transform=transform_train,
        joint_transform=transform_joint,
        settings=settings,
        train_score=train_score,
    )

    dataset_train = sampler.TrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,  # None,  # 根据数量来决定
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_train,
        frame_sample_mode=sampler_mode,
        train_cls=train_score,
        pos_prob=0.5,
    )

    # 就离谱,,,, 我选择用这个
    loader_train = DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        sampler=None,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    mean_ = np.array([0.485, 0.456, 0.406])  # 硬敲吧, 太难传参进来了
    std_ = np.array([0.229, 0.224, 0.225])

    def tensor2im(t):
        return (255 * (std_ * t.numpy().transpose([1, 2, 0]) + mean_)).copy().astype(np.uint8)

    def anno2bbox(a, size_):
        return (a.numpy() * (size_ - 1)).astype(np.int32)

    for idx, data in enumerate(loader_train):
        assert len(data["template_images_v"]) == 2  # template + online_templaet
        img_list = []  # tv ti otv oti sv si
        img_list.append(tensor2im(data["template_images_v"][0][0]))
        img_list.append(tensor2im(data["template_images_i"][0][0]))
        img_list.append(tensor2im(data["template_images_v"][1][0]))
        img_list.append(tensor2im(data["template_images_i"][1][0]))
        img_list.append(tensor2im(data["search_images_v"][0][0]))
        img_list.append(tensor2im(data["search_images_i"][0][0]))

        bbox_list = []
        bbox_list.append(anno2bbox(data["template_anno_v"][0][0], img_list[0].shape[0]))
        bbox_list.append(anno2bbox(data["template_anno_i"][0][0], img_list[1].shape[0]))
        bbox_list.append(anno2bbox(data["template_anno_v"][1][0], img_list[2].shape[0]))
        bbox_list.append(anno2bbox(data["template_anno_i"][1][0], img_list[3].shape[0]))
        bbox_list.append(anno2bbox(data["search_anno_v"][0][0], img_list[4].shape[0]))
        bbox_list.append(anno2bbox(data["search_anno_i"][0][0], img_list[5].shape[0]))

        for img, bbox in zip(img_list, bbox_list):
            cv2.rectangle(
                img,
                [bbox[0], bbox[1]],
                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                color=(255, 0, 0),
                thickness=1,
            )

        for idx in range(4):
            img_list[idx] = cv2.resize(img_list[idx], img_list[-1].shape[:2])

        img_up = np.concatenate(img_list[0::2], axis=1)
        img_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2RGB)
        img_down = np.concatenate(img_list[1::2], axis=1)
        img = np.concatenate([img_up, img_down], axis=0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/lizheng/data4/MixFormer/visualization/train_data/all.png", img)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # exit()
        time.sleep(0.8)
