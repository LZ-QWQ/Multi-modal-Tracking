import torch
from torch.utils.data.distributed import DistributedSampler

# datasets related
# from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, TNL2k
# from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import opencv_loader, LTRLoader  # , sampler, processing

# import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process

# for rgbt
from lib.train.dataset import VTUAV, LasHeR, RGBT234, KAIST, LLVIPseq, M3FDseq, DepthTrack, RGBT234_T, LasHeR_T, VTUAV_Test
from lib.train.data import sampler_rgbt
from lib.train.data import processing_rgbt
from lib.train.data import transforms_rgbt
from torch.utils.data.dataloader import DataLoader
from lib.train.data import LTRLoader


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


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in [
            "LASOT",
            "GOT10K_vottrain",
            "GOT10K_votval",
            "GOT10K_train_full",
            "COCO17",
            "VID",
            "TRACKINGNET",
            "TNL2k",
            "VTUAV",
            "LasHeR",
            "RGBT234",
            "KAIST",
            "LLVIP",
            "M3FD",
            "DepthTrack-Train",
            "DepthTrack-Test",
            "LasHeR_T",  # for single TIR exp
            "RGBT234_T",
            "VTUAV-Test"
        ]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split="train", image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split="train", image_loader=image_loader))
        if name == "TNL2k":
            datasets.append(TNL2k(settings.env.tnl2k_dir, split="train", image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split="vottrain", image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split="vottrain", image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split="train_full", image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split="train_full", image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split="votval", image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split="votval", image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))

        # RGBT!
        if name == "VTUAV":
            if settings.use_lmdb:
                raise ValueError("VTUAV not support lmdb")
            else:
                datasets.append(VTUAV(settings.env.vtuav_dir, image_loader=image_loader))
        if name == "LasHeR":
            if settings.use_lmdb:
                raise ValueError("LasHeR not support lmdb")
            else:
                datasets.append(LasHeR(settings.env.lasher_dir, image_loader=image_loader))
        if name == "RGBT234":
            if settings.use_lmdb:
                raise ValueError("RGBT234 not support lmdb")
            else:
                datasets.append(RGBT234(settings.env.rgbt234_dir, image_loader=image_loader))
        if name == "KAIST":
            if settings.use_lmdb:
                raise ValueError("KAIST not support lmdb")
            else:
                datasets.append(KAIST(settings.env.kaist_dir, image_loader=image_loader))
        if name == "LLVIP":
            if settings.use_lmdb:
                raise ValueError("LLVIP not support lmdb")
            else:
                datasets.append(LLVIPseq(settings.env.llvip_dir, image_loader=image_loader))
        if name == "M3FD":
            if settings.use_lmdb:
                raise ValueError("M3FD not support lmdb")
            else:
                datasets.append(M3FDseq(settings.env.m3fd_dir, image_loader=image_loader))
        if name == "DepthTrack-Train":
            if settings.use_lmdb:
                raise ValueError("DepthTrack not support lmdb")
            else:
                datasets.append(DepthTrack(settings.env.depthtrack_dir, split="train", image_loader=image_loader))
        if name == "DepthTrack-Test":
            if settings.use_lmdb:
                raise ValueError("DepthTrack not support lmdb")
            else:
                datasets.append(DepthTrack(settings.env.depthtrack_dir, split="test", image_loader=image_loader))
        if name == "RGBT234_T":
            if settings.use_lmdb:
                raise ValueError("RGBT234_T not support lmdb")
            else:
                datasets.append(RGBT234_T(settings.env.rgbt234_dir, image_loader=image_loader))
        if name == "LasHeR_T":
            if settings.use_lmdb:
                raise ValueError("LasHeR_T not support lmdb")
            else:
                datasets.append(LasHeR_T(settings.env.lasher_dir, image_loader=image_loader))
        if name == "VTUAV-Test":
            if settings.use_lmdb:
                raise ValueError("VTUAV-Test not support lmdb")
            else:
                datasets.append(VTUAV_Test(settings.env.vtuav_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    if "LasHeR_T" in cfg.DATA.TRAIN.DATASETS_NAME:  # 临时举措，为了微调TIR
        from lib.train.data import sampler
        from lib.train.data import processing
        import lib.train.data.transforms as tfm

        DataLoader_withname = LTRLoader
    else:
        sampler = sampler_rgbt
        processing = processing_rgbt
        tfm = transforms_rgbt
        DataLoader_withname = DataLoader_withname_

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(
        tfm.ToTensorAndJitter(0.2), tfm.RandomHorizontalFlip_Norm(probability=0.5), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )

    transform_val = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)

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

    data_processing_val = processing.MixformerProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode="sequence",
        transform=transform_val,
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

    train_sampler = DistributedSampler(dataset_train, shuffle=True) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    # 就离谱,,,, 我选择用这个
    loader_train = DataLoader_withname(
        name="train",
        training=True,
        dataset=dataset_train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(
        datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_val,
        frame_sample_mode=sampler_mode,
        train_cls=train_score,
        pos_prob=0.5,
    )

    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = DataLoader_withname(
        name="val",
        training=False,
        dataset=dataset_val,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER * 2,  # val的时候搞多点试试???
        drop_last=True,
        sampler=val_sampler,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    if settings.local_rank in [-1, 0]:
        print("train dataset info:")
        print(dataset_train.datasets, dataset_train.p_datasets)
        print("val dataset info:")
        print(dataset_val.datasets, dataset_val.p_datasets)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    freeze_stage0 = getattr(cfg.TRAIN, "FREEZE_STAGE0", False)
    freeze_first_6layers = getattr(cfg.TRAIN, "FREEZE_FIRST_6LAYERS", False)
    rgbt_track = getattr(cfg.TRAIN, "RGBT_TRACK", False)
    rgbt_track_shared = getattr(cfg.TRAIN, "RGBT_TRACK_SHARED", False)
    rgbt_track_unibackbone = getattr(cfg.TRAIN, "RGBT_TRACK_UNIBACKBONE", False)

    # 应该是必须只能有一个true
    if (
        not rgbt_track and not rgbt_track_shared and not train_score and not rgbt_track_unibackbone and not freeze_first_6layers
    ):  # 最后这个用来给TIR finetune用的
        raise NotImplementedError

    if train_score:
        print("Only training score_branch. Learnable parameters are shown below.")
        param_dicts = [{"params": [p for n, p in net.named_parameters() if "score" in n and p.requires_grad]}]

        for n, p in net.named_parameters():
            if "score" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    elif freeze_stage0:  # only for CVT-large backbone
        assert "cvt_24" == cfg.MODEL.VIT_TYPE
        print("Freeze Stage0 of MixFormer cvt backbone. Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if "stage2" not in n and "box_head" not in n and "stage1" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if (("stage2" in n or "stage1" in n) and p.requires_grad)],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
    elif freeze_first_6layers:  # only for ViT-Large backbone
        assert "large_patch16" == cfg.MODEL.VIT_TYPE
        print("Freeze the first 6 layers of MixFormer vit backbone. Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if (
                "blocks.0." in n
                or "blocks.1." in n
                or "blocks.2." in n
                or "blocks.3." in n
                or "blocks.4." in n
                or "blocks.5." in n
                or "patch_embed" in n
            ):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
    elif rgbt_track:
        # two-stream
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        fusion_linear_proj_names = ["reference_points", "sampling_offsets"]
        # 暂时计划为: 加载RGB预训练模型, 冻住RGB部分的backbone, 0.001训TIR部分的backbone, 全力训融合模块, 回归头固定(后面试试看给一点学习率or后面几个epoch再给(问题是咋给啊???))
        print("train using rgbt strategy")
        for n, p in net.named_parameters():
            if "pos_embed" in n:  # 位置编码不训练 "box_head"
                p.requires_grad = False
            else:
                p.requires_grad = True

        # net.box_head.eval()
        # net.backbone_v.eval()
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if "backbone_i" in n and p.requires_grad],
                "lr": 0.1 * cfg.TRAIN.LR,
            },
            {
                "params": [p for n, p in net.named_parameters() if "backbone_v" in n and p.requires_grad],
                "lr": 0.02 * cfg.TRAIN.LR,
            },
            {
                "params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad],
                "lr": 0.02 * cfg.TRAIN.LR,
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "fusion_vi" in n and not match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "fusion_vi" in n and match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad
                ],
                "lr": 0.1 * cfg.TRAIN.LR,
            },
        ]
        # import torch.distributed as dist

        # if dist.get_rank() == 0:
        #     for n, p in net.named_parameters():
        #         if "fusion_vi" in n and match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad:
        #             print(n)
        #     print("喵喵喵")
        #     for n, p in net.named_parameters():
        #         if "fusion_vi" in n and not match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad:
        #             print(n)
        # exit()

    elif rgbt_track_shared:
        # TODO MAE初始化的咋设置
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        fusion_linear_proj_names = ["reference_points", "sampling_offsets"]

        print("train using rgbt shared backbone strategy")
        for n, p in net.named_parameters():
            if "pos_embed" in n:  # 位置编码不训练 "box_head"
                p.requires_grad = False
            else:
                p.requires_grad = True

        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 0.02 * cfg.TRAIN.LR,  # 0.02
            },
            # {
            #     "params": [
            #         p for n, p in net.named_parameters() if "backbone" in n and "norm1_i" in n and "norm2_i" in n and p.requires_grad
            #     ],
            #     "lr": 0.1 * cfg.TRAIN.LR,
            # },
            # {
            #     "params": [
            #         p
            #         for n, p in net.named_parameters()
            #         if "backbone" in n and "norm1_i" not in n and "norm2_i" not in n and p.requires_grad
            #     ],
            #     "lr": 0.02 * cfg.TRAIN.LR,  # 0.02
            # },
            {
                "params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad],
                "lr": 0.02 * cfg.TRAIN.LR,  # 加载mixformer预训练的话就1 0.02
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "fusion_vi" in n and not match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "fusion_vi" in n and match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad
                ],
                "lr": 0.1 * cfg.TRAIN.LR,
            },
        ]
    elif rgbt_track_unibackbone:

        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        fusion_linear_proj_names = ["reference_points", "sampling_offsets"]
        print("train using rgbt uni-backbone strategy")
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 0.1 * cfg.TRAIN.LR,
            },
            {
                "params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad],
                "lr": 0.02 * cfg.TRAIN.LR,
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "fusion_vi" in n and not match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "fusion_vi" in n and match_name_keywords(n, fusion_linear_proj_names) and p.requires_grad
                ],
                "lr": 0.1 * cfg.TRAIN.LR,
            },
        ]

    else:  # train network except for score prediction module
        for n, p in net.named_parameters():
            if "score" in n:
                p.requires_grad = False
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.TRAIN.LR_DROP_EPOCH, gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE
        )
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler


# 服了, 为了兼容后续
class DataLoader_withname_(DataLoader):
    def __init__(self, name, training, *args, **kargs):
        super().__init__(*args, **kargs)
        self.name = name
        self.training = training
