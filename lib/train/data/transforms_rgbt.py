import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
import cv2


class Transform:
    """A set of transformations, used for e.g. data augmentation.
    Args of constructor:
        transforms: An arbitrary number of transformations, derived from the TransformBase class.
                    They are applied in the order they are given.

    The Transform object can jointly transform images, bounding boxes and segmentation masks.
    This is done by calling the object with the following key-word arguments (all are optional).

    The following arguments are inputs to be transformed. They are either supplied as a single instance, or a list of instances.
        image  -  Image
        coords  -  2xN dimensional Tensor of 2D image coordinates [y, x]
        bbox  -  Bounding box on the form [x, y, w, h]
        mask  -  Segmentation mask with discrete classes

    The following parameters can be supplied with calling the transform object:
        joint [Bool]  -  If True then transform all images/coords/bbox/mask in the list jointly using the same transformation.
                         Otherwise each tuple (images, coords, bbox, mask) will be transformed independently using
                         different random rolls. Default: True.
        new_roll [Bool]  -  If False, then no new random roll is performed, and the saved result from the previous roll
                            is used instead. Default: True.

    Check the DiMPProcessing class for examples.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ["image", "coords", "bbox", "mask", "att"]
        self._valid_args = ["joint", "new_roll", "after_split"]
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError(
                    'Incorrect input "{}" to transform. Only supports inputs {} and arguments {}.'.format(
                        v, self._valid_inputs, self._valid_args
                    )
                )

        joint_mode = inputs.get("joint", True)
        new_roll = inputs.get("new_roll", True)
        after_split = inputs.get("after_split", False)

        if not joint_mode:
            out = zip(*[self(**inp, after_split=True) for inp in self._split_inputs(inputs)])  # 解开了就标记为 after_split
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll, after_split=after_split)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0] != "joint" and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TransformBase:
    """Base class for transformation objects. See the Transform class for details."""

    def __init__(self):
        """2020.12.24 Add 'att' to valid inputs"""
        self._valid_inputs = ["image", "coords", "bbox", "mask", "att"]
        self._valid_args = ["new_roll"]
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        """
        input_vars : {"key1": [[v1,i1], [v2,i2], [v3,i3], ......], "key2": ......}
        """
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args["new_roll"]:  # input_args.get("new_roll", True)
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, "transform_" + var_name)
                if var_name in ["coords", "bbox"]:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params

                if not inputs["after_split"]:
                    outputs[var_name] = [transform_func(x, *params) for x in var]  # 遍历外层列表
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        """
        inputs : {"key1": [[v1,i1], [v2,i2], [v3,i3], ......], "key2": ......}
        """
        return inputs["image"][0][0].shape[:2]

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """Must be deterministic"""
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """Must be deterministic"""
        return coords

    def transform_bbox(self, bbox_vi, image_shape, *rand_params):
        """Assumes [x, y, w, h]"""
        # Check if not overloaded
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox_vi
        return [self._transform_bbox_single_(bbox_temp, image_shape, *rand_params) for bbox_temp in bbox_vi]

    def _transform_bbox_single_(self, bbox, image_shape, *rand_params):
        coord = bbox.clone().view(-1, 2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """Must be deterministic"""
        return mask

    def transform_att(self, att, *rand_params):
        """2020.12.24 Added to deal with attention masks"""
        return att


class ToTensor(TransformBase):
    """Convert to a Tensor"""

    def transform_image(self, images_vi):
        image_v = torch.from_numpy(images_vi[0].transpose((2, 0, 1))).float().div(255.0)
        image_i = cv2.applyColorMap(images_vi[1], cv2.COLORMAP_JET)  # totensor 之前, 需要将热红外转换成伪彩色, 采用JET转
        image_i = torch.from_numpy(image_i.transpose((2, 0, 1))).float().div(255.0)
        # backward compatibility
        return [image_v, image_i]

    def transfrom_mask(self, masks):
        if isinstance(masks[0], np.ndarray):
            return [torch.from_numpy(mask) for mask in masks]

    def transform_att(self, atts):
        if isinstance(atts[0], np.ndarray):
            return [torch.from_numpy(att).to(torch.bool) for att in atts]
        elif isinstance(atts[0], torch.Tensor):
            return [att.to(torch.bool) for att in atts]
        else:
            raise ValueError("dtype must be np.ndarray or torch.Tensor")


class ToTensorAndJitter(TransformBase):
    """Convert to a Tensor and jitter brightness"""

    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def roll_tir(self):  # 凭感觉给tir减半抖亮度
        return np.random.uniform(max(0, 1 - self.brightness_jitter / 2.0), 1 + self.brightness_jitter / 2.0)

    def transform_image(self, image_vi, brightness_factor):
        # handle numpy array

        image_v = torch.from_numpy(image_vi[0].transpose((2, 0, 1)))

        tir_factor = self.roll()  # using different from RGB
        image_i = (image_vi[1] * tir_factor).clip(0.0, 255.0).astype(np.uint8)
        image_i = cv2.applyColorMap(image_i, cv2.COLORMAP_JET)
        image_i = torch.from_numpy(image_i.transpose((2, 0, 1)))

        # tir_factor = self.roll()  # using different from RGB
        # backward compatibility
        if self.normalize:
            image_v = image_v.float().mul(brightness_factor / 255.0).clamp(0.0, 1.0)
            image_i = image_i.float().div(255.0).clamp(0.0, 1.0)
        else:
            image_v = image_v.float().mul(brightness_factor).clamp(0.0, 255.0)
            image_i = image_i.float().clamp(0.0, 255.0)

        # 用JET的话要注释掉他!!!! 和 暂时注释掉 jitter(brightness)
        # image_i = torch.tile(torch.mean(image_i, axis=0, keepdims=True), (3, 1, 1))

        return [image_v, image_i]

    def transform_mask(self, masks, brightness_factor):
        if isinstance(masks[0], np.ndarray):
            return [torch.from_numpy(mask) for mask in masks]
        else:
            return masks

    def transform_att(self, atts, brightness_factor):
        if isinstance(atts[0], np.ndarray):
            return [torch.from_numpy(att).to(torch.bool) for att in atts]
        elif isinstance(atts[0], torch.Tensor):
            return [att.to(torch.bool) for att in atts]
        else:
            raise ValueError("dtype must be np.ndarray or torch.Tensor")


class Normalize(TransformBase):
    """Normalize image"""

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, images_vi):
        return [
            tvisf.normalize(images_vi[0], self.mean, self.std, self.inplace),
            tvisf.normalize(images_vi[1], self.mean, self.std, self.inplace),
        ]


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, images, do_grayscale):
        """
        images : (v, i)
        TIR 不需要变
        """
        if do_grayscale:
            if torch.is_tensor(images[0]):
                raise NotImplementedError("Implement torch variant.")

            img_gray = cv.cvtColor(images[0], cv.COLOR_RGB2GRAY)
            return [np.stack([img_gray, img_gray, img_gray], axis=2), images[1]]
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return images


class ToBGR(TransformBase):
    """Converts image to BGR"""

    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError("Implement torch variant.")
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip image randomly with a probability p."""

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, images, do_flip):
        if do_flip:
            if torch.is_tensor(images[0]):
                return [image.flip((2,)) for image in images]
            return [np.fliplr(img_temp).copy() for img_temp in images]  # vi
        return images

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1, :] = (image_shape[1] - 1) - coords[1, :]
            return coords_flip
        return coords

    def transform_mask(self, masks, do_flip):
        if do_flip:
            if torch.is_tensor(masks[0]):
                return [mask.flip((-1,)) for mask in masks]
            return [np.fliplr(mask).copy() for mask in masks]
        return masks

    def transform_att(self, atts, do_flip):
        if do_flip:
            if torch.is_tensor(atts[0]):
                return [att.flip((-1,)) for att in atts]
            return [np.fliplr(att).copy() for att in atts]
        return atts


class RandomHorizontalFlip_Norm(RandomHorizontalFlip):
    """Horizontally flip image randomly with a probability p.
    The difference is that the coord is normalized to [0,1]"""

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def transform_coords(self, coords, image_shape, do_flip):
        """we should use 1 rather than image_shape"""
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1, :] = 1 - coords[1, :]
            return coords_flip
        return coords
