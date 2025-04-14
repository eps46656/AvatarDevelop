from __future__ import annotations

import typing

import torch
from beartype import beartype

from .. import (camera_utils, dataset_utils, transform_utils, utils,
                vision_utils)


@beartype
class Sample:
    def __init__(
        self,
        *,
        shape: typing.Optional[tuple[int, ...]] = None,

        camera_config: camera_utils.CameraConfig,
        camera_transform: transform_utils.ObjectTransform,  # [...]

        img: torch.Tensor,  # [..., C, H, W]
        mask: torch.Tensor,  # [..., 1, H, W]
        blending_param: object,
    ):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(
            img, (..., C, H, W),
            mask, (..., 1, H, W),
        )

        self.shape = utils.broadcast_shapes(
            shape,
            camera_transform,
            img.shape[:-3],
            mask.shape[:-3],
            blending_param,
        )

        self.raw_camera_config = camera_config
        self.raw_camera_transform = camera_transform

        self.raw_img = img
        self.raw_mask = mask
        self.raw_blending_param = blending_param

    @property
    def camera_config(self) -> camera_utils.CameraConfig:
        return self.raw_camera_config

    @property
    def camera_transform(self) -> transform_utils.ObjectTransform:
        return self.raw_camera_transform.expand(self.shape)

    @property
    def img(self) -> torch.Tensor:
        return utils.try_batch_expand(self.raw_img, self.shape, -3)

    @property
    def mask(self) -> torch.Tensor:
        return utils.try_batch_expand(self.raw_mask, self.shape, -3)

    @property
    def blending_param(self):
        return self.raw_blending_param.expand(self.shape)

    @property
    def device(self) -> torch.device:
        return self.camera_transform.device

    def __getitem__(self, idx) -> Sample:
        if not isinstance(idx, tuple):
            idx = (idx,)

        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.camera_transform[*idx],

            img=self.img[*idx, ..., :, :, :],
            mask=self.mask[*idx, ..., :, :, :],

            blending_param=self.blending_param[*idx],
        )

    def expand(self, shape: tuple[int, ...]) -> Sample:
        return Sample(
            shape=shape,

            camera_config=self.raw_camera_config,
            camera_transform=self.raw_camera_transform,

            img=self.raw_img,
            mask=self.raw_mask,
            blending_param=self.raw_blending_param,
        )

    def to(self, *args, **kwargs) -> Sample:
        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.raw_camera_transform.to(*args, **kwargs),

            img=self.raw_img.to(*args, **kwargs),
            mask=self.raw_mask.to(*args, **kwargs),
            blending_param=self.raw_blending_param.to(*args, **kwargs),
        )


@beartype
class Dataset(dataset_utils.Dataset):
    def __init__(self, sample: Sample):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(sample.img, (..., C, H, W))

        self.sample = Sample(
            shape=None,

            camera_config=sample.camera_config,
            camera_transform=sample.camera_transform,

            img=sample.img,
            mask=sample.mask,

            blending_param=sample.blending_param
        )

    @property
    def shape(self):
        return self.sample.camera_transform.shape

    @property
    def device(self) -> torch.device:
        return self.sample.device

    def __getitem__(self, idx) -> Sample:
        ret = self.sample[idx]

        ret.raw_img = vision_utils.normalize_image(ret.raw_img)

        return ret

    def to(self, *args, **kwargs) -> Dataset:
        self.sample = self.sample.to(*args, **kwargs)
        return self
