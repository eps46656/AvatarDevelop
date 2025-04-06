from __future__ import annotations

import typing

import torch
from beartype import beartype

from .. import camera_utils, dataset_utils, transform_utils, utils


@beartype
class Sample:
    def __init__(
        self,
        *,
        camera_config: camera_utils.CameraConfig,
        camera_transform: transform_utils.ObjectTransform,  # [...]

        img: torch.Tensor,  # [..., C, H, W]
        mask: torch.Tensor,  # [..., H, W]
        blending_param: object,
    ):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(
            img, (..., C, H, W),
            mask, (..., H, W),
        )

        shape = utils.broadcast_shapes(
            camera_transform,
            img.shape[:-3],
            mask.shape[:-2],
            blending_param,
        )

        camera_transform = camera_transform.expand(shape)
        img = img.expand(shape + (C, H, W))
        mask = mask.expand(shape + (H, W))
        blending_param = blending_param.expand(shape)

        # ---

        self.camera_config = camera_config
        self.camera_transform = camera_transform

        self.img = img
        self.mask = mask
        self.blending_param = blending_param

    @property
    def shape(self) -> torch.Size:
        return self.camera_transform.shape

    @property
    def device(self) -> torch.device:
        return self.camera_transform.device

    def to(self, *args, **kwargs) -> Sample:
        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.camera_transform.to(*args, **kwargs),

            img=self.img.to(*args, **kwargs),
            mask=self.mask.to(*args, **kwargs),
            blending_param=self.blending_param.to(*args, **kwargs),
        )

    def __getitem__(self, idx) -> Sample:
        if not isinstance(idx, tuple):
            idx = (idx,)

        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.camera_transform[*idx],

            img=self.img[*idx, :, :, :],
            mask=self.mask[*idx, :, :],

            blending_param=self.blending_param[*idx],
        )


@beartype
class Dataset(dataset_utils.Dataset):
    def __init__(self, sample: Sample):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(sample.img, (..., C, H, W))

        self.sample = Sample(
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

    def to(self, *args, **kwargs) -> Dataset:
        self.sample = self.sample.to(*args, **kwargs)
        return self

    def __getitem__(self, idx) -> Sample:
        return self.sample[idx]
