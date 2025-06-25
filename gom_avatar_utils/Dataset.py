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
        dilated_mask: list[torch.Tensor],  # [..., 1, H, W]

        blending_param: typing.Any,
    ):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(
            img, (..., C, H, W),
            mask, (..., 1, H, W),

            *(
                _
                for cur_dilated_mask in dilated_mask
                for _ in (cur_dilated_mask, (..., 1, H, W))
            )
        )

        self.shape = utils.broadcast_shapes(
            shape,
            camera_transform,
            img.shape[:-3],
            mask.shape[:-3],

            *(cur_dilated_mask.shape[:-3]
              for cur_dilated_mask in dilated_mask),

            blending_param,
        )

        self.raw_camera_config = camera_config
        self.raw_camera_transform = camera_transform

        self.raw_img = img

        self.raw_mask = mask
        self.raw_dilated_mask = dilated_mask

        self.raw_blending_param = blending_param

    @property
    def camera_config(self) -> camera_utils.CameraConfig:
        return self.raw_camera_config

    @property
    def camera_transform(self) -> transform_utils.ObjectTransform:
        return self.raw_camera_transform.expand(self.shape)

    @property
    def img(self) -> torch.Tensor:
        return utils.batch_expand(self.raw_img, self.shape, 3)

    @property
    def mask(self) -> torch.Tensor:
        return utils.batch_expand(self.raw_mask, self.shape, 3)

    @property
    def dilated_mask(self) -> list[torch.Tensor]:
        return [
            utils.batch_expand(x, self.shape, 3)
            for x in self.raw_dilated_mask
        ]

    @property
    def blending_param(self) -> typing.Any:
        return utils.try_expand(self.raw_blending_param, self.shape)

    def __getitem__(self, idx) -> Sample:
        if not isinstance(idx, tuple):
            idx = (idx,)

        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.camera_transform[*idx],

            img=utils.batch_indexing(self.img, None, 3, idx),

            mask=utils.batch_indexing(self.mask, None, 3, idx),

            dilated_mask=[
                utils.try_batch_indexing(cur_dilated_mask, None, 3, idx)
                for cur_dilated_mask in self.raw_dilated_mask
            ],

            blending_param=None if self.blending_param is None
            else self.blending_param[*idx],
        )

    def expand(self, shape: tuple[int, ...]) -> Sample:
        return Sample(
            shape=shape,

            camera_config=self.raw_camera_config,
            camera_transform=self.raw_camera_transform,

            img=self.raw_img,

            mask=self.raw_mask,
            dilated_mask=self.raw_dilated_mask,

            blending_param=self.raw_blending_param,
        )

    def to(self, *args, **kwargs) -> Sample:
        def _f(x):
            return None if x is None else x.to(*args, **kwargs)

        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.raw_camera_transform.to(*args, **kwargs),

            img=_f(self.raw_img),

            mask=_f(self.raw_mask),


            dilated_mask=[
                _f(cur_dilated_mask)
                for cur_dilated_mask in self.raw_dilated_mask
            ],

            blending_param=_f(self.raw_blending_param),
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
            dilated_mask=sample.dilated_mask,

            blending_param=sample.blending_param
        )

    @property
    def shape(self):
        return self.sample.shape

    def __getitem__(self, idx) -> Sample:
        ret = self.sample[idx]

        ret.raw_img = ret.raw_img / 255

        return ret

    def to(self, *args, **kwargs) -> Dataset:
        self.sample = self.sample.to(*args, **kwargs)
        return self
