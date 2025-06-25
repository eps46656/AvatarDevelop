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

        person_mask: torch.Tensor,  # [..., 1, H, W]
        skin_mask: torch.Tensor,  # [..., 1, H, W]

        blending_param: object,
    ):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(
            img, (..., C, H, W),
            person_mask, (..., 1, H, W),
            skin_mask, (..., 1, H, W),
        )

        self.shape = utils.broadcast_shapes(
            shape,
            camera_transform,
            img.shape[:-3],

            person_mask.shape[:-3],
            skin_mask.shape[:-3],

            blending_param,
        )

        self.raw_camera_config = camera_config
        self.raw_camera_transform = camera_transform.expand(self.shape)

        self.raw_img = img

        self.raw_person_mask = person_mask
        self.raw_skin_mask = skin_mask

        self.raw_blending_param = blending_param

    @property
    def camera_config(self) -> camera_utils.CameraConfig:
        return self.raw_camera_config

    @property
    def camera_transform(self) -> transform_utils.ObjectTransform:
        return self.raw_camera_transform.expand(self.shape)

    @property
    def img(self) -> torch.Tensor:
        return utils.try_batch_expand(self.raw_img, self.shape, 3)

    @property
    def person_mask(self) -> torch.Tensor:
        return utils.try_batch_expand(self.raw_person_mask, self.shape, 3)

    @property
    def skin_mask(self) -> torch.Tensor:
        return utils.try_batch_expand(self.raw_skin_mask, self.shape, 3)

    @property
    def blending_param(self):
        return self.raw_blending_param.expand(self.shape)

    def __len__(self) -> int:
        return self.shape.numel()

    def __getitem__(self, idx) -> Sample:
        if not isinstance(idx, tuple):
            idx = (idx,)

        def _f(x, cdims):
            return utils.batch_indexing(x, self.shape, cdims, idx)

        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.camera_transform[*idx],

            img=_f(self.raw_img, 3),
            person_mask=_f(self.raw_person_mask, 3),
            skin_mask=_f(self.raw_skin_mask, 3),

            blending_param=self.blending_param[*idx],
        )

    def expand(self, shape: tuple[int, ...]) -> Sample:
        return Sample(
            shape=shape,

            camera_config=self.raw_camera_config,
            camera_transform=self.raw_camera_transform,

            img=self.raw_img,

            person_mask=self.raw_person_mask,
            skin_mask=self.raw_skin_mask,

            blending_param=self.raw_blending_param,
        )

    def to(self, *args, **kwargs) -> Sample:
        return Sample(
            camera_config=self.camera_config,
            camera_transform=self.raw_camera_transform.to(*args, **kwargs),

            img=self.raw_img.to(*args, **kwargs),

            person_mask=self.person_mask.to(*args, **kwargs),
            skin_mask=self.raw_skin_mask.to(*args, **kwargs),

            blending_param=self.raw_blending_param.to(*args, **kwargs),
        )


class Dataset(dataset_utils.Dataset):
    def __init__(self, sample: Sample):
        self.sample = Sample(
            shape=None,

            camera_config=sample.camera_config,
            camera_transform=sample.camera_transform,

            img=sample.img,
            person_mask=sample.person_mask,
            skin_mask=sample.skin_mask,

            blending_param=sample.blending_param
        )

    def __getitem__(self, item: int) -> Sample:
        return tuple()

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
