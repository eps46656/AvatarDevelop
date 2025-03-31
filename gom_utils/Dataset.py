
import typing

import torch
from beartype import beartype

from .. import camera_utils, dataset_utils, transform_utils, utils


@beartype
class Sample:
    camera_transform: transform_utils.ObjectTransform  # [...]
    camera_config: camera_utils.CameraConfig

    img: torch.Tensor  # [..., C, H, W]
    mask: typing.Optional[torch.Tensor]  # [..., H, W]
    blending_param: object

    def __init__(
        self,
        *,
        camera_transform: transform_utils.ObjectTransform,  # [...]
        camera_config: camera_utils.CameraConfig,

        img: torch.Tensor,  # [..., C, H, W]
        mask: typing.Optional[torch.Tensor],  # [..., H, W]
        blending_param: object,
    ):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(img, (..., C, H, W))

        if mask is not None:
            utils.check_shapes(mask, (..., H, W))

        # ---

        self.camera_transform = camera_transform
        self.camera_config = camera_config

        self.img = img
        self.mask = mask
        self.blending_param = blending_param

    @property
    def shape(self) -> torch.Size:
        return utils.broadcast_shapes(
            self.camera_transform.shape,
            self.img.shape[:-3],
            utils.try_get_batch_shape(self.mask, -2),
            self.blending_param.shape,
        )

    @property
    def device(self) -> torch.device:
        return self.camera_transform.device

    def to(self, *args, **kwargs) -> typing.Self:
        return Sample(
            camera_transform=self.camera_transform.to(*args, **kwargs),
            camera_config=self.camera_config,

            img=self.img.to(*args, **kwargs),
            mask=None if self.mask is None else self.mask.to(*args, **kwargs),
            blending_param=self.blending_param.to(*args, **kwargs),
        )

    def batch_get(self, batch_idxes: tuple[torch.Tensor]) -> typing.Self:
        batch_shape = self.shape

        assert len(batch_idxes) == len(batch_shape)

        return Sample(
            camera_transform=self.camera_transform.batch_get(
                batch_idxes),

            camera_config=self.camera_config,

            img=self.img[batch_idxes],
            mask=None if self.mask is None else
            self.mask[batch_idxes],

            blending_param=self.blending_param.batch_get(batch_idxes)
        )


@beartype
class Dataset(dataset_utils.Dataset):
    def __init__(self, sample: Sample):
        C, H, W = -1, -2, -3

        C, H, W = utils.check_shapes(sample.img, (..., C, H, W))

        batch_shape = utils.broadcast_shapes(
            sample.camera_transform.shape,
            sample.img.shape[:-3],
        )

        if sample.mask is not None:
            utils.check_shapes(sample.mask, (..., H, W))

            batch_shape = utils.broadcast_shapes(
                batch_shape,
                sample.mask.shape[:-2],
            )

        batch_shape = utils.broadcast_shapes(
            batch_shape,
            sample.blending_param.shape,
        )

        self.sample = Sample(
            camera_transform=sample.camera_transform.expand(batch_shape),
            camera_config=sample.camera_config,

            img=sample.img.expand(batch_shape + (C, H, W)),
            mask=None if sample.mask is None else
            sample.mask.expand(batch_shape + (H, W)),

            blending_param=sample.blending_param.expand(batch_shape)
        )

    @property
    def shape(self):
        return self.sample.camera_transform.shape

    @property
    def device(self) -> torch.device:
        return self.sample.device

    def to(self, *args, **kwargs) -> typing.Self:
        self.sample = self.sample.to(*args, **kwargs)
        return self

    def batch_get(self, batch_idxes: tuple[torch.Tensor]) -> Sample:
        return self.sample.batch_get(batch_idxes)
