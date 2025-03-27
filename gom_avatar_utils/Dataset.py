
import dataclasses
import typing

import torch
from beartype import beartype

from .. import camera_utils, dataset_utils, transform_utils, utils


@beartype
@dataclasses.dataclass
class Sample:
    camera_transform: transform_utils.ObjectTransform  # [...]
    camera_config: camera_utils.CameraConfig

    img: torch.Tensor  # [..., C, H, W]
    mask: typing.Optional[torch.Tensor]  # [..., H, W]
    blending_param: object


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

            blending_param=sample.blending_param.Expand(batch_shape)
        )

    @property
    def shape(self):
        return self.sample.camera_transform.shape

    def batch_get(self, batch_idxes: tuple[torch.Tensor]):
        batch_shape = self.sample.camera_transform.shape

        assert len(batch_idxes) == len(batch_shape)

        return Sample(
            camera_transform=self.sample.camera_transform.BatchGet(
                batch_idxes),

            camera_config=self.sample.camera_config,

            img=self.sample.img[batch_idxes],
            mask=None if self.sample.mask is None else
            self.sample.mask[batch_idxes],

            blending_param=self.sample.blending_param.BatchGet(batch_idxes)
        )
