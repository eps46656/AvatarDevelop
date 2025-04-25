from __future__ import annotations

import typing

import numpy as np
import torch
from beartype import beartype

from .. import camera_utils, smplx_utils, transform_utils, utils, vision_utils


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
class Dataset(torch.utils.data.Dataset):
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
    def shape(self) -> torch.Size:
        return self.sample.camera_transform.shape

    @property
    def device(self) -> torch.device:
        return self.sample.device

    def __len__(self) -> int:
        return self.shape.numel()

    def __getitem__(self, idx):
        cur_sample = self.sample[idx]

        camera_proj_matrix: torch.Tensor = camera_utils.make_proj_mat(
            camera_config=cur_sample.camera_config,
            camera_transform=cur_sample.camera_transform,
            convention=camera_utils.Convention.OpenCV,
            traget_coord=camera_utils.Coord.Screen,
            dtype=torch.float32,
        )

        blending_param = cur_sample.blending_param

        assert isinstance(blending_param, smplx_utils.BlendingParam)

        body_shape = blending_param.body_shape[
            *((0,) * (len(blending_param.shape) - 1)), :]

        body_pose = blending_param.body_pose.reshape(23, 3)

        global_transl = blending_param.global_transl

        ret = {
            "rgb": cur_sample.img.numpy().astype(np.float32),
            "mask": cur_sample.mask.numpy().astype(np.float32),
            "K": camera_proj_matrix.numpy().astype(np.float32),
            "smpl_beta": body_shape.numpy().astype(np.float32),
            "smpl_pose": body_pose.numpy().astype(np.float32),
            "smpl_trans": global_transl.numpy().astype(np.float32),
            "idx": idx,
        }

        meta_info = {
            "video": "",
            "viz_id": f"video_dataidx{idx}",
        }

        return ret, meta_info

    def to(self, *args, **kwargs) -> Dataset:
        self.sample = self.sample.to(*args, **kwargs)
        return self
