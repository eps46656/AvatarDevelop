import torch

from .. import camera_utils, transform_utils, utils


class Dataset:
    def __len__(
        self,
        camera_transform: transform_utils.ObjectTransform,  # [...]
        camera_config: camera_utils.CameraConfig,
        blending_param: object,
        imgs: torch.Tensor,  # [..., C, H, W],
    ):
        self.camera_transform = camera_transform
        self.camera_config = camera_config

    pass
