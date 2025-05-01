from __future__ import annotations

import dataclasses
import os
import typing

import torch
from beartype import beartype

from .. import smplx_utils, utils


@beartype
@dataclasses.dataclass
class GARTResult:
    state_dict: typing.Mapping[str, typing.Any]

    body_shape: torch.Tensor  # [BS] smpl model's body shapes cnt

    gp_mean: torch.Tensor  # [N, 3]
    gp_rot_q: torch.Tensor  # [N, 4]
    gp_scale: torch.Tensor  # [N, 3]
    gp_opacity: torch.Tensor  # [N]
    gp_color: torch.Tensor  # [N, 3]


@beartype
def read_gart_result(
    gart_log_dir: os.PathLike,
    train_name: str,
    dtype: typing.Optional[torch.dtype],
    device: torch.device,
):
    gart_log_dir = utils.to_pathlib_path(gart_log_dir)

    train_dir = gart_log_dir / train_name

    dirs = [name for name in utils.to_pathlib_path(train_dir).glob("*")
            if name.is_dir()]

    assert len(dirs) == 1

    train_data_dir = dirs[0]

    state_dict = torch.load(
        train_data_dir / "model.pth",
        map_location=utils.CPU_DEVICE,
    )

    body_shape: torch.Tensor = state_dict["template.init_beta"]
    # [BS]

    gp_mean: torch.Tensor = state_dict["_xyz"]  # [N, 3]
    gp_rot_q: torch.Tensor = state_dict["_rotation"]  # [N, 4]
    scale: torch.Tensor = state_dict["_scaling"]  # [N, 1]
    opacity: torch.Tensor = state_dict["_opacity"]  # [N, 1]
    sph: torch.Tensor = state_dict["_features_dc"]  # [N, 3]

    N = -1

    N = utils.check_shapes(
        body_shape, (smplx_utils.smpl_model_config.body_shapes_cnt,),
        gp_mean, (N, 3),
        gp_rot_q, (N, 4),
        scale, (N, 3),
        opacity, (N, 1),
        sph, (N, 3),
    )

    gp_scale = scale.sigmoid()
    gp_opacity = opacity.sigmoid()[:, 0]
    gp_color = (0.28209479177387814 * sph + 0.5).clamp(0.0, None)

    return GARTResult(
        state_dict=state_dict,

        body_shape=body_shape.to(device),

        gp_mean=gp_mean.to(device, dtype),
        gp_rot_q=gp_rot_q.to(device, dtype),
        gp_scale=gp_scale.to(device, dtype),
        gp_opacity=gp_opacity.to(device, dtype),
        gp_color=gp_color.to(device, dtype),
    )
