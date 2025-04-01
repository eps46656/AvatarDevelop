import dataclasses
import typing

import torch
from beartype import beartype

from .. import (avatar_utils, camera_utils, gaussian_utils, transform_utils,
                utils)
from .utils import get_face_coord


@beartype
@dataclasses.dataclass
class ModuleForwardResult:
    avatar_model: avatar_utils.AvatarModel

    rendered_img: torch.Tensor  # [..., C, H, W]

    rgb_loss: float | torch.Tensor
    lap_smoothing_loss: float | torch.Tensor
    normal_sim_loss: float | torch.Tensor
    color_diff_loss: float | torch.Tensor


@beartype
class Module(torch.nn.Module):
    def __init__(
        self,
        avatar_blender: avatar_utils.AvatarBlender,
        color_channels_cnt: int,
    ):
        super().__init__()

        self.avatar_blender: avatar_utils.AvatarBlender = avatar_blender

        avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blender.get_avatar_model()

        faces_cnt = avatar_model.faces_cnt
        assert 0 < faces_cnt

        assert 1 <= color_channels_cnt

        gp_rot_qs = torch.empty(
            (faces_cnt, 4),
            dtype=utils.FLOAT)
        # [F, 4] quaternion wxyz

        gp_rot_qs[:, 0] = 1
        gp_rot_qs[:, 1:] = 0

        self.gp_rot_qs = torch.nn.Parameter(gp_rot_qs)
        # [F, 4]

        self.gp_log_scales = torch.nn.Parameter(torch.ones(
            (faces_cnt, 3),
            dtype=utils.FLOAT))
        # [F, 3]

        self.gp_colors = torch.nn.Parameter(torch.rand(
            (faces_cnt, color_channels_cnt),
            dtype=utils.FLOAT))
        # [F, C]

        self.gp_log_opacities = torch.nn.Parameter(torch.zeros(
            (faces_cnt, 1),
            dtype=utils.FLOAT))
        # [F, 1]

    def to(self, *args, **kwargs) -> typing.Self:
        super().to(*args, **kwargs)

        self.avatar_blender = self.avatar_blender.to(*args, **kwargs)

        return self

    def get_param_groups(self, base_lr: float) -> list[dict]:
        ret = utils.get_param_groups(self.avatar_blender, base_lr)

        ret.append({
            "params": [
                self.gp_rot_qs,
                self.gp_colors,
                self.gp_log_scales,
                self.gp_log_opacities,
            ],
            "lr": base_lr
        })

        return ret

    def forward(
        self,
        camera_config: camera_utils.CameraConfig,
        camera_transform: transform_utils.ObjectTransform,
        img: torch.Tensor,  # [..., C, H, W]
        mask: torch.Tensor,  # [..., H, W]
        blending_param: object,
    ):
        device = next(self.parameters()).device

        color_channels_cnt = self.gp_colors.shape[-1]

        H, W = -1, -2

        H, W = utils.check_shapes(img, (..., color_channels_cnt, H, W))

        avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blender(blending_param)

        faces = avatar_model.faces
        # [F, 3]

        F = faces.shape[0]

        vertex_positions = avatar_model.vertex_positions
        # [..., V, 3]

        vps_a = vertex_positions[..., faces[:, 0], :]
        vps_b = vertex_positions[..., faces[:, 1], :]
        vps_c = vertex_positions[..., faces[:, 2], :]
        # [..., F, 3]

        face_rs, face_ts = get_face_coord(vps_a, vps_b, vps_c)
        # face_rs[..., F, 3, 3]
        # face_ts[..., F, 3]

        gp_scales = torch.exp(self.gp_log_scales)
        # [F, 3]

        gp_rot_mats = utils.quaternion_to_rot_mat(
            self.gp_rot_qs,
            order="WXYZ",
            out_shape=(3, 3),
        )
        # [F, 3, 3]

        gp_rs = torch.stack([
            gp_rot_mats[:, :, 0] * gp_scales[:, 0],
            gp_rot_mats[:, :, 1] * gp_scales[:, 1],
            gp_rot_mats[:, :, 2] * gp_scales[:, 2],
        ], -2)

        gp_ts = torch.zeros((3,), dtype=gp_rs.dtype, device=gp_rs.device)

        global_gp_rs, global_gp_ts = utils.merge_rt(
            face_rs, face_ts, gp_rs, gp_ts)
        # global_gp_rs[..., F, 3, 3]
        # global_gp_ts[..., F, 3]

        global_gp_means = global_gp_ts
        # [..., F, 3]

        global_gp_scales = utils.vector_norm(global_gp_rs, -2)
        # [..., F, 3, 3] -> [..., F, 1, 3] -> [..., F, 3]

        global_gp_rot_mats = global_gp_rs / global_gp_scales.unsqueeze(-2)
        # [..., F, 3, 3] * [..., F, 1, 3] -> [..., F, 3, 3]

        global_gp_rot_qs = utils.rot_mat_to_quaternion(
            global_gp_rot_mats,
            order="WXYZ",
        )
        # [..., F, 4] wxyz

        gp_colors = self.gp_colors

        rendered_result = gaussian_utils.render_gaussian(
            camera_config=camera_config,
            camera_transform=camera_transform,

            sh_degree=0,

            bg_color=torch.ones((color_channels_cnt,),
                                dtype=utils.FLOAT, device=device),

            gp_means=global_gp_means,
            gp_rots=global_gp_rot_qs,
            gp_scales=global_gp_scales,

            gp_shs=None,
            gp_colors=gp_colors,

            gp_opacities=torch.exp(self.gp_log_opacities),

            device=device,
        )  # [...]

        rendered_img = rendered_result.colors
        # [..., C, H, W]

        mesh_data = avatar_model.mesh_data

        if not self.training:
            rgb_loss = 0.0
        else:
            mask_ = mask.unsqueeze(-3)

            white_img = torch.ones_like(
                img, dtype=utils.FLOAT, device=img.device)

            masked_img = white_img * (1 - mask_) + img * mask_

            rgb_loss = (rendered_img - masked_img).square().mean()

        if not self.training:
            lap_smoothing_loss = 0.0
        else:
            lap_smoothing_loss = mesh_data.calc_lap_smoothing_loss(
                avatar_model.vertex_positions)

        if not self.training:
            normal_sim_loss = 0.0
        else:
            normal_sim_loss = 0.0

            """
            normal_sim = mesh_data.calc_face_cos_sim(
                face_coord_result.Ts[..., :3, 2]
                # the z axis of each face
            )

            # [..., FP]

            normal_sim_loss = 1 - normal_sim.mean()"
            """

        if not self.training:
            color_diff_loss = 0.0
        else:
            """
            color_diff = mesh_data.calc_face_cos_sims(gp_colors)
            # [..., FP]

            color_diff_loss = color_diff.square().mean()
            """

            color_diff_loss = 0.0

        return ModuleForwardResult(
            avatar_model=avatar_model,
            rendered_img=rendered_img,
            rgb_loss=rgb_loss,
            lap_smoothing_loss=lap_smoothing_loss,
            normal_sim_loss=normal_sim_loss,
            color_diff_loss=color_diff_loss,
        )
