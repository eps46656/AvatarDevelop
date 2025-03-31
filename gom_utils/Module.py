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
    lap_loss: float | torch.Tensor
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

        self.gp_scales = torch.nn.Parameter(torch.ones(
            (faces_cnt, 3),
            dtype=utils.FLOAT) * 0.01)
        # [F, 3]

        self.gp_log_colors = torch.nn.Parameter(torch.rand(
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

    def forward(
        self,
        camera_config: camera_utils.CameraConfig,
        camera_transform: transform_utils.ObjectTransform,
        img: torch.Tensor,  # [..., C, H, W]
        mask: torch.Tensor,  # [..., H, W]
        blending_param: object,
    ):
        device = next(self.parameters()).device

        color_channels_cnt = self.gp_log_colors.shape[-1]

        H, W = -1, -2

        H, W = utils.check_shapes(img, (..., color_channels_cnt, H, W))

        avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blender(blending_param)

        faces = avatar_model.faces
        # [F, 3]

        vertex_positions = avatar_model.vertex_positions
        # [..., V, 3]

        vertex_positions_a = vertex_positions[..., faces[:, 0], :]
        vertex_positions_b = vertex_positions[..., faces[:, 1], :]
        vertex_positions_c = vertex_positions[..., faces[:, 2], :]
        # [..., F, 3]

        """
        face_coord_result = get_face_coord(
            vertex_positions_a, vertex_positions_b, vertex_positions_c)

        face_coord_rot_qs = utils.rot_mat_to_quaternion(
            face_coord_result.Ts[..., :3, :3],
            order="WXYZ",
        )
        """
        # [..., F, 4] wxyz

        """
        utils.check_almost_zeros(
            face_coord_result.Ts[..., :3, :3].det() - 1,
            5e-3,
        )
        """

        gp_global_means = (
            vertex_positions_a +
            vertex_positions_b +
            vertex_positions_c
        ) / 3
        # [..., F, 3]

        gp_global_rot_qs = self.gp_rot_qs

        gp_global_scales = self.gp_scales
        # [..., F, 3]

        gp_colors = torch.exp(self.gp_log_colors)

        rendered_result = gaussian_utils.render_gaussian(
            camera_config=camera_config,
            camera_transform=camera_transform,

            sh_degree=0,

            bg_color=torch.ones((color_channels_cnt,),
                                dtype=utils.FLOAT, device=device),

            gp_means=gp_global_means,
            gp_rots=gp_global_rot_qs,
            gp_scales=gp_global_scales,

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
            lap_loss = 0.0
        else:
            lap_diff = mesh_data.calc_lap_diffs(avatar_model.vertex_positions)
            # [..., V, 3]

            lap_loss = lap_diff.square().mean()

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
            color_diff = mesh_data.calc_face_cos_sims(gp_colors)
            # [..., FP]

            color_diff_loss = color_diff.square().mean()

        return ModuleForwardResult(
            avatar_model=avatar_model,
            rendered_img=rendered_img,
            rgb_loss=rgb_loss,
            lap_loss=lap_loss,
            normal_sim_loss=normal_sim_loss,
            color_diff_loss=color_diff_loss,
        )
