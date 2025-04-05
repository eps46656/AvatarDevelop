from __future__ import annotations

import dataclasses

import torch
from beartype import beartype

from .. import (avatar_utils, camera_utils, gaussian_utils, transform_utils,
                utils)
from .utils import FaceCoordResult, get_face_coord


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

    def to(self, *args, **kwargs) -> Module:
        super().to(*args, **kwargs)

        self.avatar_blender = self.avatar_blender.to(*args, **kwargs)

        return self

    def get_param_groups(self, base_lr: float) -> list[dict]:
        ret = utils.get_param_groups(self.avatar_blender, base_lr)

        ret.append({
            "params": [
                self.gp_rot_qs,
                self.gp_scales,
                self.gp_colors,
                self.gp_log_opacities,
            ],
            "lr": base_lr
        })

        return ret

    @property
    def faces_cnt(self) -> int:
        return self.gp_rot_qs.shape[0]

    @beartype
    @dataclasses.dataclass
    class WorldGPResult:
        face_coord_result: FaceCoordResult
        gp_means: torch.Tensor  # [..., F, 3]
        gp_rot_qs: torch.Tensor  # [..., F, 4]
        gp_scales: torch.Tensor  # [..., F, 3]
        gp_colors: torch.Tensor  # [..., F, 3]
        gp_opacities: torch.Tensor  # [..., F, 1]

    def get_world_gp(
        self,
        vert_pos_a: torch.Tensor,  # [..., F, 3]
        vert_pos_b: torch.Tensor,  # [..., F, 3]
        vert_pos_c: torch.Tensor,  # [..., F, 3]
    ) -> WorldGPResult:
        F = self.faces_cnt

        utils.check_shapes(
            vert_pos_a, (..., F, 3),
            vert_pos_b, (..., F, 3),
            vert_pos_c, (..., F, 3),
        )

        face_coord_result = get_face_coord(vert_pos_a, vert_pos_b, vert_pos_c)
        # face_coord_result.rs[..., F, 3, 3]
        # face_coord_result.ts[..., F, 3]
        # face_coord_result.areas[..., F]

        face_coord_rot_qs = utils.rot_mat_to_quaternion(
            face_coord_result.rs, order="WXYZ")

        world_gp_means = face_coord_result.ts
        # [..., F, 3]

        world_gp_rot_qs = utils.quaternion_mul(
            face_coord_rot_qs, self.gp_rot_qs,
            order_1="WXYZ",
            order_2="WXYZ",
            order_out="WXYZ",
        )
        # [..., F, 4] wxyz

        k = face_coord_result.areas.sqrt()
        # [..., F]

        world_gp_scales = torch.empty(
            k.shape[:-1] + (F, 3), dtype=k.dtype, device=k.device)

        world_gp_scales[..., :, 0] = k * self.gp_scales[:, 0]
        world_gp_scales[..., :, 1] = k * self.gp_scales[:, 1]
        world_gp_scales[..., :, 2] = k * self.gp_scales[:, 2] * 1e-2

        return Module.WorldGPResult(
            face_coord_result=face_coord_result,
            gp_means=world_gp_means,
            gp_rot_qs=world_gp_rot_qs,
            gp_scales=world_gp_scales,
            gp_colors=self.gp_colors,
            gp_opacities=torch.exp(self.gp_log_opacities),
        )

    @beartype
    @dataclasses.dataclass
    class ForwardResult:
        avatar_model: avatar_utils.AvatarModel

        rendered_img: torch.Tensor  # [..., C, H, W]

        rgb_loss: float | torch.Tensor
        lap_smoothing_loss: float | torch.Tensor
        nor_sim_loss: float | torch.Tensor
        color_diff_loss: float | torch.Tensor

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

        faces = avatar_model.mesh_data.f_to_vvv
        # [F, 3]

        vp = avatar_model.vert_pos
        # [..., V, 3]

        world_gp_result = self.get_world_gp(
            vp[..., faces[:, 0], :],
            vp[..., faces[:, 1], :],
            vp[..., faces[:, 2], :],
        )

        rendered_result = gaussian_utils.render_gaussian(
            camera_config=camera_config,
            camera_transform=camera_transform,

            sh_degree=0,

            bg_color=torch.ones((color_channels_cnt,),
                                dtype=world_gp_result.gp_colors.dtype),

            gp_means=world_gp_result.gp_means,
            gp_rots=world_gp_result.gp_rot_qs,
            gp_scales=world_gp_result.gp_scales,

            gp_shs=None,
            gp_colors=world_gp_result.gp_colors,

            gp_opacities=world_gp_result.gp_opacities,

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
                avatar_model.vert_pos)

        if not self.training:
            nor_sim_loss = 0.0
        else:
            nor_sim_loss = 0.0

            nor_sim = mesh_data.calc_face_cos_sims(
                world_gp_result.face_coord_result.rs[..., :, :3, 2]
                # the z axis of each face
            )

            # [..., FP]

            nor_sim_loss = 1 - nor_sim.mean()

        if not self.training:
            color_diff_loss = 0.0
        else:
            """
            color_diff = mesh_data.calc_face_cos_sims(gp_colors)
            # [..., FP]

            color_diff_loss = color_diff.square().mean()
            """

            color_diff_loss = 0.0

        return Module.ForwardResult(
            avatar_model=avatar_model,
            rendered_img=rendered_img,
            rgb_loss=rgb_loss,
            lap_smoothing_loss=lap_smoothing_loss,
            nor_sim_loss=nor_sim_loss,
            color_diff_loss=color_diff_loss,
        )
