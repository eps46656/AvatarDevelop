from __future__ import annotations

import dataclasses

import torch
from beartype import beartype

import pytorch3d.renderer

from .. import (avatar_utils, camera_utils, gaussian_utils, rendering_utils,
                transform_utils, utils)
from .utils import FaceCoordResult, get_face_coord


@beartype
@dataclasses.dataclass
class ModuleWorldGPResult:
    face_coord_result: FaceCoordResult
    gp_mean: torch.Tensor  # [..., F, 3]
    gp_rot_q: torch.Tensor  # [..., F, 4]
    gp_scale: torch.Tensor  # [..., F, 3]
    gp_color: torch.Tensor  # [..., F, 3]
    gp_opacity: torch.Tensor  # [..., F, 1]


@beartype
@dataclasses.dataclass
class ModuleForwardResult:
    avatar_model: avatar_utils.AvatarModel

    mesh_ras_result: pytorch3d.renderer.mesh.rasterizer.Fragments
    mesh_proj_area: torch.Tensor  # [...]

    gp_render_img: torch.Tensor  # [..., C, H, W]

    rgb_loss: float | torch.Tensor
    lap_smoothing_loss: float | torch.Tensor
    nor_sim_loss: float | torch.Tensor
    color_diff_loss: float | torch.Tensor


@beartype
def leaky_clamp(
    x: torch.Tensor,
    lb: float,
    rb: float,
    leaky: float = 0.05,
) -> torch.Tensor:

    assert lb < rb
    assert 0 <= leaky

    """

    f(x)=(
    Sigmoid(
        (
            (x-xcenter)/(xwidth)
        )*4
    )-0.5) xwidth (1-leaky)+leaky (x-xcenter)+xcenter

    """

    center = (lb + rb) / 2
    width = rb - lb
    k = width * (1 - leaky)

    y = k * ((x - center) * (4 / width)).sigmoid() + \
        leaky * x + (center - leaky * center - k * 0.5)

    return y


@beartype
class Module(torch.nn.Module):
    def __init__(
        self,
        avatar_blender: avatar_utils.AvatarBlender,
        color_channels_cnt: int,
    ):
        super().__init__()

        self.avatar_blender: avatar_utils.AvatarBlender = avatar_blender

        avatar_model: avatar_utils.AvatarModel = self.avatar_blender.get_avatar_model()

        faces_cnt = avatar_model.faces_cnt
        assert 0 < faces_cnt

        assert 1 <= color_channels_cnt

        self.gp_rot_z = torch.nn.Parameter(torch.zeros(
            (faces_cnt,), dtype=utils.FLOAT))

        self.gp_scale_x = torch.nn.Parameter(torch.full(
            (faces_cnt,), -3, dtype=utils.FLOAT))

        self.gp_scale_y = torch.nn.Parameter(torch.full(
            (faces_cnt,), -3, dtype=utils.FLOAT))

        self.gp_scale_z = torch.nn.Parameter(torch.full(
            (faces_cnt,), -3, dtype=utils.FLOAT))

        self.gp_color = torch.nn.Parameter(torch.rand(
            (faces_cnt, color_channels_cnt),
            dtype=utils.FLOAT))
        # [F, C]

        self.gp_log_opacity = torch.nn.Parameter(torch.zeros(
            (faces_cnt,), dtype=utils.FLOAT))
        # [F, 1]

    def to(self, *args, **kwargs) -> Module:
        super().to(*args, **kwargs)

        self.avatar_blender = self.avatar_blender.to(*args, **kwargs)

        return self

    def get_param_groups(self, base_lr: float) -> list[dict]:
        ret = utils.get_param_groups(self.avatar_blender, base_lr)

        ret.append({
            "params": [
                self.gp_rot_z,
                self.gp_scale_x,
                self.gp_scale_y,
                self.gp_scale_z,
                self.gp_color,
                self.gp_log_opacity,
            ],
            "lr": base_lr
        })

        return ret

    @property
    def faces_cnt(self) -> int:
        return self.gp_color.shape[0]

    def get_world_gp(
        self,
        vert_pos_a: torch.Tensor,  # [..., F, 3]
        vert_pos_b: torch.Tensor,  # [..., F, 3]
        vert_pos_c: torch.Tensor,  # [..., F, 3]
    ) -> ModuleWorldGPResult:
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
            face_coord_result.r, order="WXYZ")

        zeros = torch.zeros((), dtype=self.gp_rot_z.dtype,
                            device=self.gp_rot_z.device).expand(F)
        # [F]

        half_gp_rot_thetas = self.gp_rot_z * 0.5
        # [F]

        local_gp_rot_qs = torch.stack([
            half_gp_rot_thetas.cos(),  # W
            zeros,  # X
            zeros,  # Y
            half_gp_rot_thetas.sin(),  # Z
        ], -1)
        # [F, 4]

        utils.check_shapes(
            local_gp_rot_qs, (F, 4)
        )

        world_gp_mean = face_coord_result.t
        # [..., F, 3]

        world_gp_rot_q = utils.quaternion_mul(
            face_coord_rot_qs, local_gp_rot_qs,
            order_1="WXYZ",
            order_2="WXYZ",
            order_out="WXYZ",
        )
        # [..., F, 4] wxyz

        scale_factor = face_coord_result.area.sqrt()
        # [..., F]

        world_gp_scale = torch.empty(
            scale_factor.shape[:-1] + (F, 3),
            dtype=scale_factor.dtype, device=scale_factor.device)

        world_gp_scale[..., :, 0] = scale_factor * utils.smooth_clamp(
            x=torch.exp(self.gp_scale_x),  # clamp local scale
            x_lb=0.1,
            x_rb=3.0,

            slope_l=0.0,
            slope_r=0.005,
        )

        world_gp_scale[..., :, 1] = scale_factor * utils.smooth_clamp(
            x=torch.exp(self.gp_scale_y),  # clamp local scale
            x_lb=0.1,
            x_rb=3.0,

            slope_l=0.0,
            slope_r=0.005,
        )

        world_gp_scale[..., :, 2] = utils.smooth_clamp(
            x=scale_factor * torch.exp(self.gp_scale_z),  # clamp global scale
            x_lb=0.001,
            x_rb=0.01,

            slope_l=0.0,
            slope_r=0.005,
        )

        return ModuleWorldGPResult(
            face_coord_result=face_coord_result,
            gp_mean=world_gp_mean,
            gp_rot_q=world_gp_rot_q,
            gp_scale=world_gp_scale,
            gp_color=self.gp_color,
            gp_opacity=torch.exp(self.gp_log_opacity),
        )

    def forward(
        self,
        camera_config: camera_utils.CameraConfig,
        camera_transform: transform_utils.ObjectTransform,
        img: torch.Tensor,  # [..., C, H, W]
        mask: torch.Tensor,  # [..., H, W]
        blending_param: object,
    ) -> ModuleForwardResult:
        device = next(self.parameters()).device

        C = self.gp_color.shape[-1]

        H, W = -1, -2

        H, W = utils.check_shapes(img, (..., C, H, W))

        avatar_model: avatar_utils.AvatarModel = self.avatar_blender(
            blending_param)

        faces = avatar_model.mesh_data.f_to_vvv
        # [F, 3]

        vp = avatar_model.vert_pos
        # [..., V, 3]

        world_gp_result = self.get_world_gp(
            vp[..., faces[:, 0], :],
            vp[..., faces[:, 1], :],
            vp[..., faces[:, 2], :],
        )

        mesh_ras_result = rendering_utils.rasterize_mesh(
            vert_pos=avatar_model.vert_pos,
            faces=avatar_model.mesh_data.f_to_vvv,
            camera_config=camera_config,
            camera_transform=camera_transform,
            faces_per_pixel=1,
        )

        # mesh_ras_result.pix_to_face[B, H, W, 1]
        # mesh_ras_result.bary_coords[B, H, W, 1, 3]
        # mesh_ras_result.pix_dists[B, H, W, 1]

        mesh_proj_area = (mesh_ras_result.pix_to_face != -1).count_nonzero(
            dim=(1, 2, 3))
        # [B]

        gp_render_result = gaussian_utils.render_gaussian(
            camera_config=camera_config,
            camera_transform=camera_transform,

            sh_degree=0,

            bg_color=torch.ones((C,),
                                dtype=world_gp_result.gp_color.dtype),

            gp_mean=world_gp_result.gp_mean,
            gp_rot_q=world_gp_result.gp_rot_q,
            gp_scale=world_gp_result.gp_scale,

            gp_sh=None,
            gp_color=world_gp_result.gp_color,

            gp_opacity=world_gp_result.gp_opacity,

            device=device,
        )  # [...]

        gp_render_img = gp_render_result.colors
        # [..., C, H, W]

        mesh_data = avatar_model.mesh_data

        mask = mask.unsqueeze(-3)
        # [..., 1, H, W]

        if not self.training:
            rgb_loss = torch.Tensor()
        else:
            person_masked_img = img * mask + (1 - mask)

            rgb_sum_sq_diff = (
                gp_render_img - person_masked_img).square().sum()
            # [...]

            rgb_loss = rgb_sum_sq_diff / mesh_proj_area.sum()

        if not self.training:
            lap_smoothing_loss = torch.Tensor()
        else:
            lap_smoothing_loss = mesh_data.calc_lap_smoothing_loss(
                avatar_model.vert_pos)

        if not self.training:
            nor_sim_loss = torch.Tensor()
        else:
            nor_sim_loss = torch.Tensor()

            nor_sim = mesh_data.calc_face_cos_sims(
                world_gp_result.face_coord_result.r[..., :, :3, 2]
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

        return ModuleForwardResult(
            avatar_model=avatar_model,

            mesh_ras_result=mesh_ras_result,
            mesh_proj_area=mesh_proj_area,

            gp_render_img=gp_render_img,

            rgb_loss=rgb_loss,
            lap_smoothing_loss=lap_smoothing_loss,
            nor_sim_loss=nor_sim_loss,
            color_diff_loss=color_diff_loss,
        )

    def refresh(self) -> None:
        if hasattr(self, "avatar_blender"):
            self.avatar_blender.refresh()
