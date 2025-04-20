from __future__ import annotations

import dataclasses

import torch
from beartype import beartype

from .. import (avatar_utils, camera_utils, gaussian_utils, mesh_utils,
                rendering_utils, transform_utils, utils)
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

    gp_render_img: torch.Tensor  # [..., C, H, W]

    rgb_loss: torch.Tensor  # [...]
    lap_smoothness_loss: torch.Tensor  # [...]
    nor_sim_loss: torch.Tensor  # [...]
    color_diff_loss: torch.Tensor  # [...]
    gp_scale_diff_loss: torch.Tensor  # [...]


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
            (faces_cnt, color_channels_cnt), dtype=utils.FLOAT))
        # [F, C]

        self.gp_opacity = torch.nn.Parameter(torch.zeros(
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
                self.gp_opacity,
            ],
            "lr": base_lr
        })

        return ret

    @property
    def faces_cnt(self) -> int:
        return self.gp_color.shape[0]

    def get_world_gp(
        self,
        mesh_data: mesh_utils.MeshData
    ) -> ModuleWorldGPResult:
        F = self.faces_cnt

        assert mesh_data.faces_cnt == F

        face_coord_result = get_face_coord(mesh_data)
        # face_coord_result.rs[..., F, 3, 3]
        # face_coord_result.ts[..., F, 3]
        # face_coord_result.areas[..., F]

        face_coord_rot_qs = utils.rot_mat_to_quaternion(
            face_coord_result.r, order="WXYZ")

        zeros = utils.zeros_like(self.gp_rot_z, shape=()).expand(F)
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

        world_gp_scale = utils.empty_like(
            scale_factor, shape=scale_factor.shape[:-1] + (F, 3))

        world_gp_scale[..., :, 0] = scale_factor * leaky_clamp(
            x=self.gp_scale_x.exp(),  # clamp local scale
            lb=0.1,
            rb=3.0,
            leaky=0.1,
        )

        world_gp_scale[..., :, 1] = scale_factor * leaky_clamp(
            x=self.gp_scale_y.exp(),  # clamp local scale
            lb=0.1,
            rb=3.0,
            leaky=0.1,
        )

        world_gp_scale[..., :, 2] = leaky_clamp(
            x=scale_factor * self.gp_scale_z.exp(),  # clamp global scale
            lb=0.001,
            rb=0.010,
            leaky=0.005,
        )

        return ModuleWorldGPResult(
            face_coord_result=face_coord_result,
            gp_mean=world_gp_mean,
            gp_rot_q=world_gp_rot_q,
            gp_scale=world_gp_scale,
            gp_color=self.gp_color,
            gp_opacity=self.gp_opacity.exp(),
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

        H, W = utils.check_shapes(
            img, (..., C, H, W),
            mask, (..., 1, H, W),
        )

        avatar_model: avatar_utils.AvatarModel = self.avatar_blender(
            blending_param)

        faces = avatar_model.mesh_graph.f_to_vvv
        # [F, 3]

        vp = avatar_model.vert_pos
        # [..., V, 3]

        world_gp_result = self.get_world_gp(avatar_model.mesh_data)

        gp_render_result = gaussian_utils.render_gaussian(
            camera_config=camera_config,
            camera_transform=camera_transform,

            sh_degree=0,

            bg_color=torch.ones((C,), dtype=world_gp_result.gp_color.dtype),

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

        mesh_graph = avatar_model.mesh_graph

        if not self.training:
            rgb_loss = torch.Tensor()
        else:
            person_masked_img = img * mask + (1 - mask)

            rgb_sum_sq_diff = (
                gp_render_img - person_masked_img).square().sum((-1, -2, -3))
            # [...]

            rgb_loss = rgb_sum_sq_diff / mask.sum((-1, -2, -3))
            # [...]

        if not self.training:
            lap_smoothness_loss = torch.Tensor()
        else:
            lap_smoothness_loss = mesh_graph.calc_l2_cot_lap_smoothness(
                avatar_model.vert_pos)
            # [...]

        if not self.training:
            nor_sim_loss = torch.Tensor()
        else:
            nor_sim = mesh_graph.calc_face_cos_sim(
                world_gp_result.face_coord_result.r[..., :, :3, 2]
                # the z axis of each face
            )
            # [..., FP]

            nor_sim_loss = 1 - nor_sim.square().mean(-1)
            # [...]

        if not self.training:
            color_diff_loss = torch.Tensor()
        else:
            color_diff = mesh_graph.calc_face_cos_sim(
                world_gp_result.gp_color)
            # [..., FP]

            color_diff_loss = color_diff.square().mean(-1)
            # [...]

        if not self.training:
            gp_scale_diff_loss = torch.Tensor()
        else:
            world_gp_x_scale = world_gp_result.gp_scale[..., :, 0]
            world_gp_y_scale = world_gp_result.gp_scale[..., :, 1]

            word_gp_xy_scale = world_gp_x_scale * world_gp_y_scale
            # [..., F]

            word_gp_xy_scale_diff = avatar_model.mesh_graph.calc_face_diff(
                word_gp_xy_scale.unsqueeze(-1))
            # [..., FP, 1]

            gp_scale_diff_loss = word_gp_xy_scale_diff.mean((-1, -2))
            # [...]

        return ModuleForwardResult(
            avatar_model=avatar_model,

            gp_render_img=gp_render_img,

            rgb_loss=rgb_loss,
            lap_smoothness_loss=lap_smoothness_loss,
            nor_sim_loss=nor_sim_loss,
            color_diff_loss=color_diff_loss,
            gp_scale_diff_loss=gp_scale_diff_loss,
        )

    def refresh(self) -> None:
        if hasattr(self, "avatar_blender"):
            self.avatar_blender.refresh()
