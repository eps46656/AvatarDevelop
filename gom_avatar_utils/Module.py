from __future__ import annotations

import collections
import dataclasses
import typing

import torch
from beartype import beartype

from .. import (avatar_utils, camera_utils, gaussian_utils, mesh_utils,
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

    gp_render_img: torch.Tensor  # [..., C, H, W]

    img_diff: torch.Tensor  # [...]
    lap_diff: torch.Tensor  # [...]
    nor_sim: torch.Tensor  # [...]
    edge_var: torch.Tensor  # [...]
    gp_color_diff: torch.Tensor  # [...]
    gp_scale_diff: torch.Tensor  # [...]


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
            (faces_cnt,), 0, dtype=utils.FLOAT))

        self.gp_scale_y = torch.nn.Parameter(torch.full(
            (faces_cnt,), 0, dtype=utils.FLOAT))

        self.gp_scale_z = torch.nn.Parameter(torch.full(
            (faces_cnt,), 0, dtype=utils.FLOAT))

        self.gp_color = torch.nn.Parameter(torch.rand(
            (faces_cnt, color_channels_cnt), dtype=utils.FLOAT))

        self.gp_opacity = torch.nn.Parameter(torch.zeros(
            (faces_cnt,), dtype=utils.FLOAT))

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

    def state_dict(self) -> collections.OrderedDict:
        return collections.OrderedDict([
            ("avatar_blender", self.avatar_blender.state_dict()),

            ("gp_rot_z", self.gp_rot_z),

            ("gp_scale_x", self.gp_scale_x),
            ("gp_scale_y", self.gp_scale_y),
            ("gp_scale_z", self.gp_scale_z),

            ("gp_color", self.gp_color),
            ("gp_opacity", self.gp_opacity),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.avatar_blender.load_state_dict(
            state_dict["avatar_blender"])

        self.gp_rot_z = torch.nn.Parameter(
            state_dict["gp_rot_z"].to(self.gp_rot_z, copy=True))

        self.gp_scale_x = torch.nn.Parameter(
            state_dict["gp_scale_x"].to(self.gp_scale_x, copy=True))

        self.gp_scale_y = torch.nn.Parameter(
            state_dict["gp_scale_y"].to(self.gp_scale_y, copy=True))

        self.gp_scale_z = torch.nn.Parameter(
            state_dict["gp_scale_z"].to(self.gp_scale_z, copy=True))

        self.gp_color = torch.nn.Parameter(
            state_dict["gp_color"].to(self.gp_color, copy=True))

        self.gp_opacity = torch.nn.Parameter(
            state_dict["gp_opacity"].to(self.gp_opacity, copy=True))

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> Module:
        mesh_subdivide_result = self.avatar_blender.subdivide(
            target_edges=target_edges,
            target_faces=target_faces,
        )

        face_src_table = mesh_subdivide_result.face_src_table

        self.gp_rot_z = torch.nn.Parameter(self.gp_rot_z[face_src_table])
        # [F_]

        self.gp_scale_x = torch.nn.Parameter(self.gp_scale_x[face_src_table])
        # [F_]

        self.gp_scale_y = torch.nn.Parameter(self.gp_scale_y[face_src_table])
        # [F_]

        self.gp_scale_z = torch.nn.Parameter(self.gp_scale_z[face_src_table])
        # [F_]

        self.gp_color = torch.nn.Parameter(self.gp_color[face_src_table])
        # [F_, C]

        self.gp_opacity = torch.nn.Parameter(self.gp_opacity[face_src_table])
        # [F_]

        return self

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

        zeros = utils.zeros(like=self.gp_rot_z, shape=()).expand(F)
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

        world_gp_scale_x = scale_factor * utils.smooth_clamp(
            x=self.gp_scale_x,  # clamp local scale
            lb=2.0 / 3,
            rb=4.0 / 3,
        )
        # [..., F]

        world_gp_scale_y = scale_factor * utils.smooth_clamp(
            x=self.gp_scale_y,  # clamp local scale
            lb=2.0 / 3,
            rb=4.0 / 3,
        )
        # [..., F]

        world_gp_scale_z = utils.smooth_clamp(
            x=scale_factor * self.gp_scale_z,  # clamp global scale
            lb=20e-3 / 3,
            rb=30e-3 / 3,
        )
        # [..., F]

        world_gp_scale = torch.stack([
            world_gp_scale_x, world_gp_scale_y, world_gp_scale_z], -1)
        # [..., F, 3]

        world_gp_opacity = utils.smooth_clamp(
            x=self.gp_opacity,
            lb=0.8,
            rb=1.0,
        )

        return ModuleWorldGPResult(
            face_coord_result=face_coord_result,
            gp_mean=world_gp_mean,
            gp_rot_q=world_gp_rot_q,
            gp_scale=world_gp_scale,
            gp_color=self.gp_color,
            gp_opacity=world_gp_opacity,
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

        world_gp_result = self.get_world_gp(avatar_model.mesh_data)

        normal_bg_color = 1.0
        abnormal_bg_color = 2.0

        gp_render_result = gaussian_utils.render_gaussian(
            camera_config=camera_config,
            camera_transform=camera_transform,

            bg_color=utils.full(
                normal_bg_color,
                shape=(C,),
                dtype=world_gp_result.gp_color.dtype,
            ),

            gp_mean=world_gp_result.gp_mean,
            gp_rot_q=world_gp_result.gp_rot_q,
            gp_scale=world_gp_result.gp_scale,

            gp_sh=None,
            gp_color=world_gp_result.gp_color,

            gp_opacity=world_gp_result.gp_opacity,

            device=device,
        )  # [...]

        gp_render_img = gp_render_result.color
        # [..., C, H, W]

        if not self.training:
            ret_img_diff = torch.Tensor()
        else:
            normal_person_masked_img = img * mask + \
                normal_bg_color * (1 - mask)

            abnormal_person_masked_img = img * mask + \
                abnormal_bg_color * (1 - mask)

            raw_rgb_sum_sq_diff_base = (
                normal_person_masked_img - abnormal_person_masked_img
            ).square().sum((-3, -2, -1))
            # [...]

            raw_rgb_sum_sq_diff = (
                gp_render_img - abnormal_person_masked_img
            ).square().sum((-3, -2, -1))
            # [...]

            rgb_sum_sq_diff = raw_rgb_sum_sq_diff - raw_rgb_sum_sq_diff_base
            # [...]

            ret_img_diff = rgb_sum_sq_diff / mask.sum((-3, -2, -1))
            # [...]

        if not self.training:
            ret_lap_diff = torch.Tensor()
        else:
            raw_lap_diff = avatar_model.mesh_data.uni_lap_diff
            # [..., V, 3]

            """
            lap_diff_norm = utils.vec_norm(raw_lap_diff.detach(), keepdim=True)
            # [..., V, 1]

            lap_diff = raw_lap_diff * (
                lap_diff_clamp_norm /
                lap_diff_norm.clamp(lap_diff_clamp_norm, None)
            )
            # [..., V, 3]
            """

            lap_diff = raw_lap_diff

            ret_lap_diff = lap_diff.square().sum((-2, -1)) / lap_diff.shape[-2]
            # [...]

        if not self.training:
            ret_nor_sim = torch.Tensor()
        else:
            nor_sim = avatar_model.mesh_data.face_norm_cos_sim
            # [..., FP]

            ret_nor_sim = (1 - nor_sim).square().mean(-1)
            # [...]

        if not self.training:
            ret_edge_var = torch.Tensor()
        else:
            rel_edge_var = avatar_model.mesh_data.face_edge_rel_var
            # [..., F]

            ret_edge_var = rel_edge_var.mean(-1)
            # [...]

        if not self.training:
            ret_gp_color_diff = torch.Tensor()
        else:
            gp_color_diff = avatar_model.mesh_graph.calc_face_cos_sim(
                world_gp_result.gp_color)
            # [..., FP]

            ret_gp_color_diff = gp_color_diff.square().mean(-1)
            # [...]

        if not self.training:
            ret_gp_scale_diff = torch.Tensor()
        else:
            world_gp_x_scale = world_gp_result.gp_scale[..., 0]
            world_gp_y_scale = world_gp_result.gp_scale[..., 1]

            word_gp_xy_scale = world_gp_x_scale * world_gp_y_scale
            # [..., F]

            word_gp_xy_scale_diff = avatar_model.mesh_graph.calc_face_diff(
                word_gp_xy_scale[..., None])
            # [..., FP, 1]

            ret_gp_scale_diff = word_gp_xy_scale_diff.sum((-2, -1)) / \
                word_gp_xy_scale_diff.shape[-2]
            # [...]

        return ModuleForwardResult(
            avatar_model=avatar_model,

            gp_render_img=gp_render_img,

            img_diff=ret_img_diff,
            lap_diff=ret_lap_diff,
            nor_sim=ret_nor_sim,
            edge_var=ret_edge_var,
            gp_color_diff=ret_gp_color_diff,
            gp_scale_diff=ret_gp_scale_diff,
        )

    def refresh(self) -> None:
        if hasattr(self, "avatar_blender"):
            self.avatar_blender.refresh()
