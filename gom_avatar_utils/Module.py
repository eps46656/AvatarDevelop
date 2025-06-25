from __future__ import annotations

import collections
import dataclasses
import typing
import math

import torch
from beartype import beartype

from .. import (avatar_utils, camera_utils, gaussian_utils, mesh_utils,
                rendering_utils, transform_utils, utils, vision_utils)
from .utils import FaceCoordResult, get_face_coord


@beartype
@dataclasses.dataclass
class ModuleObservationGPResult:
    obs_gp_mean: torch.Tensor  # [..., F, 3]

    local_gp_rot: torch.Tensor  # [F, 3, 3]
    local_gp_scale: torch.Tensor  # [F, 3]
    obs_gp_cov3d: torch.Tensor  # [..., F, 3, 3]

    gp_color: torch.Tensor  # [..., F, 3]
    gp_opacity: torch.Tensor  # [..., F, 1]


@beartype
@dataclasses.dataclass
class ModuleForwardResult:
    avatar_model: avatar_utils.AvatarModel

    gp_render_img: torch.Tensor  # [..., C, H, W]

    img_diff: torch.Tensor  # [...]
    gp_mask_diff: list[torch.Tensor]  # [][...]
    lap_diff: torch.Tensor  # [...]
    nor_sim: torch.Tensor  # [...]
    edge_var: torch.Tensor  # [...]


@beartype
class Module(torch.nn.Module):
    def __init__(
        self,
        avatar_blender: avatar_utils.AvatarBlender,
        color_channels_cnt: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.avatar_blender: avatar_utils.AvatarBlender = avatar_blender

        avatar_model: avatar_utils.AvatarModel = self.avatar_blender.get_avatar_model()

        verts_cnt = avatar_model.verts_cnt
        assert 0 < verts_cnt

        assert 1 <= color_channels_cnt

        self.gp_rot = torch.nn.Parameter(torch.zeros(
            (verts_cnt, 3),
            dtype=dtype,
            device=device,
        ))

        self.gp_scale = torch.nn.Parameter(torch.full(
            (verts_cnt, 3),
            5,
            dtype=dtype,
            device=device,
        ))

        self.gp_color = torch.nn.Parameter(torch.zeros(
            (verts_cnt, color_channels_cnt),
            dtype=dtype,
            device=device,
        ))

        self.gp_opacity = torch.nn.Parameter(torch.zeros(
            (verts_cnt,),
            dtype=dtype,
            device=device,
        ))

    def to(self, *args, **kwargs) -> Module:
        super().to(*args, **kwargs)

        self.avatar_blender = self.avatar_blender.to(*args, **kwargs)

        return self

    def get_param_groups(self, base_lr: float) -> list[dict]:
        ret = utils.get_param_groups(self.avatar_blender, base_lr)

        ret.append({
            "params": [
                self.gp_rot,
                self.gp_scale,
                self.gp_color,
                self.gp_opacity,
            ],
            "lr": base_lr
        })

        return ret

    def state_dict(self) -> collections.OrderedDict:
        return collections.OrderedDict([
            ("avatar_blender", self.avatar_blender.state_dict()),

            ("gp_rot", self.gp_rot),

            ("gp_scale", self.gp_scale),

            ("gp_color", self.gp_color),
            ("gp_opacity", self.gp_opacity),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.avatar_blender.load_state_dict(
            state_dict["avatar_blender"])

        self.gp_rot = torch.nn.Parameter(
            state_dict["gp_rot"].to(self.gp_rot, copy=True))

        self.gp_scale = torch.nn.Parameter(
            state_dict["gp_scale"].to(self.gp_scale, copy=True))

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

        vert_src_table = mesh_subdivide_result.vert_src_table
        # [V_, 2]

        self.gp_rot = torch.nn.Parameter(
            self.gp_rot[vert_src_table].mean(-2))
        # [V_, 3]

        self.gp_scale = torch.nn.Parameter(
            self.gp_scale[vert_src_table].mean(-2))
        # [V_, 3]

        self.gp_color = torch.nn.Parameter(
            self.gp_color[vert_src_table].mean(-2))
        # [V_, C]

        self.gp_opacity = torch.nn.Parameter(
            self.gp_opacity[vert_src_table].mean(-1))
        # [V_]

        return self

    def get_obs_gp(
        self,
        avatar_model: avatar_utils.AvatarModel,
    ) -> ModuleObservationGPResult:
        obs_gp_mean = avatar_model.vert_pos
        # [..., V, 3]

        local_gp_rot = utils.axis_angle_to_rot_mat(
            axis_angle=self.gp_rot,  # [V, 3]
            out_shape=(3, 3),
        )
        # [V, 3, 3]

        local_gp_scale = 5e-3 + 1e-4 * torch.nn.functional.softplus(
            self.gp_scale)
        # [..., V, 3]

        vert_trans = avatar_model.vert_trans[..., :3, :3]
        # [..., V, 3, 3]

        world_gp_cov3d = gaussian_utils.trans_to_cov3d(utils.mat_mul(
            vert_trans, local_gp_rot, utils.make_diag(local_gp_scale)
        ))
        # [..., V, 3, 3]

        world_gp_color = self.gp_color.sigmoid()

        world_gp_opacity = utils.smooth_clamp(
            x=self.gp_opacity,
            lb=0.8,
            rb=1.0,
        )

        return ModuleObservationGPResult(
            obs_gp_mean=obs_gp_mean,

            local_gp_rot=local_gp_rot,
            local_gp_scale=local_gp_scale,
            obs_gp_cov3d=world_gp_cov3d,

            gp_color=world_gp_color,
            gp_opacity=world_gp_opacity,
        )

    def forward(
        self,
        *,
        camera_config: camera_utils.CameraConfig,
        camera_transform: transform_utils.ObjectTransform,
        img: torch.Tensor,  # [..., C, H, W]
        mask: torch.Tensor,  # [..., 1, H, W]
        dilated_mask: list[torch.Tensor],  # [..., 1, H, W]
        blending_param: object,

        silhouette_sigma: list[float],
        silhouette_opacity: list[float],
    ) -> ModuleForwardResult:
        device = next(self.parameters()).device

        V, C = self.gp_color.shape

        H, W = -1, -2

        H, W = utils.check_shapes(
            img, (..., C, H, W),
            mask, (..., 1, H, W),
        )

        batch_shape = utils.broadcast_shapes(
            camera_transform,
            img.shape[:-3],
            mask.shape[:-3],
            *(x.shape[:-3] for x in dilated_mask),
            blending_param,
        )

        first_idx = (0,) * len(batch_shape)

        mask_sum = mask.sum((-3, -2, -1))
        # [...]

        sigma_n = len(dilated_mask)

        dilated_mask = [
            (1 - (1 - cur_dilated_mask).pow(V))
            for cur_dilated_mask in dilated_mask
        ]

        avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blender(blending_param)

        canon_avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blender.get_avatar_model()

        obs_gp_result = self.get_obs_gp(avatar_model)

        gp_rgb_render_result: gaussian_utils.RenderGaussianResult = \
            gaussian_utils.render_gaussian(
                camera_config=camera_config,
                camera_transform=camera_transform,

                bg_color=utils.ones(
                    shape=(C,),
                    like=obs_gp_result.gp_color,
                ),

                gp_mean=obs_gp_result.obs_gp_mean,
                gp_cov3d=obs_gp_result.obs_gp_cov3d,

                gp_sh=None,
                gp_color=obs_gp_result.gp_color,

                gp_opacity=obs_gp_result.gp_opacity,

                calc_alpha=False,
            )  # [...]

        gp_render_img = gp_rgb_render_result.color
        # [..., C, H, W]

        # ---

        gp_dist = utils.vec_dot(
            obs_gp_result.obs_gp_mean.detach() -
            camera_transform.pos[..., None, :],

            camera_transform.vec_f[..., None, :],
        )
        # [..., V]

        gp_silhouette_base_scale = gp_dist * (1.0 * math.sqrt(
            ((camera_config.foc_u + camera_config.foc_d) ** 2 +
             (camera_config.foc_l + camera_config.foc_r) ** 2) /
            (H ** 2 + W ** 2)
        ))
        # [..., V]

        gp_silhouette_rot_q = torch.tensor(
            [0, 0, 0, 1],
            dtype=obs_gp_result.obs_gp_mean.dtype,
            device=gp_silhouette_base_scale.device,
        ).expand(V, 4)

        gp_silhouette_color = utils.dummy_ones(
            shape=(V, C,), like=obs_gp_result.gp_color)

        # ---

        if not self.training:
            ret_img_diff = utils.zeros(
                shape=img.shape[:-3],
                like=img,
            )
        else:
            person_masked_img = img * mask + (1 - mask)

            vision_utils.show_image(
                "img",
                utils.rct(
                    img[*first_idx, :, :, :] * 255,
                    dtype=torch.uint8,
                ),
            )

            vision_utils.show_image(
                "gp_render_img",
                utils.rct(
                    gp_render_img[*first_idx, :, :, :] * 255,
                    dtype=torch.uint8,
                ),
            )

            vision_utils.show_image(
                "merged_gp_render_img",
                utils.rct((
                    img[*first_idx, :, :, :] +
                    gp_render_img[*first_idx, :, :, :]
                ) * (255 / 2),
                    dtype=torch.uint8,
                ),
            )

            ret_img_diff = (
                gp_render_img - person_masked_img
            ).abs().sum((-3, -2, -1)) / (H * W)
            # [...]

        ret_gp_mask_diff: list[torch.Tensor] = list()

        for sigma_idx in range(sigma_n):
            if not self.training:
                ret_gp_mask_diff.append(utils.zeros(
                    shape=batch_shape,
                    like=dilated_mask[sigma_idx],
                ))

                continue

            gp_silhouette_scale = silhouette_sigma[sigma_idx] * \
                gp_silhouette_base_scale
            # [..., V]

            gp_silhouette_scale = gp_silhouette_scale[..., None].expand(
                *gp_silhouette_scale.shape, 3)
            # [..., V, 3]

            gp_silhouette_render_result: gaussian_utils.RenderGaussianResult = \
                gaussian_utils.render_gaussian(
                    camera_config=camera_config,
                    camera_transform=camera_transform,

                    bg_color=utils.zeros(
                        shape=(C,),
                        requires_grad=False,
                        like=obs_gp_result.gp_color,
                    ),

                    gp_mean=obs_gp_result.obs_gp_mean,

                    gp_rot_q=gp_silhouette_rot_q,

                    gp_scale=gp_silhouette_scale,

                    gp_sh=None,
                    gp_color=gp_silhouette_color,

                    gp_opacity=torch.where(
                        1e-2 <= obs_gp_result.gp_opacity,  # [V]

                        torch.tensor(
                            silhouette_opacity[sigma_idx],
                            dtype=obs_gp_result.gp_opacity.dtype,
                            device=obs_gp_result.gp_opacity.device,
                        ),

                        torch.tensor(
                            0.0,
                            dtype=obs_gp_result.gp_opacity.dtype,
                            device=obs_gp_result.gp_opacity.device,
                        ),
                    ),

                    calc_alpha=False,
                )  # [...]

            gp_silhouette_render_img = \
                gp_silhouette_render_result.color[..., :1, :, :]
            # [..., 1, H, W]

            vision_utils.show_image(
                f"dilated_mask ({sigma_idx})",
                utils.rct(
                    dilated_mask[sigma_idx][first_idx] * 255,
                    dtype=torch.uint8,
                ),
            )

            vision_utils.show_image(
                f"dilated_mask + alpha ({sigma_idx})",

                torch.cat([
                    utils.rct(
                        gp_silhouette_render_img[first_idx] * 255,
                        dtype=torch.uint8,
                    ),

                    utils.rct(
                        dilated_mask[sigma_idx][first_idx] * 255,
                        dtype=torch.uint8,
                    ),

                    utils.dummy_zeros(
                        like=gp_silhouette_render_img[first_idx],
                        dtype=torch.uint8,
                    ),
                ], 0)
            )

            gp_mask_diff = gp_silhouette_render_img - dilated_mask[sigma_idx]

            """

            pr gt
            0  1     neg
            1  0     pos

            """

            """
            ret_gp_mask_diff.append((
                0.3 * gp_mask_diff.clamp(0, None) -
                1.7 * gp_mask_diff.clamp(None, 0)
            ).sum((-3, -2, -1)) / (H * W))
            """

            ret_gp_mask_diff.append((
                gp_mask_diff.abs()
            ).sum((-3, -2, -1)) / (H * W))
            # [...]

        if not self.training:
            ret_lap_diff = utils.zeros(
                shape=batch_shape,
                like=canon_avatar_model.vert_pos,
            )
        else:
            vert_lap_diff = canon_avatar_model.mesh_data.uni_lap_diff
            # [..., V, D]

            ret_lap_diff = utils.vec_sq_norm(vert_lap_diff).mean(-1)
            # [...]

        if not self.training:
            ret_nor_sim = utils.zeros(
                shape=batch_shape,
                like=canon_avatar_model.vert_pos,
            )
        else:
            nor_sim = canon_avatar_model.mesh_data.face_norm_cos_sim
            # [..., FP]

            ret_nor_sim = (1 - nor_sim).mean(-1)
            # [...]

        if not self.training:
            ret_edge_var = utils.zeros(
                shape=batch_shape,
                like=canon_avatar_model.vert_pos,
            )
        else:
            edge_var = canon_avatar_model.mesh_data.face_edge_var
            # [..., F]

            ret_edge_var = edge_var.mean(-1)
            # [...]

        return ModuleForwardResult(
            avatar_model=avatar_model,

            gp_render_img=gp_render_img,

            img_diff=ret_img_diff,
            gp_mask_diff=ret_gp_mask_diff,
            lap_diff=ret_lap_diff,
            nor_sim=ret_nor_sim,
            edge_var=ret_edge_var,
        )

    def refresh(self) -> None:
        self.avatar_blender.refresh()
