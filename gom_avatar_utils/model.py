import dataclasses
import math
import typing

import torch
from beartype import beartype

from .. import (avatar_utils, camera_utils, gaussian_utils, transform_utils,
                utils)


@beartype
@dataclasses.dataclass
class FaceCoordResult:
    Ts: torch.Tensor  # [..., F, 4, 4]
    normalized_Ts: torch.Tensor  # [..., F, 4, 4]
    face_area: torch.Tensor  # [..., F]


@beartype
def GetFaceCoord(
    vertex_positions_a: torch.Tensor,  # [..., 3]
    vertex_positions_b: torch.Tensor,  # [..., 3]
    vertex_positions_c: torch.Tensor,  # [..., 3]
):
    vpa = vertex_positions_a
    vpb = vertex_positions_b
    vpc = vertex_positions_c

    utils.CheckShapes(
        vpa, (..., 3),
        vpb, (..., 3),
        vpc, (..., 3),
    )

    device = utils.CheckDevice(vpa, vpb, vpc)

    batch_shapes = utils.BroadcastShapes(
        vpa.shape[:-1],
        vpb.shape[:-1],
        vpc.shape[:-1],
    )

    vpa = vpa.expand(batch_shapes + (3,))
    vpb = vpb.expand(batch_shapes + (3,))
    vpc = vpc.expand(batch_shapes + (3,))

    s = (vpa + vpb + vpc) / 3
    # [..., 3]

    f1 = vpc - s
    f2 = (vpa - vpb) / math.sqrt(3)

    f1_sq = f1.square().sum(-1)
    f2_sq = f2.square().sum(-1)
    f1_dot_f2 = (f1 * f2).sum(-1)

    """

    half_cos_2t = 0.5 / (1 + tan_2t.square()).sqrt()

    cos_2t = 1 / (1 + tan_2t.square()).sqrt()

    half_cos_2t = 0.5 / (1 + tan_2t.square()).sqrt()

    cos_t = ((1 + cos_2t) / 2).sqrt()
          = (0.5 + half_cos_2t).sqrt

    cos_t = ((utils.EPS + 0.5) + half_cos_2t).sqrt().unsqueeze(-1)
    sin_t = ((utils.EPS + 0.5) - half_cos_2t).sqrt().unsqueeze(-1)
    """

    t = torch.atan2(2 * f1_dot_f2, f1_sq - f2_sq) * 0.5

    cos_t = torch.cos(t).unsqueeze(-1)
    sin_t = torch.sin(t).unsqueeze(-1)

    Ts = torch.empty(
        batch_shapes + (4, 4), dtype=utils.FLOAT, device=device)

    normalized_Ts = torch.empty(
        batch_shapes + (4, 4), dtype=utils.FLOAT, device=device)

    axis_x = Ts[..., :3, 0] = f1 * cos_t + f2 * sin_t
    axis_y = Ts[..., :3, 1] = f2 * cos_t - f1 * sin_t
    axis_z = Ts[..., :3, 2] = torch.linalg.cross(axis_x, axis_y)

    z_norm = utils.EPS + utils.VectorNorm(axis_z)

    normalized_Ts[..., :3, 0] = utils.Normalized(axis_x)
    normalized_Ts[..., :3, 1] = utils.Normalized(axis_y)
    normalized_Ts[..., :3, 2] = axis_z / z_norm.unsqueeze(-1)

    err = (normalized_Ts[..., :3, 0] *
           normalized_Ts[..., :3, 1]).sum(-1).abs().max()

    assert err <= 2e-3, err

    Ts[..., :3, 3] = s
    Ts[..., 3, :3] = 0
    Ts[..., 3, 3] = 1

    normalized_Ts[..., :3, 3] = s
    normalized_Ts[..., 3, :3] = 0
    normalized_Ts[..., 3, 3] = 1

    ret = FaceCoordResult(
        Ts=Ts,
        normalized_Ts=normalized_Ts,
        face_area=z_norm * 3,
    )

    return ret


@beartype
@dataclasses.dataclass
class GoMAvatarModelForwardResult:
    rendered_img: torch.Tensor  # [..., C, H, W]

    rgb_loss: typing.Optional[torch.Tensor]
    lap_loss: typing.Optional[torch.Tensor]
    normal_sim_loss: typing.Optional[torch.Tensor]
    color_diff_loss: typing.Optional[torch.Tensor]


@beartype
class GoMAvatarModel(torch.nn.Module):
    def __init__(
        self,
        avatar_blending_layer: avatar_utils.AvatarBlendingLayer,
        color_channels_cnt: int,
    ):
        super(GoMAvatarModel, self).__init__()

        self.avatar_blending_layer: avatar_utils.AvatarBlendingLayer = \
            avatar_blending_layer

        faces_cnt = self.avatar_blending_layer.GetFacesCnt()
        assert 0 <= faces_cnt

        assert 1 <= color_channels_cnt

        gp_rot_qs = torch.empty((faces_cnt, 4), dtype=utils.FLOAT)
        # quaternion wxyz

        gp_rot_qs[:, 0] = 1
        gp_rot_qs[:, 1:] = 0

        self.gp_rot_qs = torch.nn.Parameter(gp_rot_qs)

        self.gp_scales = torch.nn.Parameter(
            torch.ones((faces_cnt, 3), dtype=utils.FLOAT) * 10)
        # self.gp_scales = torch.ones((faces_cnt, 3), dtype=utils.FLOAT) * 10
        # [F, 3]

        self.gp_colors = torch.nn.Parameter(torch.rand(
            (faces_cnt, color_channels_cnt),
            dtype=utils.FLOAT))

        self.gp_opacities = torch.nn.Parameter(torch.ones(
            (faces_cnt, 1),
            dtype=utils.FLOAT))

    def forward(
        self,
        camera_transform: transform_utils.ObjectTransform,
        camera_config: camera_utils.CameraConfig,
        img: torch.Tensor,  # [..., C, H, W]
        mask: torch.Tensor,  # [..., H, W]
        blending_param: object,
    ):
        device = next(self.parameters()).device

        color_channels_cnt = self.gp_colors.shape[-1]

        H, W = -1, -2

        H, W = utils.CheckShapes(img, (..., color_channels_cnt, H, W))

        avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blending_layer(blending_param)

        utils.PrintCudaMemUsage()

        faces = avatar_model.GetFaces()
        # [F, 3]

        vertex_positions = avatar_model.GetVertexPositions()
        # [..., V, 3]

        utils.PrintCudaMemUsage()

        vertex_positions_a = vertex_positions[..., faces[:, 0], :]
        vertex_positions_b = vertex_positions[..., faces[:, 1], :]
        vertex_positions_c = vertex_positions[..., faces[:, 2], :]
        # [..., F, 3]

        utils.PrintCudaMemUsage()

        face_coord_result = GetFaceCoord(
            vertex_positions_a, vertex_positions_b, vertex_positions_c)

        face_coord_rot_qs = utils.RotMatToQuaternion(
            face_coord_result.normalized_Ts[..., :3, :3],
            order="WXYZ",
        )
        # [..., F, 4]

        # d = face_coord_result.normalized_Ts[..., :3, :3].det()

        # print(f"{d.min()=}")
        # print(f"{d.max()=}")

        utils.PrintCudaMemUsage()

        gp_global_means = face_coord_result.Ts[..., :3, 3]
        # [..., F, 3]

        """
        gp_global_rot_qs = utils.QuaternionMul(
            face_coord_rot_qs, self.gp_rot_qs,
            order_1="WXYZ", order_2="WXYZ", order_out="WXYZ")
        """

        gp_global_rot_qs = self.gp_rot_qs

        # print(f"{face_coord_rot_qs=}")

        # d2 = (utils.VectorNorm(gp_global_rot_qs) - 1).square()

        # print(f"{d2.min()=}")
        # print(f"{d2.max()=}")
        # print(f"{(utils.VectorNorm(self.gp_rot_qs) - 1).square().mean()=}")

        # [..., F, 4] wxyz

        # print(f"{self.gp_scales.square().sum()}")
        # print(f"{self.gp_rot_qs.square().sum()}")

        face_area = face_coord_result.face_area.unsqueeze(-1)
        # [..., F, 1]

        # face_area_min = face_area.min()
        # face_area_max = face_area.max()

        # print(f"{face_area_min=}")
        # print(f"{face_area_max=}")

        # assert 0 < face_area_min
        # assert face_area_max <= 1

        # utils.Exit(0)

        gp_global_scales = face_area * self.gp_scales
        # [..., F, 3]

        utils.PrintCudaMemUsage()

        rendered_result = gaussian_utils.RenderGaussian(
            camera_transform=camera_transform,
            camera_config=camera_config,

            sh_degree=0,

            bg_color=torch.ones((color_channels_cnt,),
                                dtype=utils.FLOAT, device=device),

            gp_means=gp_global_means,
            gp_rots=gp_global_rot_qs,
            gp_scales=gp_global_scales,

            gp_shs=None,
            gp_colors=self.gp_colors,

            gp_opacities=self.gp_opacities,

            device=device,
        )  # [...]

        rendered_img = rendered_result.colors
        # [..., C, H, W]

        mesh_data = avatar_model.GetMeshData()

        if not self.training:
            rgb_loss = None
        else:
            mask_ = mask.unsqueeze(-3)

            white_img = torch.ones_like(img)

            masked_img = white_img * (1 - mask_) + img * mask_

            rgb_loss = (rendered_img - masked_img).square().mean()

        if not self.training:
            lap_loss = None
        else:
            lap_diff = mesh_data.GetLapDiff(
                avatar_model.GetVertexPositions())
            # [..., V, 3]

            lap_loss = lap_diff.square().mean()

        if not self.training:
            normal_sim_loss = None
        else:
            normal_sim = mesh_data.GetFaceCosSim(
                face_coord_result.normalized_Ts[..., :3, 2]
                # the z axis of each face
            )
            # [..., FP]

            normal_sim_loss = 1 - normal_sim.mean()

        if not self.training:
            color_diff_loss = None
        else:
            color_diff = mesh_data.GetFaceCosSim(self.gp_colors)
            # [..., FP]

            color_diff_loss = color_diff.square().mean()

        return GoMAvatarModelForwardResult(
            rendered_img=rendered_img,
            rgb_loss=rgb_loss,
            lap_loss=lap_loss,
            normal_sim_loss=normal_sim_loss,
            color_diff_loss=color_diff_loss,
        )
