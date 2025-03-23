import dataclasses
import math

import einops
import torch
from beartype import beartype

from .. import (avatar_utils, camera_utils, gaussian_utils, transform_utils,
                utils)


@beartype
@dataclasses.dataclass
class FaceCoordResult:
    normalized_Ts: torch.Tensor  # [..., 4, 4]
    Ts: torch.Tensor  # [..., F, 4, 4]


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

    device = vpa.device

    assert vpa.device == device
    assert vpb.device == device
    assert vpc.device == device

    batch_shapes = utils.BroadcastShapes(
        vpa.shape[:-1],
        vpb.shape[:-1],
        vpc.shape[:-1],
    )

    vpa: torch.Tensor = vpa.expand(batch_shapes + (3,))
    vpb: torch.Tensor = vpb.expand(batch_shapes + (3,))
    vpc: torch.Tensor = vpc.expand(batch_shapes + (3,))

    s = (vpa + vpb + vpc) / 3
    # [..., 3]

    f1 = vpc - s
    f2 = (vpb - vpa) / math.sqrt(3)

    f1_sq = f1.square().sum(dim=-1)
    f2_sq = f2.square().sum(dim=-1)
    f1_dot_f2 = (f1 * f2).sum(dim=-1)

    t0 = 0.5 * torch.atan((2 * f1_dot_f2) / (f1_sq - f2_sq))

    cos_t0 = torch.cos(t0)
    sin_t0 = torch.sin(t0)

    normalized_Ts = torch.empty(
        batch_shapes + (4, 4), dtype=utils.FLOAT, device=device)
    Ts = torch.empty(batch_shapes + (4, 4), dtype=utils.FLOAT, device=device)

    axis_x = Ts[..., :3, 0] = f1 * cos_t0 + f2 * sin_t0
    axis_y = Ts[..., :3, 1] = f2 * cos_t0 - f1 * sin_t0

    normalized_axis_x = normalized_Ts[..., :3, 0] = utils.Normalized(axis_x)
    normalized_axis_y = normalized_Ts[..., :3, 1] = utils.Normalized(axis_y)

    normalized_axis_z = axis_z = normalized_Ts[..., :3, 2] = Ts[..., :3, 2] = \
        torch.linalg.cross(normalized_axis_x, normalized_axis_y)

    normalized_Ts[..., :3, 3] = s
    normalized_Ts[..., 3, :3] = 0
    normalized_Ts[..., 3, 3] = 1

    Ts[..., :3, 3] = s
    Ts[..., 3, :3] = 0
    Ts[..., 3, 3] = 1

    ret = FaceCoordResult(
        normalized_Ts=normalized_Ts,
        Ts=Ts,
    )

    return ret


@beartype
@dataclasses.dataclass
class GoMAvatarModelForwardResult:
    rendered_imgs: torch.Tensor  # [..., (C, H, W)]

    rgb_loss: torch.Tensor
    lap_loss: torch.Tensor
    normal_sim_loss: torch.Tensor


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

        self.gp_rot_qs = torch.nn.Parameter(
            torch.empty((faces_cnt, 4), dtype=utils.FLOAT))
        # quaternion wxyz

        self.gp_rot_qs[:, 0] = 1
        self.gp_rot_qs[:, 1] = 0
        self.gp_rot_qs[:, 2] = 0
        self.gp_rot_qs[:, 3] = 0

        self.gp_scales = torch.nn.Parameter(
            torch.ones((faces_cnt, 3), dtype=utils.FLOAT))
        # [F, 3]

        self.gp_colors = torch.nn.Parameter(torch.ones(
            (faces_cnt, color_channels_cnt),
            dtype=utils.FLOAT))

        self.gp_opacities = torch.nn.Parameter(torch.ones(
            (faces_cnt, 1),
            dtype=utils.FLOAT))

    def forward(
        self,
        camera_transform: transform_utils.ObjectTransform,
        camera_config: camera_utils.CameraConfig,
        imgs: torch.Tensor,  # [..., C, H, W]
        blending_param: object,
    ):
        device = next(self.paremeters()).device

        color_channels_cnt = self.gp_colors.shape[-1]

        H, W = -1, -2

        H, W = utils.CheckShapes(imgs, (..., color_channels_cnt, H, W))

        avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blending_layer(blending_param)

        faces = avatar_model.GetFaces()
        # [F, 3]

        vertex_positions = avatar_model.GetVertexPositions()
        # [..., V, 3]

        vertex_positions_a = vertex_positions[..., faces[:, 0], :]
        vertex_positions_b = vertex_positions[..., faces[:, 1], :]
        vertex_positions_c = vertex_positions[..., faces[:, 2], :]
        # [..., F, 3]

        face_coord_result = GetFaceCoord(
            vertex_positions_a, vertex_positions_b, vertex_positions_c)

        face_coord_rot_qs = utils.RotMatToQuaternion(
            face_coord_result.normalized_Ts[..., :3, :3],
            order="WXYZ",
        )

        gp_global_means = face_coord_result.Ts[..., :3, 3]
        # [..., F, 3]

        gp_global_rot_qs = utils.QuaternionMul(
            face_coord_rot_qs, self.gp_rot_qs,
            order_1="WXYZ", order_2="WXYZ", order_out="WXYZ")
        # [..., F, 3, 3] wxyz

        face_area_vec = face_coord_result.Ts[..., :3, 3]
        # [..., F, 3]

        face_area = utils.VectorNorm(face_area_vec)
        # [..., F]

        gp_global_scales = face_area * self.gp_scales
        # [..., F, 3]

        rendered_img: torch.Tensor = gaussian_utils.RenderGaussian(
            camera_transform=camera_transform,
            camera_config=camera_config,

            sh_degree=0,

            bg_color=torch.ones((color_channels_cnt,),
                                dtype=utils.FLOAT, device=device),

            gp_means=gp_global_means,
            gp_rots=gp_global_rot_qs,
            gp_scales=gp_global_scales,

            gp_shs=torch.Tensor([]),
            gp_colors=self.gp_colors,

            gp_opacities=self.gp_opacities,

            device=device,
        )  # [..., C, H, W]

        mesh_data = avatar_model.GetMeshData()

        if not self.training:
            rgb_loss = None
        else:
            rgb_loss = (rendered_img - imgs).square()

        if not self.training:
            lap_loss = None
        else:
            lap_diff = mesh_data.GetLapDiff(
                avatar_model.vertex_positions)
            # [..., V, 3]

            lap_loss = lap_diff.square().mean()

        if not self.training:
            normal_sim_loss = None
        else:
            normal_sim = mesh_data.GetNormalSim(
                face_coord_result.normalized_Ts[..., :3, 2]
                # the z axis of each face
            )
            # [..., FP]

            normal_sim_loss = normal_sim.mean()

        GoMAvatarModelForwardResult(
            rgb_loss=rgb_loss,
            lap_loss=lap_loss,
            normal_sim_loss=normal_sim_loss,
        )
