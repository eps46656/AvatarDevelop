import itertools
import math
import dataclasses

import torch
from beartype import beartype

import pytorch3d
import pytorch3d.renderer


from .. import avatar_utils, utils, gaussian_utils


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


class Model(torch.nn.Model):
    def __init__(
        self,
        avatar_blending_layer: avatar_utils.AvatarBlendingLayer,
        gp_sh_degree: int,
    ):
        self.avatar_blending_layer: avatar_utils.AvatarBlendingLayer = \
            avatar_blending_layer

        faces_cnt = self.avatar_blending_layer.faces_cnt
        assert 0 <= faces_cnt

        assert 0 <= gp_sh_degree <= 2

        self.gp_sh_degree = gp_sh_degree

        self.gp_rot_qs = torch.nn.Parameter(
            torch.empty((faces_cnt, 4), dtype=utils.FLOAT))
        # quaternion xyzw

        self.gp_rot_qs[:, 0] = 0
        self.gp_rot_qs[:, 1] = 0
        self.gp_rot_qs[:, 2] = 0
        self.gp_rot_qs[:, 3] = 1

        self.gp_scales = torch.nn.Parameter(
            torch.ones((faces_cnt, 3), dtype=utils.FLOAT))
        # [F, 3]

        self.gp_shs = torch.nn.Parameter(torch.ones(
            (faces_cnt, (self.gp_sh_degree + 1)**2, 3),
            dtype=utils.FLOAT))

    def forward(
        self,
        cameras: pytorch3d.renderer.cameras.CamerasBase,
        images: torch.Tensor,  # [C, H, W]
        blending_param: object,
    ):
        avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blending_layer(blending_param)

        faces = avatar_model.faces
        # [F, 3]

        vertex_positions = avatar_model.vertex_positions
        # [..., V, 3]

        vertex_positions_a = vertex_positions[..., faces[:, 0], :]
        vertex_positions_b = vertex_positions[..., faces[:, 1], :]
        vertex_positions_c = vertex_positions[..., faces[:, 2], :]
        # [..., F, 3]

        face_coord_result = GetFaceCoord(
            vertex_positions_a, vertex_positions_b, vertex_positions_c)

        gp_global_means = face_coord_result.Ts[..., :3, 3]
        # [..., F, 3]

        gp_global_rot_mats = face_coord_result.normalized_Ts[..., :3, :3] @ \
            utils.QuaternionToRotMat(self.gp_rot_qs)
        # [..., F, 3, 3]

        gp_global_rot_qs = utils.RotMatToQuaternion(
            gp_global_rot_mats, order="wxyz")

        face_area_vec = face_coord_result.Ts[..., :3, 3]
        # [..., F, 3]

        face_area = utils.VectorNorm(face_area_vec)
        # [..., F]

        gp_global_scales = face_area * self.gp_scales
        # [..., F, 3]

        rendered_img: torch.Tensor = gaussian_utils.RenderGaussian(
            gp_sh_degree=self.gp_sh_degree,

            gp_means=gp_global_means,
            gp_rots=gp_global_rot_qs,
            gp_scales=gp_global_scales,
            gp_shs=self.gp_shs,
        )  # [..., C, H, W]

        mesh_data = avatar_model.mesh_data

        if not self.training:
            rgb_loss = None
        else:
            rgb_loss = (rendered_img - images).square()

        if not self.training:
            lap_loss = None
        else:
            lap_diff = avatar_model.mesh_data.GetLapDiff(
                avatar_model.vertex_positions)
            # [..., V, 3]

            lap_loss = lap_diff.square().mean()

        if not self.training:
            normal_sim_loss = None
        else:
            normal_sim = avatar_model.mesh_data.GetNormalSim(
                avatar_model.vertex_positions)
            # [..., FP]

            normal_sim_loss = normal_sim.mean()

        pass
