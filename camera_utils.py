import dataclasses
import enum
import math

import torch
from beartype import beartype

import utils


@beartype
def MakeViewMatWithURF(
    *,
    origin: torch.Tensor,  # [..., 3]
    u_vec: torch.Tensor,  # [..., 3]
    r_vec: torch.Tensor,  # [..., 3]
    f_vec: torch.Tensor,  # [..., 3]
    view_coord: utils.Coord3,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:  # [..., 4, 4]
    utils.CheckShapes(
        origin, (..., 3),
        u_vec, (..., 3),
        r_vec, (..., 3),
        f_vec, (..., 3),
    )

    def GetVec(axis):
        match axis:
            case utils.Dir6.F: return f_vec
            case utils.Dir6.B: return -f_vec
            case utils.Dir6.U: return u_vec
            case utils.Dir6.D: return -u_vec
            case utils.Dir6.L: return -r_vec
            case utils.Dir6.R: return r_vec

    x_vec = GetVec(view_coord.dirs[0])
    y_vec = GetVec(view_coord.dirs[1])
    z_vec = GetVec(view_coord.dirs[2])

    batch_dims = list(utils.GetCommonShape([
        x_vec.shape[:-1],
        y_vec.shape[:-1],
        z_vec.shape[:-1],
    ]))

    ret = torch.empty(batch_dims + [4, 4], dtype=dtype, device=device)

    ret[..., 0, 0] = x_vec[..., 0]
    ret[..., 0, 1] = y_vec[..., 0]
    ret[..., 0, 2] = z_vec[..., 0]
    ret[..., 0, 3] = origin[..., 0]

    ret[..., 1, 0] = x_vec[..., 1]
    ret[..., 1, 1] = y_vec[..., 1]
    ret[..., 1, 2] = z_vec[..., 1]
    ret[..., 1, 3] = origin[..., 1]

    ret[..., 2, 0] = x_vec[..., 2]
    ret[..., 2, 1] = y_vec[..., 2]
    ret[..., 2, 2] = z_vec[..., 2]
    ret[..., 2, 3] = origin[..., 2]

    ret[..., 3, :3] = 0
    ret[..., 3, 3] = 1

    return torch.inverse(ret)


@beartype
def MakeViewMat(
    origin: torch.Tensor,  # [..., 3]
    aim: torch.Tensor,  # [..., 3]
    quasi_u_dir: torch.Tensor,  # [..., 3]
    view_coord: utils.Coord3,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:  # [..., 4, 4]
    assert 1 <= origin.dim()
    assert 1 <= aim.dim()
    assert 1 <= quasi_u_dir.dim()

    assert origin.shape[-1] == 3
    assert aim.shape[-1] == 3
    assert quasi_u_dir.shape[-1] == 3

    f_vec = utils.Normalized(aim - origin, -1)
    r_vec = utils.Normalized(torch.cross(f_vec, quasi_u_dir, dim=-1), -1)
    u_vec = torch.cross(r_vec, f_vec, dim=0)

    return MakeViewMatWithURF(
        origin=origin,
        u_vec=u_vec,
        r_vec=r_vec,
        f_vec=f_vec,
        view_coord=view_coord,
        dtype=dtype,
        device=device,
    )


@beartype
def GetFocalLengthByDiagFoV(img_h: float, img_w: float, fov_diag: float):
    assert 0 < img_h
    assert 0 < img_w

    assert 0 < fov_diag < 180 * utils.DEG

    return math.sqrt(img_h**2 + img_w**2) / (2 * math.tan(fov_diag / 2))


class Coord(enum.StrEnum):
    World = "World"
    View = "View"
    NDC = "NDC"
    Screen = "Screen"


@beartype
class Volume:
    def __init__(
        self,
        delta_f: float,
        delta_b: float,
        delta_u: float,
        delta_d: float,
        delta_l: float,
        delta_r: float,
    ):
        self.delta_f = delta_f
        self.delta_b = delta_b
        self.delta_u = delta_u
        self.delta_d = delta_d
        self.delta_l = delta_l
        self.delta_r = delta_r

    @staticmethod
    def FromFovDiag(
        *,
        img_h: float,
        img_w: float,
        fov_diag: float,
        depth_near: float,
        depth_far: float,
    ):
        assert 0 < img_h
        assert 0 < img_w
        assert 0 < fov_diag < 180 * utils.DEG

        focal_length = GetFocalLengthByDiagFoV(img_h, img_w, fov_diag)

        foc_ud = img_h / (2 * focal_length)
        foc_lr = img_w / (2 * focal_length)

        return Volume(depth_far, -depth_near,
                      foc_ud, foc_ud,
                      foc_lr, foc_lr)

    @staticmethod
    def FromFovHW(
        *,
        fov_h: float,
        fov_w: float,
        depth_near: float,
        depth_far: float,
    ):
        assert 0 < fov_h < 180 * utils.DEG
        assert 0 < fov_w < 180 * utils.DEG

        delta_ud = math.tan(fov_h / 2)
        delta_lr = math.tan(fov_w / 2)

        return Volume(depth_far, -depth_near,
                      delta_ud, delta_ud,
                      delta_lr, delta_lr)

    @staticmethod
    def FromFovUDLR(
        *,
        fov_u: float,
        fov_d: float,
        fov_l: float,
        fov_r: float,
        depth_near: float,
        depth_far: float,
    ):
        assert -90 * utils.DEG < fov_u < 90 * utils.DEG
        assert -90 * utils.DEG < fov_d < 90 * utils.DEG
        assert -90 * utils.DEG < fov_l < 90 * utils.DEG
        assert -90 * utils.DEG < fov_r < 90 * utils.DEG

        return Volume(
            depth_far, -depth_near,
            math.tan(fov_u), math.tan(fov_d),
            math.tan(fov_l), math.tan(fov_r),
        )

    @staticmethod
    def FromFocUDLR(
        *,
        foc_u: float,
        delta_d: float,
        delta_l: float,
        delta_r: float,
        depth_near: float,
        depth_far: float,
    ):
        return Volume(depth_far, -depth_near,
                      foc_u, delta_d,
                      delta_l, delta_r)


class ProjType(enum.StrEnum):
    ORTH = "ORTH"
    PERS = "PERS"


@beartype
def MakeProjMat(
    *,
    src_coord: utils.Coord3,
    dst_coord: utils.Coord3,

    src_volume: Volume,
    dst_volume: Volume,

    proj_type: ProjType,
) -> torch.Tensor:
    std_coord = utils.Coord3(utils.Dir6.L, utils.Dir6.U, utils.Dir6.F)

    src_to_std = src_coord.GetTransTo(std_coord)
    std_to_dst = std_coord.GetTransTo(dst_coord)

    src_f = +src_volume.delta_f
    src_b = -src_volume.delta_b
    src_u = +src_volume.delta_u
    src_d = -src_volume.delta_d
    src_l = +src_volume.delta_l
    src_r = -src_volume.delta_r

    dst_f = +dst_volume.delta_f
    dst_b = -dst_volume.delta_b
    dst_u = +dst_volume.delta_u
    dst_d = -dst_volume.delta_d
    dst_l = +dst_volume.delta_l
    dst_r = -dst_volume.delta_r

    src_fb = src_f - src_b
    src_ud = src_u - src_d
    src_lr = src_l - src_r

    std_dst_points = torch.tensor([
        [dst_l,     0, dst_b, 1],
        [dst_r,     0, dst_b, 1],
        [0, dst_u, dst_b, 1],
        [0, dst_d, dst_b, 1],

        [dst_l,     0, dst_f, 1],
        [dst_r,     0, dst_f, 1],
        [0, dst_u, dst_f, 1],
        [0, dst_d, dst_f, 1],
    ], dtype=torch.float)

    match proj_type:
        case ProjType.ORTH:
            std_src_points = torch.tensor([
                [src_l, 0, src_b, 1],
                [src_r, 0, src_b, 1],
                [0, src_u, src_b, 1],
                [0, src_d, src_b, 1],

                [src_l, 0, src_f, 1],
                [src_r, 0, src_f, 1],
                [0, src_u, src_f, 1],
                [0, src_d, src_f, 1],
            ], dtype=torch.float)

            M = torch.zeros((4, 4), dtype=torch.float)

            M[0, 0] = (dst_l - dst_r) / src_lr
            M[0, 2] = (src_l * dst_r - src_r * dst_l) / src_lr / src_b

            M[1, 1] = (dst_u - dst_d) / src_ud
            M[1, 2] = (src_u * dst_d - src_d * dst_u) / src_ud / src_b

            M[2, 2] = (dst_f - dst_b) / src_fb
            M[2, 3] = (src_f * dst_b - src_b * dst_f) / src_fb

            M[3, 3] = 1

            re_std_dst_points = (M @ std_src_points.unsqueeze(-1)).squeeze(-1)

            err = utils.GetL2RMS(re_std_dst_points - std_dst_points)

            assert err <= 1e-4

        case ProjType.PERS:
            std_src_points = torch.tensor([
                [src_l * src_b, 0, src_b, 1],
                [src_r * src_b, 0, src_b, 1],
                [0, src_u * src_b, src_b, 1],
                [0, src_d * src_b, src_b, 1],

                [src_l * src_f, 0, src_f, 1],
                [src_r * src_f, 0, src_f, 1],
                [0, src_u * src_f, src_f, 1],
                [0, src_d * src_f, src_f, 1],
            ], dtype=torch.float)

            M = torch.zeros((4, 4), dtype=torch.float)

            M[0, 0] = (dst_l - dst_r) / src_lr
            M[0, 2] = (src_l * dst_r - src_r * dst_l) / src_lr

            M[1, 1] = (dst_u - dst_d) / src_ud
            M[1, 2] = (src_u * dst_d - src_d * dst_u) / src_ud

            M[2, 2] = (src_f * dst_f - src_b * dst_b) / src_fb
            M[2, 3] = (src_f * src_b) * (dst_b - dst_f) / src_fb

            M[3, 2] = 1

            re_std_dst_points = utils.DoHomo(M, std_src_points)

            err = utils.GetL2RMS(re_std_dst_points - std_dst_points)

            assert err <= 1e-4

        case _:
            assert False, f"Unknown proj type {proj_type}."

    return std_to_dst @ M @ src_to_std


class Convention(enum.StrEnum):
    OpenGL = "OpenGL"
    PyTorch3D = "Pytorch3D"
    Unity = "Unity"


@beartype
def MakeOpenGLProjMat(
    *,
    view_coord: utils.Coord3,

    target_coord: Coord,

    view_volume: Volume,

    img_h: float,
    img_w: float,

    proj_type: ProjType,
):
    assert 0 < img_h
    assert 0 < img_w

    match target_coord:
        case Coord.NDC:
            proj_coord = utils.Coord3(utils.Dir6.R, utils.Dir6.U, utils.Dir6.B)

            proj_volume = Volume(
                delta_f=1.0,
                delta_b=1.0,

                delta_u=1.0,
                delta_d=1.0,

                delta_l=1.0,
                delta_r=1.0,
            )

        case Coord.Screen:
            proj_coord = utils.Coord3(utils.Dir6.R, utils.Dir6.D, utils.Dir6.B)

            proj_volume = Volume(
                delta_f=1.0,
                delta_b=1.0,

                delta_u=img_h / 2,
                delta_d=img_h / 2,

                delta_l=img_w / 2,
                delta_r=img_w / 2,
            )

    return MakeProjMat(
        src_coord=view_coord,
        dst_coord=proj_coord,

        src_volume=view_volume,
        dst_volume=proj_volume,

        proj_type=proj_type,
    )


@beartype
def MakePytorch3DProjMat(
    *,
    view_coord: utils.Coord3,

    target_coord: Coord,

    view_volume: Volume,

    img_h: float,
    img_w: float,

    proj_type: ProjType,
):
    assert 0 < img_h
    assert 0 < img_w

    match target_coord:
        case Coord.NDC:
            proj_coord = utils.Coord3(utils.Dir6.L, utils.Dir6.U, utils.Dir6.F)

            img_s = min(img_h, img_w)

            h_ratio = img_h / img_s
            w_ratio = img_w / img_s

            proj_volume = Volume(
                delta_f=1 / view_volume.delta_f,
                delta_b=1 / view_volume.delta_b,

                delta_u=h_ratio,
                delta_d=h_ratio,

                delta_l=w_ratio,
                delta_r=w_ratio,
            )

        case Coord.Screen:
            proj_coord = utils.Coord3(utils.Dir6.R, utils.Dir6.D, utils.Dir6.F)

            proj_volume = Volume(
                delta_f=1 / view_volume.delta_f,
                delta_b=1 / view_volume.delta_b,

                delta_u=img_h / 2,
                delta_d=img_h / 2,

                delta_l=img_w / 2,
                delta_r=img_w / 2,
            )

    return MakeProjMat(
        src_coord=view_coord,
        dst_coord=proj_coord,

        src_volume=view_volume,
        dst_volume=proj_volume,

        proj_type=proj_type,
    )


@dataclasses.dataclass
class ProjSetting:
    coord: utils.Coord3
    volume: Volume


class ProtoCamera:
    def __init__(
        self,
        view_volume: Volume,
        img_h: float,
        img_w: float,
        proj_type: ProjType,
    ):
        self.view_volume = view_volume
        self.img_h = img_h
        self.img_w = img_w
        self.proj_type = proj_type

    def _GetOpenGLProjSetting(self, target_coord: Coord) -> ProjSetting:
        match target_coord:
            case Coord.NDC:
                proj_coord = utils.Coord3(
                    utils.Dir6.R, utils.Dir6.U, utils.Dir6.B)

                proj_volume = Volume(
                    delta_f=1.0,
                    delta_b=1.0,

                    delta_u=1.0,
                    delta_d=1.0,

                    delta_l=1.0,
                    delta_r=1.0,
                )

            case Coord.Screen:
                proj_coord = utils.Coord3(
                    utils.Dir6.R, utils.Dir6.D, utils.Dir6.B)

                proj_volume = Volume(
                    delta_f=1.0,
                    delta_b=1.0,

                    delta_u=self.img_h / 2,
                    delta_d=self.img_h / 2,

                    delta_l=self.img_w / 2,
                    delta_r=self.img_w / 2,
                )

        return ProjSetting(
            coord=proj_coord,
            volume=proj_volume,
        )

    def _GetPytorch3DProjSetting(self, target_coord: Coord) -> ProjSetting:
        match target_coord:
            case Coord.NDC:
                proj_coord = utils.Coord3(
                    utils.Dir6.L, utils.Dir6.U, utils.Dir6.F)

                img_s = min(self.img_h, self.img_w)

                h_ratio = self.img_h / img_s
                w_ratio = self.img_w / img_s

                proj_volume = Volume(
                    delta_f=1 / self.view_volume.delta_f,
                    delta_b=1 / self.view_volume.delta_b,

                    delta_u=h_ratio,
                    delta_d=h_ratio,

                    delta_l=w_ratio,
                    delta_r=w_ratio,
                )

            case Coord.Screen:
                proj_coord = utils.Coord3(
                    utils.Dir6.R, utils.Dir6.D, utils.Dir6.F)

                proj_volume = Volume(
                    delta_f=1 / self.view_volume.delta_f,
                    delta_b=1 / self.view_volume.delta_b,

                    delta_u=self.img_h / 2,
                    delta_d=self.img_h / 2,

                    delta_l=self.img_w / 2,
                    delta_r=self.img_w / 2,
                )

        return ProjSetting(
            coord=proj_coord,
            volume=proj_volume,
        )

    def GetProjSetting(
            self,
            convention: Convention,
            target_coord: Coord,
    ) -> ProjSetting:
        match convention:
            case Convention.OpenGL:
                return self._GetOpenGLProjSetting(target_coord)

            case Convention.PyTorch3D:
                return self._GetPytorch3DProjSetting(target_coord)

    def GetProjWithSetting(
        self,
        view_coord: utils.Coord3,
        proj_setting: ProjSetting,
    ):
        return MakeProjMat(
            src_coord=view_coord,
            dst_coord=proj_setting.coord,

            src_volume=self.view_volume,
            dst_volume=proj_setting.volume,

            proj_type=self.proj_type,
        )

    def GetProj(
        self,
        view_coord: utils.Coord3,
        convention: Convention,
        target_coord: Coord,
    ) -> torch.Tensor:  # [4, 4]
        return self.GetProjWithSetting(
            view_coord=view_coord,
            proj_setting=self.GetProjSetting(convention, target_coord),
        )
