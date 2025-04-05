from __future__ import annotations

import dataclasses
import enum
import math

import torch
from beartype import beartype

from . import transform_utils, utils


class Coord(enum.StrEnum):
    World = "World"
    View = "View"
    NDC = "NDC"
    Screen = "Screen"


class ProjType(enum.StrEnum):
    ORTH = "ORTH"
    PERS = "PERS"


@beartype
def make_view(
    origin: torch.Tensor,  # [..., 3]
    aim: torch.Tensor,  # [..., 3]
    quasi_u_dir: torch.Tensor,  # [..., 3]
) -> transform_utils.ObjectTransform:
    utils.check_shapes(
        origin, (..., 3),
        aim, (..., 3),
        quasi_u_dir, (..., 3),
    )

    f_vec = utils.vec_normed(aim - origin)
    r_vec = utils.vec_normed(utils.vec_cross(f_vec, quasi_u_dir))
    u_vec = utils.vec_cross(r_vec, f_vec)

    return transform_utils.ObjectTransform.from_matching(
        dirs="FRU",
        pos=origin,
        vecs=(f_vec, r_vec, u_vec),
    )


@beartype
def make_focal_length_by_fov_diag(img_h: int, img_w: int, fov_diag: float) \
        -> float:
    assert 0 < img_h
    assert 0 < img_w

    assert 0 < fov_diag < 180 * utils.DEG

    return math.sqrt(img_h**2 + img_w**2) / (2 * math.tan(fov_diag / 2))


@beartype
class CameraConfig:
    def __init__(
        self,
        proj_type: ProjType,
        foc_u: float,
        foc_d: float,
        foc_l: float,
        foc_r: float,
        depth_near: float,
        depth_far: float,
        img_h: int,
        img_w: int,
    ):
        assert 0 < foc_u + foc_d
        assert 0 < foc_l + foc_r

        assert 0 < depth_near
        assert depth_near < depth_far

        assert 0 < img_h
        assert 0 < img_w

        self.proj_type = proj_type
        self.foc_u = foc_u
        self.foc_d = foc_d
        self.foc_l = foc_l
        self.foc_r = foc_r
        self.depth_near = depth_near
        self.depth_far = depth_far
        self.img_h = img_h
        self.img_w = img_w

    @staticmethod
    def from_fov_diag(
        *,
        fov_diag: float,
        depth_near: float,
        depth_far: float,
        img_h: int,
        img_w: int,
    ) -> CameraConfig:
        assert 0 < fov_diag < 180 * utils.DEG

        focal_length = make_focal_length_by_fov_diag(img_h, img_w, fov_diag)

        foc_ud = img_h / (2 * focal_length)
        foc_lr = img_w / (2 * focal_length)

        return CameraConfig(
            ProjType.PERS,
            foc_ud, foc_ud,
            foc_lr, foc_lr,
            depth_near, depth_far,
            img_h, img_w,
        )

    @staticmethod
    def from_fov_hw(
        *,
        fov_h: float,
        fov_w: float,
        depth_near: float,
        depth_far: float,
        img_h: int,
        img_w: int,
    ) -> CameraConfig:
        assert 0 < fov_h < 180 * utils.DEG
        assert 0 < fov_w < 180 * utils.DEG

        foc_ud = math.tan(fov_h / 2)
        foc_lr = math.tan(fov_w / 2)

        return CameraConfig(
            ProjType.PERS,
            foc_ud, foc_ud,
            foc_lr, foc_lr,
            depth_near, depth_far,
            img_h, img_w,
        )

    @staticmethod
    def from_fov_udlr(
        *,
        fov_u: float,
        fov_d: float,
        fov_l: float,
        fov_r: float,
        depth_near: float,
        depth_far: float,
        img_h: int,
        img_w: int,
    ) -> CameraConfig:
        assert -90 * utils.DEG < fov_u < 90 * utils.DEG
        assert -90 * utils.DEG < fov_d < 90 * utils.DEG
        assert -90 * utils.DEG < fov_l < 90 * utils.DEG
        assert -90 * utils.DEG < fov_r < 90 * utils.DEG

        return CameraConfig(
            ProjType.PERS,
            math.tan(fov_u), math.tan(fov_d),
            math.tan(fov_l), math.tan(fov_r),
            depth_near, depth_far,
            img_h, img_w,
        )

    @staticmethod
    def from_slope_udlr(
        *,
        slope_u: float,
        slope_d: float,
        slope_l: float,
        slope_r: float,
        depth_near: float,
        depth_far: float,
        img_h: int,
        img_w: int,
    ) -> CameraConfig:
        return CameraConfig(
            ProjType.PERS,
            slope_u, slope_d,
            slope_l, slope_r,
            depth_near, depth_far,
            img_h, img_w,
        )

    @staticmethod
    def from_delta_hw(
        *,
        delta_h: float,
        delta_w: float,
        depth_near: float,
        depth_far: float,
        img_h: int,
        img_w: int,
    ) -> CameraConfig:
        foc_ud = delta_h / 2
        foc_lr = delta_w / 2

        return CameraConfig(
            ProjType.ORTH,
            foc_ud, foc_ud,
            foc_lr, foc_lr,
            depth_near, depth_far,
            img_h, img_w,
        )

    @staticmethod
    def from_delta_udlr(
        *,
        delta_u: float,
        delta_d: float,
        delta_l: float,
        delta_r: float,
        depth_near: float,
        depth_far: float,
        img_h: int,
        img_w: int,
    ) -> CameraConfig:
        return CameraConfig(
            ProjType.ORTH,
            delta_u, delta_d,
            delta_l, delta_r,
            depth_near, depth_far,
            img_h, img_w,
        )

    @property
    def fov_h(self) -> float:
        assert self.proj_type == ProjType.PERS
        return math.atan(self.foc_u) + math.atan(self.foc_d)

    @property
    def fov_w(self) -> float:
        assert self.proj_type == ProjType.PERS
        return math.atan(self.foc_l) + math.atan(self.foc_r)

    @property
    def pixel_h(self) -> float:
        return (self.foc_u + self.foc_d) / self.img_h

    @property
    def pixel_w(self) -> float:
        return (self.foc_l + self.foc_r) / self.img_w


@beartype
@dataclasses.dataclass
class ProjConfig:
    camera_proj_transform: transform_utils.ObjectTransform
    # Describe the transform relation between camera and proj coord

    delta_u: float
    delta_d: float
    delta_l: float
    delta_r: float

    delta_f: float
    delta_b: float


class Convention(enum.StrEnum):
    OpenGL = "OpenGL"
    OpenCV = "OpenCV"
    PyTorch3D = "PyTorch3D"
    Unity = "Unity"
    Blender = "Blender"


@beartype
def make_proj_config_OpenGL(
    *,
    camera_config: CameraConfig,
    target_coord: Coord,
) -> ProjConfig:
    match target_coord:
        case Coord.NDC:
            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("RUB"),

                delta_u=1.0,
                delta_d=1.0,

                delta_l=1.0,
                delta_r=1.0,

                delta_f=1.0,
                delta_b=1.0,
            )

            return proj_config

        case Coord.Screen:
            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("RUB"),

                delta_u=camera_config.img_h / 2,
                delta_d=camera_config.img_h / 2,

                delta_l=camera_config.img_w / 2,
                delta_r=camera_config.img_w / 2,

                delta_f=1.0,
                delta_b=1.0,
            )

            return proj_config

        case _:
            raise utils.MismatchException()


@beartype
def MakeProjConfig_OpenCV(
    *,
    camera_config: CameraConfig,
    target_coord: Coord,
) -> ProjConfig:
    match target_coord:
        case Coord.NDC:
            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("RDF"),

                delta_u=1.0,
                delta_d=1.0,

                delta_l=1.0,
                delta_r=1.0,

                delta_f=+1.0 / camera_config.depth_far,
                delta_b=-1.0 / camera_config.depth_near,
            )

            return proj_config

        case Coord.Screen:
            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("RDF"),

                delta_u=0.0,
                delta_d=camera_config.img_h * 1.0,

                delta_l=0.0,
                delta_r=camera_config.img_w * 1.0,

                delta_f=+1.0 / camera_config.depth_far,
                delta_b=-1.0 / camera_config.depth_near,
            )

            return proj_config

        case _:
            raise utils.MismatchException()


@beartype
def make_proj_config_PyTorch3D(
    *,
    camera_config: CameraConfig,
    target_coord: Coord,
) -> ProjConfig:
    match target_coord:
        case Coord.NDC:
            img_s = min(camera_config.img_h, camera_config.img_w)

            h_ratio = camera_config.img_h / img_s
            w_ratio = camera_config.img_w / img_s

            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("LUF"),

                delta_u=h_ratio,
                delta_d=h_ratio,

                delta_l=w_ratio,
                delta_r=w_ratio,

                delta_f=+1.0 / camera_config.depth_far,
                delta_b=-1.0 / camera_config.depth_near,
            )

            return proj_config

        case Coord.Screen:
            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("RDF"),

                delta_u=0.0,
                delta_d=camera_config.img_h * 1.0,

                delta_l=0.0,
                delta_r=camera_config.img_w * 1.0,

                delta_f=+1.0 / camera_config.depth_far,
                delta_b=-1.0 / camera_config.depth_near,
            )

            return proj_config

        case _:
            raise utils.MismatchException()


@beartype
def MakeProjConfig_Unity(
    *,
    camera_config: CameraConfig,
    target_coord: Coord,
) -> ProjConfig:
    match target_coord:
        case Coord.NDC:
            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("RUF"),

                delta_u=1.0,
                delta_d=1.0,

                delta_l=1.0,
                delta_r=1.0,

                delta_f=+1.0 / camera_config.depth_far,
                delta_b=-1.0 / camera_config.depth_near,
            )

            return proj_config

        case Coord.Screen:
            proj_config = ProjConfig(
                camera_proj_transform=transform_utils.ObjectTransform.
                from_matching("RUF"),

                delta_u=0.0,
                delta_d=camera_config.img_h * 1.0,

                delta_l=0.0,
                delta_r=camera_config.img_w * 1.0,

                delta_f=+1.0 / camera_config.depth_far,
                delta_b=-1.0 / camera_config.depth_near,
            )

            return proj_config

        case _:
            raise utils.MismatchException()


@beartype
def make_proj_config(
    *,
    camera_config: CameraConfig,
    convention: Convention,
    target_coord: Coord,
) -> ProjConfig:
    match convention:
        case Convention.OpenGL:
            return make_proj_config_OpenGL(
                camera_config=camera_config,
                target_coord=target_coord,
            )

        case Convention.PyTorch3D:
            return make_proj_config_PyTorch3D(
                camera_config=camera_config,
                target_coord=target_coord,
            )

        case _:
            raise utils.MismatchException()


@beartype
def make_proj_mat_with_config(
    *,
    camera_config: CameraConfig,
    camera_view_transform: transform_utils.ObjectTransform,
    proj_config: ProjConfig,
    dtype: torch.dtype = utils.FLOAT,
) -> torch.Tensor:  # [4, 4]
    device = camera_view_transform.device

    camera_std_transform = \
        transform_utils.ObjectTransform.from_matching("LUF").to(device)
    # camera <-> std

    # camera_view_transform
    # camera <-> view

    # proj_config.camera_proj_transform
    # camera <-> proj

    src_to_std = camera_view_transform.get_trans_to(camera_std_transform)
    # view -> std
    # [..., 4, 4]

    std_to_dst = camera_std_transform.get_trans_to(
        proj_config.camera_proj_transform.to(device))
    # std -> proj
    # [4, 4]

    src_u = +camera_config.foc_u
    src_d = -camera_config.foc_d
    src_l = +camera_config.foc_l
    src_r = -camera_config.foc_r
    src_n = +camera_config.depth_near
    src_f = +camera_config.depth_far

    dst_u = +proj_config.delta_u
    dst_d = -proj_config.delta_d
    dst_l = +proj_config.delta_l
    dst_r = -proj_config.delta_r
    dst_f = +proj_config.delta_f
    dst_b = -proj_config.delta_b

    src_ud = src_u - src_d
    src_lr = src_l - src_r
    src_fn = src_f - src_n

    dst_ud_mid = (src_u * dst_d - src_d * dst_u) / src_ud
    dst_lr_mid = (src_l * dst_r - src_r * dst_l) / src_lr

    std_dst_points = torch.tensor([
        [dst_l, dst_ud_mid, dst_b, 1],
        [dst_r, dst_ud_mid, dst_b, 1],
        [dst_lr_mid, dst_u, dst_b, 1],
        [dst_lr_mid, dst_d, dst_b, 1],

        [dst_l, dst_ud_mid, dst_f, 1],
        [dst_r, dst_ud_mid, dst_f, 1],
        [dst_lr_mid, dst_u, dst_f, 1],
        [dst_lr_mid, dst_d, dst_f, 1],
    ], dtype=dtype)

    match camera_config.proj_type:
        case ProjType.ORTH:
            std_src_points = torch.tensor([
                [src_l, 0, src_n, 1],
                [src_r, 0, src_n, 1],
                [0, src_u, src_n, 1],
                [0, src_d, src_n, 1],

                [src_l, 0, src_f, 1],
                [src_r, 0, src_f, 1],
                [0, src_u, src_f, 1],
                [0, src_d, src_f, 1],
            ], dtype=dtype)

            M = torch.zeros((4, 4), dtype=dtype)

            M[0, 0] = (dst_l - dst_r) / src_lr
            M[0, 2] = (src_l * dst_r - src_r * dst_l) / src_lr / src_n

            M[1, 1] = (dst_u - dst_d) / src_ud
            M[1, 2] = (src_u * dst_d - src_d * dst_u) / src_ud / src_n

            M[2, 2] = (dst_f - dst_b) / src_fn
            M[2, 3] = (src_f * dst_b - src_n * dst_f) / src_fn

            M[3, 3] = 1

        case ProjType.PERS:
            std_src_points = torch.tensor([
                [src_l * src_n, 0, src_n, 1],
                [src_r * src_n, 0, src_n, 1],
                [0, src_u * src_n, src_n, 1],
                [0, src_d * src_n, src_n, 1],

                [src_l * src_f, 0, src_f, 1],
                [src_r * src_f, 0, src_f, 1],
                [0, src_u * src_f, src_f, 1],
                [0, src_d * src_f, src_f, 1],
            ], dtype=dtype)

            M = torch.zeros((4, 4), dtype=dtype)

            M[0, 0] = (dst_l - dst_r) / src_lr
            M[0, 2] = (src_l * dst_r - src_r * dst_l) / src_lr

            M[1, 1] = (dst_u - dst_d) / src_ud
            M[1, 2] = (src_u * dst_d - src_d * dst_u) / src_ud

            M[2, 2] = (src_f * dst_f - src_n * dst_b) / src_fn
            M[2, 3] = (src_f * src_n) * (dst_b - dst_f) / src_fn

            M[3, 2] = 1

        case _:
            raise utils.MismatchException()

    re_std_dst_points = utils.do_homo(M, std_src_points)

    err = utils.get_l2_rms(re_std_dst_points - std_dst_points)

    assert err <= 1e-4, f"{err=}"

    return (std_to_dst @ M.to(device)) @ src_to_std


@beartype
def make_proj_mat(
    *,
    camera_config: CameraConfig,
    camera_view_transform: transform_utils.ObjectTransform,
    convention: Convention,
    target_coord: Coord,
    dtype: torch.dtype = utils.FLOAT,
) -> torch.Tensor:  # [4, 4]
    return make_proj_mat_with_config(
        camera_config=camera_config,
        camera_view_transform=camera_view_transform,
        proj_config=make_proj_config(
            camera_config=camera_config,
            convention=convention,
            target_coord=target_coord,
        ),
        dtype=dtype,
    )
