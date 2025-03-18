import enum
import itertools
import math
import os
import pathlib
import random
import sys
import time
import types
import typing

import einops
import torch
import torchvision
from beartype import beartype


EPS = 1e-8

RAD = 1.0
DEG = math.pi / 180.0

INT = torch.int32
FLOAT = torch.float64

CPU = torch.device("cpu")

ORIGIN = torch.tensor([0, 0, 0], dtype=torch.float)
X_AXIS = torch.tensor([1, 0, 0], dtype=torch.float)
Y_AXIS = torch.tensor([0, 1, 0], dtype=torch.float)
Z_AXIS = torch.tensor([0, 0, 1], dtype=torch.float)


class Empty:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(key, value)


@beartype
def SetAdd(s: set[object], obj: object):
    old_size = len(s)
    s.add(obj)
    return old_size != len(s)


@beartype
def SetDiscard(s: set[object], obj: object):
    old_size = len(s)
    s.discard(obj)
    return old_size != len(s)


@beartype
def DictInsert(d: dict[object, object], key: object, value: object):
    old_size = len(d)
    value = d.setdefault(key, value)
    return key, value, old_size != len(d)


@beartype
def DictPop(d: dict[object, object], key: object):
    old_size = len(d)
    d.pop(key, None)
    return old_size != len(d)


def Clamp(x, lb, ub):
    assert lb <= ub
    return max(lb, min(x, ub))


@beartype
class Timer:
    def __init__(self):
        self.beg: typing.Optional[float] = None
        self.end: typing.Optional[float] = None

        self.filename: typing.Optional[str] = None
        self.line_num: typing.Optional[int] = None
        self.function: typing.Optional[str] = None

    def duration(self):
        return -1.0 if self.beg is None or self.end is None else \
            self.end - self.beg

    def Start(self):
        torch.cuda.synchronize()

        self.beg = time.time()
        self.end = None

    def Stop(self):
        assert self.beg is not None

        torch.cuda.synchronize()
        self.end = time.time()

    def __enter__(self):
        self.Start()

        frame = sys._getframe(1)

        self.filename = frame.f_code.co_filename
        self.line_num = frame.f_lineno
        self.function = frame.f_code.co_name

        return self

    def __exit__(self, type: object, value: object, traceback: object):
        self.Stop()

        print(
            f"{self.filename}:{self.line_num}\t\t{self.function}\t\tduration: {self.duration() * 1000:>18.6f} ms")


@beartype
def AllocateID(lb: int, rb: int, s=None):
    if s is None:
        return random.randint(lb, rb)

    while True:
        ret = random.randint(lb, rb)

        if ret not in s:
            return ret


@beartype
def ReadImage(path: os.PathLike, order: str):
    img = torchvision.io.read_image(
        path, torchvision.io.ImageReadMode.RGB)
    # [c, h, w]

    return einops.rearrange(img, f"c h w -> {order.lower()}")


@beartype
def WriteImage(
    path: os.PathLike,
    img: torch.Tensor,
    order: str,
):
    assert img.dim() == 3

    img = einops.rearrange(
        img.to(dtype=torch.uint8, device=CPU), f"{order.lower()} -> c h w")

    path = pathlib.Path(path)

    os.makedirs(path.parents[0], exist_ok=True)

    if path.suffix == ".png":
        torchvision.io.write_png(img, path)
        return

    if path.suffix == ".jpg" or path.suffix == ".jpeg":
        torchvision.io.write_jpeg(img, path)
        return

    assert False, f"unknown extension: {path.suffix}"


class Dir6(enum.StrEnum):
    F = "F"  # front
    B = "B"  # back

    U = "U"  # up
    D = "D"  # down

    L = "L"  # left
    R = "R"  # right

    def GetInverse(self):
        match self:
            case Dir6.F: return Dir6.B
            case Dir6.B: return Dir6.F

            case Dir6.U: return Dir6.D
            case Dir6.D: return Dir6.U

            case Dir6.L: return Dir6.R
            case Dir6.R: return Dir6.L

        assert False, f"Unknown value {self}."


@beartype
class Coord3:
    vecs = {
        Dir6.F: torch.tensor([0, 0, +1, 0], dtype=torch.float),
        Dir6.B: torch.tensor([0, 0, -1, 0], dtype=torch.float),
        Dir6.U: torch.tensor([0, +1, 0, 0], dtype=torch.float),
        Dir6.D: torch.tensor([0, -1, 0, 0], dtype=torch.float),
        Dir6.L: torch.tensor([-1, 0, 0, 0], dtype=torch.float),
        Dir6.R: torch.tensor([+1, 0, 0, 0], dtype=torch.float),
    }

    zero_vec = torch.tensor([0, 0, 0, 1], dtype=torch.float)

    def __init__(self, dir_x: Dir6, dir_y: Dir6, dir_z: Dir6):
        self.dirs = (dir_x, dir_y, dir_z)

        assert self.dirs.count(Dir6.F) + self.dirs.count(Dir6.B) == 1
        assert self.dirs.count(Dir6.U) + self.dirs.count(Dir6.D) == 1
        assert self.dirs.count(Dir6.L) + self.dirs.count(Dir6.R) == 1

        self.trans = torch.stack([
            Coord3.vecs[self.dirs[0]],
            Coord3.vecs[self.dirs[1]],
            Coord3.vecs[self.dirs[2]],
            Coord3.zero_vec,
        ], dim=1)

    @staticmethod
    def FromStr(dirs: str):
        assert len(dirs) == 3
        dirs = dirs.upper()
        return Coord3(Dir6[dirs[0]], Dir6[dirs[1]], Dir6[dirs[2]])

    def GetTransTo(self, coord3: typing.Self) -> torch.Tensor:  # [4, 4]
        return torch.inverse(coord3.trans) @ self.trans


@beartype
def CheckShapes(*args: torch.Tensor | tuple[types.EllipsisType | int, ...]):
    assert len(args) % 2 == 0

    undet_shapes: dict[int, int] = dict()

    for i in range(0, len(args), 2):
        t = args[i]
        p = args[i + 1]

        assert isinstance(t, torch.Tensor)
        assert isinstance(p, tuple)

        ellipsis_cnt = p.count(...)

        assert ellipsis_cnt <= 1

        for p_val in p:
            assert p_val is ... or isinstance(p_val, int)

        if ellipsis_cnt == 0:
            assert len(p) == t.dim(), \
                f"Tensor dimenion {t.shape} mismatches pattern {p}."

            t_idx_iter = range(t.dim())
            p_idx_iter = range(len(p))
        else:
            assert len(p) - 1 <= t.dim(), \
                f"Tensor dimenion {t.shape} mismatches pattern {p}."

            ellipsis_idx = p.index(...)

            t_idx_iter = itertools.chain(
                range(ellipsis_idx),
                range(t.dim() - len(p) + ellipsis_idx + 1, t.dim()))

            p_idx_iter = itertools.chain(
                range(ellipsis_idx), range(ellipsis_idx + 1, len(p)))

        for t_idx, p_idx in zip(t_idx_iter, p_idx_iter):
            t_val = t.shape[t_idx]
            p_val: int = p[p_idx]

            if 0 <= p_val:
                assert t_val == p_val, \
                    f"Tensor shape {t.shape} mismatches pattern {p}."
            else:
                old_p_val = undet_shapes.setdefault(
                    p_val, t_val)

                assert old_p_val == t_val, \
                    f"Tensor shape {old_p_val} and {t_val} are inconsistant."

    return tuple(
        undet_shape
        for _, undet_shape in sorted(undet_shapes.items(), reverse=True)
    )


def AssertXYZAxes(axes: str):
    s = {
        "-x", "+x",
        "-y", "+y",
        "-z", "+z",
    }

    assert len(axes) == 6
    assert axes[0:2] in s
    assert axes[2:4] in s
    assert axes[4:6] in s

    assert axes.count("x") == 1
    assert axes.count("y") == 1
    assert axes.count("z") == 1


def GetInvAxis(axis: str):
    if axis[0] == "+":
        return f"-{axis[1:]}"

    if axis[0] == "-":
        return f"+{axis[1:]}"

    assert False


def ArrangeXYZ(a: object,
               b: object,
               c: object,
               src_axes: str,
               dst_axes: str):
    AssertXYZAxes(src_axes)
    AssertXYZAxes(dst_axes)

    values = {
        src_axes[0:2]: a,
        src_axes[2:4]: b,
        src_axes[4:6]: c,
    }

    return \
        values[dst_axes[0:2]] \
        if dst_axes[0:2] in values else -values[GetInvAxis(dst_axes[0:2])], \
        values[dst_axes[2:4]] \
        if dst_axes[2:4] in values else -values[GetInvAxis(dst_axes[2:4])], \
        values[dst_axes[4:6]]\
        if dst_axes[4:6] in values else -values[GetInvAxis(dst_axes[4:6])]


@beartype
def Sph2Cart(radius: float, theta: float, phi: float, axes: str):
    AssertXYZAxes(axes)

    xy_radius = radius * math.sin(theta)

    x = xy_radius * math.cos(phi)
    y = xy_radius * math.sin(phi)
    z = radius * math.cos(theta)

    return ArrangeXYZ(x, y, z, "+x+y+z", axes)


@beartype
def Sph2XYZ(radius: float, theta: float, phi: float, x_axis, y_axis, z_axis):
    x, y, z = Sph2Cart(radius, theta, phi, "+x+y+z")
    return x * x_axis + y * y_axis + z * z_axis


@beartype
def Cart2Sph(x: float, y: float, z: float):
    xy_radius_sq = x**2 + y**2
    xy_radius = math.sqrt(xy_radius_sq)

    radius = math.sqrt(xy_radius_sq + z**2)
    theta = math.atan2(xy_radius, z)
    phi = math.atan2(y, x)

    return radius, theta, phi


@beartype
def NormalizedIdx(idx: int, length: int):
    assert -length <= idx
    assert idx < length

    return idx % length


@beartype
def GetCommonShape(shapes: typing.Iterable[typing.Iterable[int]]):
    shape_mat = [[int(d) for d in shape] for shape in shapes]

    k = max(len(shape) for shape in shape_mat)

    ret = tuple(
        max((shape[i] if i < len(shape) else 1)
            for shape in shape_mat)
        for i in range(k)
    )

    assert all(all(
        shape[i] == 1 or shape[i] == ret[i]
        for i in range(-1, -1 - len(shape), -1))
        for shape in shape_mat)

    return ret


@beartype
def BatchEye(
    batch_shape,
    n: int,
    *,
    dtype: torch.dtype = None,
    device: torch.device = None,
):
    ret = torch.empty(list(batch_shape) + [n, n], dtype=dtype, device=device)
    ret[..., :, :] = torch.eye(n, dtype=dtype)
    return ret


@beartype
def RandUnit(
    size,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    v = torch.normal(mean=0, std=1, size=size, dtype=dtype, device=device)
    return v / (EPS + VectorNorm(v, keepdim=True))


@beartype
def RandRotVec(
    size,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
):
    return RandUnit(size, dtype=dtype, device=device) * \
        torch.rand(size, dtype=dtype, device=device) * math.pi


@beartype
def Dot(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
):
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


@beartype
def VectorNorm(
    x: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)


@beartype
def Normalized(
    x: torch.Tensor,
    dim: int = -1,
    length: typing.Optional[int | float | torch.Tensor] = None,
):
    norm = (EPS + VectorNorm(x, dim=dim, keepdim=True))
    return x / norm if length is None else x * (length / norm)


@beartype
def GetDiff(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
):
    return VectorNorm(x - y, dim=dim, keepdim=keepdim)


@beartype
def GetCosAngle(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
):
    x_norm = VectorNorm(x, dim=dim, keepdim=keepdim)
    y_norm = VectorNorm(y, dim=dim, keepdim=keepdim)

    return Dot(x, y, dim=dim, keepdim=keepdim) / (EPS + x_norm * y_norm)


@beartype
def GetAngle(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
):
    return GetCosAngle(x, y, dim=dim, keepdim=keepdim).acos()


@beartype
def BoolMatMul(
    x: torch.Tensor,  # [..., P, Q]
    y: torch.Tensor,  # [..., Q, R]
) -> torch.Tensor:  # [..., P, R]
    P, Q, R = -1, -2, -3

    P, Q, R = CheckShapes(
        x, (..., P, Q),
        y, (..., Q, R),
    )

    x = x.unsqueeze(-2)
    # [..., P, 1, Q]

    y = y.transpose(-2, -1).unsqueeze(-3)
    # [..., 1, R, Q]

    return (x & y).max(dim=-1)[0]


@beartype
def AxisAngleToQuaternion(
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor] = None,  # [...]
    *,
    order: str,  # permutation of "xyzw"
) -> torch.Tensor:  # [..., 4]
    CheckShapes(axis, (..., 3))

    assert len(order) == 4

    order = order.lower()

    assert all(d in order for d in "xyzw")

    norm = VectorNorm(axis)

    if angle is None:
        angle = norm

    unit_axis = axis / (EPS + norm.unsqueeze(-1))

    half_angle = angle / 2

    c = half_angle.cos()
    s = half_angle.sin()

    x = unit_axis[..., 0] * s
    y = unit_axis[..., 1] * s
    z = unit_axis[..., 2] * s
    w = c

    ret = torch.empty(list(x.shape) + [4],
                      dtype=unit_axis.dtype, device=unit_axis.device)

    for i, d in enumerate(order):
        match d:
            case "x": ret[..., i] = x
            case "y": ret[..., i] = y
            case "z": ret[..., i] = z
            case "w": ret[..., i] = w

    return ret


@beartype
def QuaternionToAxisAngle(
    quaternion: torch.Tensor,  # [..., 4]
    *,
    order: str,  # permutation of "xyzw"
) -> tuple[
    torch.Tensor,  # [..., 3]
    torch.Tensor,  # [...]
]:
    CheckShapes(quaternion, (..., 4))

    assert len(order) == 4

    order = order.lower()

    assert all(d in order for d in "xyzw")

    k = 1 / VectorNorm(quaternion)

    x: torch.Tensor = None
    y: torch.Tensor = None
    z: torch.Tensor = None
    w: torch.Tensor = None

    for i, d in enumerate(order):
        match d:
            case "x": x = quaternion[..., i] * k
            case "y": y = quaternion[..., i] * k
            case "z": z = quaternion[..., i] * k
            case "w": w = quaternion[..., i] * k

    p = ((1 + EPS) - w.square()).rsqrt()

    axis = torch.empty(
        list(quaternion.shape[:-1]) + [3],
        dtype=quaternion.dtype, device=quaternion.device)

    axis[..., 0] = x * p
    axis[..., 1] = y * p
    axis[..., 2] = z * p

    return axis, w.acos() * 2


@beartype
def AxisAngleToRotMat(
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor] = None,  # [...]
    *,
    homo: bool,
) -> torch.Tensor:  # [..., 3, 3] or [..., 4, 4]
    CheckShapes(axis, (..., 3))

    norm = VectorNorm(axis)

    if angle is None:
        angle = norm

    unit_axis = axis / (EPS + norm.unsqueeze(-1))

    batch_shape = list(GetCommonShape([unit_axis.shape[:-1], angle.shape]))

    c = angle.cos()
    s = angle.sin()

    nc = 1 - c

    vx = unit_axis[..., 0]
    vy = unit_axis[..., 1]
    vz = unit_axis[..., 2]

    vxx_nc = vx.square() * nc
    vyy_nc = vy.square() * nc
    vzz_nc = vz.square() * nc

    vxy_nc = vyx_nc = vx * vy * nc
    vyz_nc = vzy_nc = vy * vz * nc
    vzx_nc = vxz_nc = vz * vx * nc

    vxs = vx * s
    vys = vy * s
    vzs = vz * s

    if homo:
        ret = torch.empty(
            batch_shape + [4, 4],
            dtype=axis.dtype, device=axis.device)

        ret[..., :3, 3] = 0
        ret[..., 3, :3] = 0
        ret[..., 3, 3] = 1
    else:
        ret = torch.empty(
            batch_shape + [3, 3],
            dtype=axis.dtype, device=axis.device)

    ret[..., 0, 0] = vxx_nc + c
    ret[..., 0, 1] = vxy_nc - vzs
    ret[..., 0, 2] = vxz_nc + vys

    ret[..., 1, 0] = vyx_nc + vzs
    ret[..., 1, 1] = vyy_nc + c
    ret[..., 1, 2] = vyz_nc - vxs

    ret[..., 2, 0] = vzx_nc - vys
    ret[..., 2, 1] = vzy_nc + vxs
    ret[..., 2, 2] = vzz_nc + c

    return ret


@beartype
def RotMatToAxisAngle(
    rot_mat: torch.Tensor  # [..., 3, 3]
) -> tuple[
    torch.Tensor,  # axis[..., 3]
    torch.Tensor,  # angle[..., 3]
]:
    CheckShapes(rot_mat, (..., 3, 3))

    tr = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]

    k = 0.5 * (4 - (tr - 1).square()).clamp(min=EPS).rsqrt()

    axis = torch.empty(rot_mat.shape[:-1],
                       dtype=rot_mat.dtype, device=rot_mat.device)

    axis[..., 0] = (rot_mat[..., 2, 1] - rot_mat[..., 1, 2]) * k
    axis[..., 1] = (rot_mat[..., 0, 2] - rot_mat[..., 2, 0]) * k
    axis[..., 2] = (rot_mat[..., 1, 0] - rot_mat[..., 0, 1]) * k

    return axis, ((tr - 1) / 2).acos()


@beartype
def QuaternionToRotMat(
    quaternion: torch.Tensor,  # [..., 4]
    *,
    order: str,  # permutation of "xyzw"
    homo: bool,
) -> torch.Tensor:  # [..., 3, 3] or [..., 4, 4]
    CheckShapes(quaternion, (..., 4))

    assert len(order) == 4

    order = order.lower()

    assert all(d in order for d in "xyzw")

    k = math.sqrt(2) / VectorNorm(quaternion)

    x: torch.Tensor = None
    y: torch.Tensor = None
    z: torch.Tensor = None
    w: torch.Tensor = None

    for i, d in enumerate(order):
        match d:
            case "x": x = quaternion[..., i] * k
            case "y": y = quaternion[..., i] * k
            case "z": z = quaternion[..., i] * k
            case "w": w = quaternion[..., i] * k

    xx = x.square()
    yy = y.square()
    zz = z.square()

    xy = yx = x * y
    yz = zy = y * z
    zx = xz = z * x

    xw = x * w
    yw = y * w
    zw = z * w

    if homo:
        ret = torch.empty(
            list(quaternion.shape[:-1]) + [4, 4],
            dtype=quaternion.dtype, device=quaternion.device)

        ret[..., :3, 3] = 0
        ret[..., 3, :3] = 0
        ret[..., 3, 3] = 1
    else:
        ret = torch.empty(
            list(quaternion.shape[:-1]) + [3, 3],
            dtype=quaternion.dtype, device=quaternion.device)

    ret[..., 0, 0] = 1 - yy - zz
    ret[..., 0, 1] = xy - zw
    ret[..., 0, 2] = xz + yw

    ret[..., 1, 0] = yx + zw
    ret[..., 1, 1] = 1 - zz - xx
    ret[..., 1, 2] = yz - xw

    ret[..., 2, 0] = zx - yw
    ret[..., 2, 1] = zy + xw
    ret[..., 2, 2] = 1 - xx - yy

    return ret


@beartype
def RotMatToQuaternion(
    rot_mat: torch.Tensor,  # [..., 3, 3]
    *,
    order: str  # permutation of "xyzw"
) -> torch.Tensor:  # [..., 4]
    CheckShapes(rot_mat, (..., 3, 3))

    assert len(order) == 4

    order = order.lower()

    assert all(d in order for d in "xyzw")

    tr = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]

    k = (1 + tr).clamp(min=EPS).sqrt()
    l = 0.5 / k

    x = (rot_mat[..., 2, 1] - rot_mat[..., 1, 2]) * l
    y = (rot_mat[..., 0, 2] - rot_mat[..., 2, 0]) * l
    z = (rot_mat[..., 1, 0] - rot_mat[..., 0, 1]) * l
    w = k / 2

    quaternion = torch.empty(
        list(rot_mat.shape[:-2]) + [4],
        dtype=rot_mat.dtype, device=rot_mat.device
    )

    for i, d in enumerate(order):
        match d:
            case "x": quaternion[..., i] = x
            case "y": quaternion[..., i] = y
            case "z": quaternion[..., i] = z
            case "w": quaternion[..., i] = w

    return quaternion


@beartype
def MakeHomo(
    x: torch.Tensor,  # [...]
    *,
    dim: int = -1
) -> torch.Tensor:  # [...]
    dim = NormalizedIdx(dim, x.dim())

    shape = list(x.shape)
    shape[dim] += 1

    ret = torch.empty(shape, dtype=x.dtype, device=x.device)

    idxes = [slice(None)] * x.dim()

    idxes[dim] = slice(None, -1)
    ret[tuple(idxes)] = x

    idxes[dim] = -1
    ret[tuple(idxes)] = 1

    return ret


@beartype
def HomoNormalize(
    x: torch.Tensor,  # [...]
    *,
    dim: int = -1
):
    dim = NormalizedIdx(dim, x.dim())

    idxes = [slice(None)] * x.dim()
    idxes[dim] = slice(-1, None)

    return x / x[tuple(idxes)]


@beartype
def DoRT(
    rs: torch.Tensor,  # [..., P, Q]
    ts: torch.Tensor,  # [..., P]
    vs: torch.Tensor,  # [..., Q]
) -> torch.Tensor:  # us[..., P]
    P, Q = -1, -2

    P, Q = CheckShapes(
        rs, (..., P, Q),
        ts, (..., P),
        vs, (..., Q),
    )

    return (rs @ vs.unsqueeze(-1)).squeeze(-1) + ts


@beartype
def MergeRT(
    a_rs: torch.Tensor,  # [..., P, Q]
    a_ts: torch.Tensor,  # [..., P]
    b_rs: torch.Tensor,  # [..., Q, R]
    b_ts: torch.Tensor,  # [..., Q]
) -> tuple[
    torch.Tensor,  # rs[..., P, R]
    torch.Tensor,  # ts[..., P]
]:
    P, Q, R = -1, -2, -3

    P, Q, R = CheckShapes(
        a_rs, (..., P, Q),
        a_ts, (..., P),
        b_rs, (..., Q, R),
        b_ts, (..., Q),
    )

    ret_rs = a_rs @ b_rs
    # [..., P, R]

    ret_ts = (a_rs @ b_ts.unsqueeze(-1)).squeeze(-1) + a_ts
    # [..., P]

    return ret_rs, ret_ts


@beartype
def GetInvRT(
    rs: torch.Tensor,  # [..., D, D]
    ts: torch.Tensor,  # [..., D]
) -> tuple[
    torch.Tensor,  # inv_rs[..., D, D]
    torch.Tensor,  # inv_ts[..., D]
]:
    D = CheckShapes(
        rs, (..., -1, -1),
        ts, (..., -1),
    )

    inv_rs = torch.inverse(rs)
    # [..., D, D]

    inv_ts = (inv_rs @ -ts.unsqueeze(-1)).squeeze(-1)
    # [..., D]

    return inv_rs, inv_ts


@beartype
def DoHomo(
    hs: torch.Tensor,  # [..., P, Q]
    vs: torch.Tensor,  # [..., Q]
) -> torch.Tensor:  # [..., P]
    P, Q = CheckShapes(
        hs, (..., -1, -2),
        vs, (..., -2),
    )

    return HomoNormalize((hs @ vs.unsqueeze(-1)).squeeze(-1))


@beartype
def GetL2RMS(
    x: torch.Tensor,  # [...]
    dim: int = -1,
):
    n = x.numel() // x.shape[dim]
    return (x.square().sum() / n).sqrt()


def GetNormalizeH(
    points: torch.Tensor,  # [N, D]
    dist: float,
) -> torch.Tensor:  # [D+1, D+1]
    N, D = CheckShapes(points, (-1, -2))

    mean = points.mean(dim=0, keepdim=True)
    # [1, D]

    odist = VectorNorm(points - mean).mean()

    k = dist / odist

    h = torch.eye(D + 1, dtype=points.dtype, device=points.device) * k
    h[:-1, -1] = -k * mean.squeeze(-1)
    h[-1, -1] = 1

    return h


def DLT(
    src: torch.Tensor,  # [N, P]
    dst: torch.Tensor,  # [N, Q]
    normalize: bool = False,
    calc_err: bool = False,
) -> tuple[
    torch.Tensor,  # H[Q, P]
    float,  # err
]:
    N, P, Q = -1, -2, -3

    N, P, Q = CheckShapes(
        src, (N, P),
        dst, (N, Q),
    )

    assert 2 <= P
    assert 2 <= Q

    if normalize:
        src_h = GetNormalizeH(src[:, :-1], math.sqrt(P-1))
        # src_h[P, P]

        dst_h = GetNormalizeH(dst[:, :-1], math.sqrt(Q-1))
        # dst_h[Q, Q]

        rep_src = (src_h @ src.unsqueeze(-1)).squeeze(-1)
        rep_dst = (dst_h @ dst.unsqueeze(-1)).squeeze(-1)
    else:
        rep_src = src
        rep_dst = dst

    A = torch.empty([N*(Q-1), Q*P],
                    dtype=torch.promote_types(rep_src.dtype, rep_dst.dtype))

    A[:, :-P] = 0

    for q in range(Q-1):
        A[q::Q-1, P*q:P*q+P] = rep_src

    A[:, -P:] = (rep_src.unsqueeze(-2) * -rep_dst[:, :-1, None]
                 ).reshape((N*(Q-1), P))
    # [N*(Q-1), P] = (N, 1, P) * (N, Q-1, 1) = (N, Q-1, P) = (N*(Q-1), P)

    Vh: torch.Tensor = torch.linalg.svd(A)[2]

    H = Vh[-1, :].reshape((Q, P))

    if normalize:
        H = torch.inverse(dst_h) @ H @ src_h

    if calc_err:
        err = math.sqrt((DoHomo(H, src) - dst).square().sum() / N)
    else:
        err = -1.0

    return H, err
