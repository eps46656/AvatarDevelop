import dataclasses
import enum
import functools
import itertools
import math
import os
import pathlib
import pickle
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
FLOAT = torch.float32

CPU = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")

ORIGIN = torch.tensor([0, 0, 0], dtype=FLOAT)
X_AXIS = torch.tensor([1, 0, 0], dtype=FLOAT)
Y_AXIS = torch.tensor([0, 1, 0], dtype=FLOAT)
Z_AXIS = torch.tensor([0, 0, 1], dtype=FLOAT)


DEPTH_NEAR = 0.01
DEPTH_FAR = 100.0


class Empty:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(key, value)


@beartype
def Exit(code: int = 0):
    sys.exit(code)


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


def MinMax(x, y):
    return (x, y) if x <= y else (y, x)


def Clamp(x, lb, ub):
    assert lb <= ub
    return max(lb, min(x, ub))


@beartype
def RandInt(lb: int, rb: int):
    assert lb <= rb
    return random.randint(lb, rb)


@beartype
def RandFloat(lb: float, rb: float):
    assert lb <= rb
    return random.random() * (rb - lb) + lb


class UnimplementationError(Exception):
    pass


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
        return RandInt(lb, rb)

    while True:
        ret = RandInt(lb, rb)

        if ret not in s:
            return ret


@beartype
def CreateFile(path: os.PathLike, mode: str = "w"):
    path = pathlib.Path(path)

    os.makedirs(path.parents[0], exist_ok=True)

    return open(path, mode)


@beartype
def ReadFile(
    path: os.PathLike,
    mode: str,
):
    with open(path, mode) as f:
        return f.read()


@beartype
def WriteFile(
    path: os.PathLike,
    mode: str,
    data,
):
    with CreateFile(path, mode) as f:
        f.write(data)


@beartype
def ReadPickle(
    path: os.PathLike,
    *,
    mode: str = "rb",
    encoding: str = "latin1",
):
    with open(path, mode=mode) as f:
        return pickle.load(f, encoding=encoding)


@beartype
def WritePickle(
    path: os.PathLike,
    data,
    *,
    mode: str = "wb+",
):
    path = pathlib.Path(path)

    os.makedirs(path.parents[0], exist_ok=True)

    with open(path, mode=mode) as f:
        pickle.dump(data, f)


@beartype
def ReadImage(path: os.PathLike):
    img = torchvision.io.decode_image(
        path, torchvision.io.ImageReadMode.RGB)
    # [C, H, W]

    return img


@beartype
def WriteImage(
    path: os.PathLike,
    img: torch.Tensor,  # [C, H, W]
):
    assert img.dim() == 3

    path = pathlib.Path(path)

    os.makedirs(path.parents[0], exist_ok=True)

    if path.suffix == ".png":
        torchvision.io.write_png(img, path)
        return

    if path.suffix == ".jpg" or path.suffix == ".jpeg":
        torchvision.io.write_jpeg(img, path)
        return

    assert False, f"unknown extension: {path.suffix}"


@beartype
def ReadVideo(path: os.PathLike):
    video, audio, d = torchvision.io.read_video(
        path,
        output_format="TCHW",
        pts_unit="sec",
    )
    # [T, C, H, W]

    return video, audio, d


@beartype
def WriteVideo(
    path: os.PathLike,
    video: torch.Tensor,  # [T, C, H, W]
    fps: int,
    codec: str = "h264",
):
    assert video.dtype == torch.uint8

    torchvision.io.write_video(
        filename=path,
        video_array=einops.rearrange(video, "t c h w -> t h w c"),
        fps=fps,
        video_codec=codec,
    )


@beartype
def ImageNormalize(img: torch.Tensor):
    return torch.div(img, 255, out=torch.empty_like(img, dtype=FLOAT))


@beartype
def ImageDenormalize(img: torch.Tensor):
    return (img * 255).round().clamp(0, 255).to(dtype=torch.uint8)


"""
@beartype
def DataclassStr(x):
    field_names: list[str] = list()
    field_vals = list()

    for field in dataclasses.fields(x):
        field_names.append(field.name)
        field_vals.append(getattr(x, field.name))

    k = max(len(field_name) for field_name in field_names)

    return "\n" + "\n".join((
        f"\t {field_name.rjust(k)} = {field_val}"
        for field_name, field_val in zip(field_names, field_vals)
    )) + "\n"""


@beartype
def IsZeros(x: torch.Tensor):
    return x.abs().max() <= 5e-4


@beartype
def ToEinopsOrder(order: str):
    order = order.upper()
    assert order.isalpha()
    return " ".join(order)


class Dir(enum.StrEnum):
    F = "F"  # front
    B = "B"  # back

    U = "U"  # up
    D = "D"  # down

    L = "L"  # left
    R = "R"  # right

    def GetInversed(self):
        match self:
            case Dir.F: return Dir.B
            case Dir.B: return Dir.F

            case Dir.U: return Dir.D
            case Dir.D: return Dir.U

            case Dir.L: return Dir.R
            case Dir.R: return Dir.L

        assert False, f"Unknown value {self}."


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


@beartype
def GetIdxes(shape):
    return itertools.product(*(range(s) for s in shape))


@beartype
def PrintCudaMemUsage(device=None):
    mem = torch.cuda.memory_allocated(device) * 1.0

    unit = "Bytes"

    if 1024 <= mem:
        unit = "KiB"
        mem /= 1024

    if 1024 <= mem:
        unit = "MiB"
        mem /= 1024

    if 1024 <= mem:
        unit = "GiB"
        mem /= 1024

    print(f"Cuda Mem Usage ({device}): {mem} {unit}")


@beartype
def CheckQuaternionOrder(order: str):
    assert len(order) == 4
    order = order.upper()
    assert all(d in order for d in "WXYZ")
    return order


@beartype
def GetQuaternionWXYZ(
    q: torch.Tensor,  # [..., 4]
    order: str,  # wxyz
):
    CheckShapes(q, (..., 4))

    w, x, y, z = None, None, None, None

    for i, k in enumerate(CheckQuaternionOrder(order)):
        match k:
            case "W": w = q[..., i]
            case "X": x = q[..., i]
            case "Y": y = q[..., i]
            case "Z": z = q[..., i]

    return w, x, y, z


@beartype
def SetQuaternionWXYZ(
    w: torch.Tensor,  # [...]
    x: torch.Tensor,  # [...]
    y: torch.Tensor,  # [...]
    z: torch.Tensor,  # [...]
    order: str,  # wxyz
    dst: torch.Tensor,  # [..., 4]
):
    CheckShapes(dst, (..., 4))

    for i, k in enumerate(CheckQuaternionOrder(order)):
        match k:
            case "W": dst[..., i] = w
            case "X": dst[..., i] = x
            case "Y": dst[..., i] = y
            case "Z": dst[..., i] = z


@beartype
def Sph2Cart(radius: float, theta: float, phi: float):
    xy_radius = radius * math.sin(theta)

    x = xy_radius * math.cos(phi)
    y = xy_radius * math.sin(phi)
    z = radius * math.cos(theta)

    return x, y, z


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
def BroadcastShapes(*args: torch.Tensor | typing.Iterable[int]):
    shapes = [
        [int(d) for d in (arg.shape if isinstance(arg, torch.Tensor) else arg)]
        for arg in args
    ]

    k = max(len(shape) for shape in shapes)

    ret = tuple(
        max((shape[i - k + len(shape)] if 0 <= i - k + len(shape) else 1)
            for shape in shapes)
        for i in range(k)
    )

    if all(all(
            shape[i] == 1 or shape[i] == ret[i]
            for i in range(-1, -1 - len(shape), -1))
            for shape in shapes):
        return ret

    print(f"{shapes=}")
    print(f"{ret=}")

    assert False


@beartype
def PromoteTypes(*args: torch.Tensor | torch.dtype):
    return functools.reduce(
        torch.promote_types,
        (arg if isinstance(arg, torch.dtype) else arg.dtype for arg in args),
        torch.bool,
    )


@beartype
def CheckDevice(*args: torch.Tensor | torch.device):
    def GetDevice(x):
        return x if isinstance(x, torch.device) else x.device

    ret = GetDevice(args[0])

    assert all(GetDevice(arg) == ret for arg in args)

    return ret


@beartype
def BatchEye(
    batch_shape,
    n: int,
    *,
    dtype: torch.dtype = None,
    device: torch.device = None,
):
    ret = torch.empty(batch_shape + (n, n), dtype=dtype, device=device)
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
    return v / (EPS + VectorNorm(v, -1, True))


@beartype
def RandQuaternion(
    size,  # [...]
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
):
    return RandUnit(size + (4,), dtype=dtype, device=device)


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
    dim: int = -1,
    keepdim: bool = False,
):
    return (x * y).sum(dim, keepdim)


@beartype
def VectorNorm(
    x: torch.Tensor,
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
    norm = (EPS + VectorNorm(x, dim, True))
    return x / norm if length is None else x * (length / norm)


@beartype
def GetDiff(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
):
    return VectorNorm(x - y, dim, keepdim)


@beartype
def GetCosAngle(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
):
    x_norm = VectorNorm(x, dim, keepdim)
    y_norm = VectorNorm(y, dim, keepdim)

    return Dot(x, y, dim, keepdim) / (EPS + x_norm * y_norm)


@beartype
def GetAngle(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
):
    return GetCosAngle(x, y, dim, keepdim).acos()


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

    return (x & y).max(-1)[0]


@beartype
def AxisAngleToQuaternion(
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor] = None,  # [...]
    *,
    order: str,  # permutation of "wxyz"
) -> torch.Tensor:  # [..., 4]
    CheckShapes(axis, (..., 3))

    order = CheckQuaternionOrder(order)

    norm = VectorNorm(axis)

    if angle is None:
        angle = norm

    unit_axis = axis / (EPS + norm.unsqueeze(-1))

    half_angle = angle / 2

    c = half_angle.cos()
    s = half_angle.sin()

    w = c
    x = unit_axis[..., 0] * s
    y = unit_axis[..., 1] * s
    z = unit_axis[..., 2] * s

    ret = torch.empty(x.shape + (4,),
                      dtype=unit_axis.dtype, device=unit_axis.device)

    SetQuaternionWXYZ(w, x, y, z, order, ret)

    return ret


@beartype
def QuaternionToAxisAngle(
    quaternion: torch.Tensor,  # [..., 4]
    *,
    order: str,  # permutation of "wxyz"
) -> tuple[
    torch.Tensor,  # [..., 3]
    torch.Tensor,  # [...]
]:
    CheckShapes(quaternion, (..., 4))

    order = CheckQuaternionOrder(order)

    w, x, y, z = GetQuaternionWXYZ(quaternion, order)

    k = 1 / VectorNorm(quaternion)

    w = w * k
    x = x * k
    y = y * k
    z = z * k

    p = ((1 + EPS) - w.square()).rsqrt()

    axis = torch.empty(
        quaternion.shape[:-1] + (3,),
        dtype=quaternion.dtype, device=quaternion.device)

    axis[..., 0] = x * p
    axis[..., 1] = y * p
    axis[..., 2] = z * p

    return axis, w.acos() * 2


@beartype
def _AxisAngleToRotMat(
    *,
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor],  # [...]
    out: torch.Tensor,  # [..., 3, 3]
):
    CheckShapes(axis, (..., 3))

    norm = VectorNorm(axis)

    if angle is None:
        angle = norm

    unit_axis = axis / (EPS + norm.unsqueeze(-1))

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

    out[..., 0, 0] = vxx_nc + c
    out[..., 0, 1] = vxy_nc - vzs
    out[..., 0, 2] = vxz_nc + vys

    out[..., 1, 0] = vyx_nc + vzs
    out[..., 1, 1] = vyy_nc + c
    out[..., 1, 2] = vyz_nc - vxs

    out[..., 2, 0] = vzx_nc - vys
    out[..., 2, 1] = vzy_nc + vxs
    out[..., 2, 2] = vzz_nc + c


@beartype
def AxisAngleToRotMat(
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor] = None,  # [...]
    *,
    out_shape: typing.Optional[tuple[int, int]] = None,  # (3/4, 3/4)
    out: typing.Optional[torch.Tensor] = None,  # [..., 3/4, 3/4]
):
    CheckShapes(axis, (..., 3))

    if out_shape is None:
        assert out is not None
        out_shape = CheckShapes(out, (..., -1, -2))

    assert 3 <= out_shape[0] <= 4
    assert 3 <= out_shape[1] <= 4

    batch_shape = tuple(axis.shape[:-1])

    if angle is not None:
        batch_shape = BroadcastShapes(batch_shape, angle)

    if out is None:
        out = torch.empty(
            batch_shape + out_shape,
            dtype=axis.dtype, device=axis.device
        )
    else:
        CheckShapes(out, [..., *out_shape])

    _AxisAngleToRotMat(
        axis=axis,
        angle=angle,
        out=out,
    )

    if out_shape == (4, 4):
        out[..., 3, 3] = 1

    if out_shape[0] == 4:
        out[..., 3, :3] = 0

    if out_shape[1] == 4:
        out[..., :3, 3] = 0

    return out


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
def QuaternionToRotMat_(
    *,
    quaternion: torch.Tensor,  # [..., 4]
    order: str,  # permutation of "wxyz"
    out: torch.Tensor,  # [..., 3, 3]
):
    CheckShapes(quaternion, (..., 4))

    order = CheckQuaternionOrder(order)

    w, x, y, z = GetQuaternionWXYZ(quaternion, order)

    k = math.sqrt(2) / VectorNorm(quaternion)

    w = w * k
    x = x * k
    y = y * k
    z = z * k

    xx = x.square()
    yy = y.square()
    zz = z.square()

    xy = yx = x * y
    yz = zy = y * z
    zx = xz = z * x

    wx = w * x
    wy = w * y
    wz = w * z

    out[..., 0, 0] = 1 - yy - zz
    out[..., 0, 1] = xy - wz
    out[..., 0, 2] = xz + wy

    out[..., 1, 0] = yx + wz
    out[..., 1, 1] = 1 - zz - xx
    out[..., 1, 2] = yz - wx

    out[..., 2, 0] = zx - wy
    out[..., 2, 1] = zy + wx
    out[..., 2, 2] = 1 - xx - yy


@beartype
def QuaternionToRotMat(
    quaternion: torch.Tensor,  # [..., 4]
    *,
    order: str,  # permutation of "wxyz"
    out_shape: typing.Optional[tuple[int, int]] = None,  # (3/4, 3/4)
    out: typing.Optional[torch.Tensor] = None,  # [..., 3/4, 3/4]
):
    CheckShapes(quaternion, (..., 4))

    if out_shape is None:
        out_shape = CheckShapes(out, (..., -1, -2))

    assert 3 <= out_shape[0] <= 4
    assert 3 <= out_shape[1] <= 4

    if out is None:
        out = torch.empty(
            quaternion.shape[:-1] + out_shape,
            dtype=quaternion.dtype, device=quaternion.device
        )
    else:
        CheckShapes(out, [..., *out_shape])

    QuaternionToRotMat_(
        quaternion=quaternion,
        order=order,
        out=out,
    )

    if out_shape == (4, 4):
        out[..., 3, 3] = 1

    if out_shape[0] == 4:
        out[..., 3, :3] = 0

    if out_shape[1] == 4:
        out[..., :3, 3] = 0

    return out


@beartype
def RotMatToQuaternion(
    rot_mat: torch.Tensor,  # [..., 3, 3]
    *,
    order: str,  # permutation of "wxyz"
    out: typing.Optional[torch.Tensor] = None,  # [..., 4]
) -> torch.Tensor:  # [..., 4]
    CheckShapes(rot_mat, (..., 3, 3))

    order = CheckQuaternionOrder(order)

    for i, k in enumerate(order):
        match k:
            case "W": wi = i
            case "X": xi = i
            case "Y": yi = i
            case "Z": zi = i

    m00 = rot_mat[..., 0, 0]
    m01 = rot_mat[..., 0, 1]
    m02 = rot_mat[..., 0, 2]
    m10 = rot_mat[..., 1, 0]
    m11 = rot_mat[..., 1, 1]
    m12 = rot_mat[..., 1, 2]
    m20 = rot_mat[..., 2, 0]
    m21 = rot_mat[..., 2, 1]
    m22 = rot_mat[..., 2, 2]
    # [...]

    m10_add_m01 = m10 + m01
    m10_sub_m01 = m10 - m01

    m21_add_m12 = m21 + m12
    m21_sub_m12 = m21 - m12

    m02_add_m20 = m02 + m20
    m02_sub_m20 = m02 - m20

    tr = m00 + m11 + m22
    tr_n = 1 - tr

    a_mat = torch.empty(
        rot_mat.shape[:-2] + (4,),
        dtype=rot_mat.dtype, device=rot_mat.device)

    a0 = a_mat[..., 0] = (1 + tr)
    a1 = a_mat[..., 1] = (tr_n + m00 * 2)
    a2 = a_mat[..., 2] = (tr_n + m11 * 2)
    a3 = a_mat[..., 3] = (tr_n + m22 * 2)

    s0 = a0.sqrt() * 2
    s1 = a1.sqrt() * 2
    s2 = a2.sqrt() * 2
    s3 = a3.sqrt() * 2

    q_mat = torch.empty(
        rot_mat.shape[:-2] + (4, 4),
        dtype=rot_mat.dtype, device=rot_mat.device)

    q_mat[..., wi, 0] = s0 / 4
    q_mat[..., xi, 0] = m21_sub_m12 / s0
    q_mat[..., yi, 0] = m02_sub_m20 / s0
    q_mat[..., zi, 0] = m10_sub_m01 / s0

    q_mat[..., wi, 1] = m21_sub_m12 / s1
    q_mat[..., xi, 1] = s1 / 4
    q_mat[..., yi, 1] = m10_add_m01 / s1
    q_mat[..., zi, 1] = m02_add_m20 / s1

    q_mat[..., wi, 2] = m02_sub_m20 / s2
    q_mat[..., xi, 2] = m10_add_m01 / s2
    q_mat[..., yi, 2] = s2 / 4
    q_mat[..., zi, 2] = m21_add_m12 / s2

    q_mat[..., wi, 3] = m10_sub_m01 / s3
    q_mat[..., xi, 3] = m02_add_m20 / s3
    q_mat[..., yi, 3] = m21_add_m12 / s3
    q_mat[..., zi, 3] = s3 / 4

    a_idxes = a_mat.argmax(-1, True).unsqueeze(-1)
    a_idxes = a_idxes.expand(a_idxes.shape[:-2] + (4, 1))
    # [..., 4, 1]

    if out is None:
        out = torch.empty(
            rot_mat.shape[:-2] + (4,),
            dtype=rot_mat.dtype, device=rot_mat.device
        )

    out = out.unsqueeze(-1)
    # [..., 4] -> [..., 4, 1]

    torch.gather(q_mat, -1, a_idxes, out=out)

    """

    q_mat[..., q channel 4, choices 4]

    a_idxes[..., dummy 4, dummy 1]

    out[..., q channel 4, dummy 1]

    q_mat[
        ...,
        q channel 4,
        a_idxes[..., dummy 4, dummy 1],
    ]

    """

    out = out.squeeze(-1)
    # [4, ..., 1] -> [..., 4]

    return out


@beartype
def QuaternionMul(
    q1: torch.Tensor,  # [..., 4]
    q2: torch.Tensor,  # [..., 4]
    *,
    order_1: str,  # wxyz
    order_2: str,  # wxyz
    order_out: str,  # wxyz

    out: typing.Optional[torch.Tensor] = None,  # [..., 4]
):
    batch_shape = BroadcastShapes(q1, q2)

    q1 = q1.expand(batch_shape)
    q2 = q2.expand(batch_shape)

    if out is None:
        out = torch.empty(
            batch_shape, dtype=PromoteTypes(q1, q2), device=CheckDevice(q1, q2))

    q1w, q1x, q1y, q1z = GetQuaternionWXYZ(q1, order_1)
    q2w, q2x, q2y, q2z = GetQuaternionWXYZ(q2, order_2)

    out_w = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
    out_x = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
    out_y = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
    out_z = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

    if out is None:
        out = torch.empty(
            batch_shape,
            dtype=PromoteTypes(q1, q2), device=CheckDevice(q1, q2))

    SetQuaternionWXYZ(out_w, out_x, out_y, out_z, order_out, out)

    return out


@beartype
def MakeHomo(
    x: torch.Tensor,  # [...]
    dim: int = -1,
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
    dim: int = -1,
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
    *,
    out: typing.Optional[torch.Tensor] = None,  # [..., P]
) -> torch.Tensor:  # out[..., P]
    P, Q = -1, -2

    P, Q = CheckShapes(
        rs, (..., P, Q),
        ts, (..., P),
        vs, (..., Q),
    )

    vs = (rs @ vs.unsqueeze(-1)).squeeze(-1) + ts

    if out is None:
        out = vs
    else:
        out.copy_(vs)

    return out


@beartype
def MergeRT(
    a_rs: torch.Tensor,  # [..., P, Q]
    a_ts: torch.Tensor,  # [..., P]
    b_rs: torch.Tensor,  # [..., Q, R]
    b_ts: torch.Tensor,  # [..., Q]
    *,
    out_rs: typing.Optional[torch.Tensor] = None,  # [..., P, R]
    out_ts: typing.Optional[torch.Tensor] = None,  # [..., P]
) -> tuple[
    torch.Tensor,  # out_rs[..., P, R]
    torch.Tensor,  # out_ts[..., P]
]:
    P, Q, R = -1, -2, -3

    P, Q, R = CheckShapes(
        a_rs, (..., P, Q),
        a_ts, (..., P),
        b_rs, (..., Q, R),
        b_ts, (..., Q),
    )

    if out_rs is None:
        out_rs = a_rs @ b_rs
    else:
        torch.matmul(a_rs, b_rs, out=out_rs)

    # [..., P, R]

    inv_ts = (a_rs @ b_ts.unsqueeze(-1)).squeeze(-1) + a_ts
    # [..., P]

    if out_ts is None:
        out_ts = inv_ts
    else:
        out_ts.copy_(inv_ts)

    return out_rs, out_ts


@beartype
def GetInvRT(
    rs: torch.Tensor,  # [..., D, D]
    ts: torch.Tensor,  # [..., D]
    *,
    out_rs: typing.Optional[torch.Tensor] = None,  # [..., D, D]
    out_ts: typing.Optional[torch.Tensor] = None,  # [..., D]
) -> tuple[
    torch.Tensor,  # out_rs[..., D, D]
    torch.Tensor,  # out_ts[..., D]
]:
    D = CheckShapes(
        rs, (..., -1, -1),
        ts, (..., -1),
    )

    if out_rs is None:
        out_rs = torch.empty_like(rs)

    if out_ts is None:
        out_ts = torch.empty(
            ts.shape + (1,),
            dtype=PromoteTypes(rs, ts),
            device=CheckDevice(out_rs, ts)
        )

    torch.inverse(rs, out=out_rs)
    # [..., D, D]

    torch.matmul(
        out_rs,
        -ts.unsqueeze(-1),
        out=out_ts,
    )
    # [..., D, 1]

    return out_rs, out_ts.unsqueeze(-1)


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

    mean = points.mean(0, True)
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
