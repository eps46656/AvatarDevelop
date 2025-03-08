import math
import os
import pathlib
import sys
import time
import typing

import cv2 as cv
import numpy as np
import torch
import torchvision
from typeguard import typechecked

import config

EPS = 1e-5

RAD = 1.0
DEG = math.pi / 180.0


@typechecked
def CheckStatusChanged(func: typing.Callable[[], object],
                       probe: typing.Callable[[], object]):
    old_status = probe()
    func()
    return old_status != probe()


@typechecked
def SetAdd(s: set[object], obj: object):
    return CheckStatusChanged(lambda: s.add(obj), lambda: len(s))


@typechecked
def SetDiscard(s: set[object], obj: object):
    return CheckStatusChanged(lambda: s.discard(obj), lambda: len(s))


@typechecked
def DictInsert(d: dict[object, object], key: object, value: object):
    old_size = len(d)
    value = d.setdefault(key, value)
    return key, value, old_size != len(d)


@typechecked
def DictPop(d: dict[object, object], key: object):
    return CheckStatusChanged(lambda: d.pop(key), lambda: len(d))


def Clamp(x, lb, ub):
    assert lb <= ub
    return max(lb, min(x, ub))


@typechecked
def MakeIdxTable(l: typing.Iterable):
    ret: dict[object, int] = dict()

    for idx, x in enumerate(l):
        assert x not in ret
        ret[x] = idx

    return ret


@typechecked
def DimPermute(data: np.ndarray | torch.Tensor, src: str, dst: str):
    assert len(data.shape) == len(src)
    assert len(data.shape) == len(dst)

    src_idx_table = MakeIdxTable(src)
    dst_idx_table = MakeIdxTable(dst)
    # check unique

    if isinstance(data, np.ndarray):
        return np.transpose(
            data,
            [src_idx_table[x] for x in dst])

    if isinstance(data, torch.Tensor):
        return torch.permute(
            data,
            [src_idx_table[x] for x in dst])


@typechecked
def ReadImage(path: object, order: str):
    '''
    img = cv.cvtColor(cv.imdecode(np.fromfile(
        path, dtype=np.uint8), -1), cv.COLOR_BGR2RGB)
    '''

    img = torchvision.io.read_image(
        path, torchvision.io.ImageReadMode.RGB)
    # [c, h, w]

    return DimPermute(img, "chw", order)


@typechecked
def WriteImage(path: object, img: torch.Tensor, order: str):
    assert len(img.shape) == 3

    img_ = img.to(dtype=torch.uint8, device=torch.device("cpu"))

    img_ = DimPermute(img_, order.lower(), "chw")

    path = pathlib.Path(path)

    os.makedirs(path.parents[0], exist_ok=True)

    if path.suffix == ".png":
        torchvision.io.write_png(img_, path)
        return

    if path.suffix == ".jpg" or path.suffix == ".jpeg":
        torchvision.io.write_jpeg(img_, path)
        return

    assert False, f"unknown extension: {path.suffix}"


@typechecked
class Timer:
    def __init__(self):
        self.beg: typing.Optional[float] = None
        self.end: typing.Optional[float] = None

        self.filename = None
        self.line_num = None
        self.function = None

    def duration(self):
        return None if self.beg is None or self.end is None else \
            self.end - self.beg

    def Start(self):
        self.beg = time.time()
        self.end = None

    def Stop(self):
        assert self.beg is not None
        self.end = time.time()

    def __enter__(self):
        self.Start()

        frame = sys._getframe(1)

        self.filename = frame.f_code.co_filename
        self.line_num = frame.f_lineno
        self.function = frame.f_code.co_name

        return self

    def __exit__(self, type, value, traceback):
        self.Stop()

        print(
            f"{self.filename}:{self.line_num}\t\t{self.function}\t\tduration: {self.duration()} sec")


@typechecked
def Union(*iters: typing.Iterable):
    s = set()

    for iter in iters:
        for o in iter:
            if SetAdd(s, o):
                yield o


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


def ArrangeXYZ(a, b, c, src_axes: str, dst_axes: str):
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
        values[dst_axes[4:6]] \
        if dst_axes[4:6] in values else -values[GetInvAxis(dst_axes[4:6])], \



@typechecked
def Sph2Cart(radius: float, theta: float, phi: float, axes: str):
    AssertXYZAxes(axes)

    xy_radius = radius * math.sin(theta)

    x = xy_radius * math.cos(phi)
    y = xy_radius * math.sin(phi)
    z = radius * math.cos(theta)

    return ArrangeXYZ(x, y, z, "+x+y+z", axes)


@typechecked
def Sph2XYZ(radius: float, theta: float, phi: float, x_axis, y_axis, z_axis):
    x, y, z = Sph2Cart(radius, theta, phi, "+x+y+z")
    return x * x_axis + y * y_axis + z * z_axis


@typechecked
def Cart2Sph(x: float, y: float, z: float):
    xy_radius_sq = x**2 + y**2
    xy_radius = math.sqrt(xy_radius_sq)

    radius = math.sqrt(xy_radius_sq + z**2)
    theta = math.atan2(xy_radius, z)
    phi = math.atan2(y, x)

    return radius, theta, phi


@typechecked
def RandUnit(size):
    v = torch.normal(mean=0, std=1, size=size)
    return v / (EPS + torch.linalg.vector_norm(v, dim=-1, keepdim=True))


@typechecked
def GetAngle(
    x: torch.Tensor,  # [..., D]
    y: torch.Tensor,  # [..., D]
) -> torch.Tensor:  # [...]
    D = x.shape[-1]

    assert x.shape[-1] == D
    assert y.shape[-1] == D

    x_norm = torch.linalg.vector_norm(x, dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1)

    return torch.acos(torch.einsum("...d,...d->...", x, y) /
                      (EPS + x_norm * y_norm))


@typechecked
def FindTransMat(
    D: int,  # dimention
    src_points: typing.Iterable,
    dst_points: typing.Iterable,
):
    pass


@typechecked
def GetRotMat(
    axis: torch.Tensor,  # [..., 3]
    angle: torch.Tensor | None = None,  # [...]
):
    assert 1 <= len(axis)

    assert axis.shape[-1] == 3

    norm = torch.linalg.vector_norm(axis, dim=-1)

    if angle is None:
        angle = norm

    unit_axis = axis / (EPS + norm.unsqueeze(-1))

    c = torch.cos(angle)
    s = torch.sin(angle)

    nc = 1 - c

    vx = unit_axis[..., 0]
    vy = unit_axis[..., 1]
    vz = unit_axis[..., 2]

    ret = torch.empty(list(axis.shape[:-1]) + [3, 3],
                      dtype=axis.dtype, device=axis.device)

    vxx_nc = vx**2 * nc
    vyy_nc = vy**2 * nc
    vzz_nc = vz**2 * nc

    vxy_nc = vyx_nc = vx * vy * nc
    vyz_nc = vzy_nc = vy * vz * nc
    vzx_nc = vxz_nc = vz * vx * nc

    vxs = vx * s
    vys = vy * s
    vzs = vz * s

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


@typechecked
def DoRT(
    rs: torch.Tensor,  # [..., P, Q],
    ts: torch.Tensor,  # [..., P],
    vs: torch.Tensor,  # [..., Q],
):
    assert 2 <= len(rs.shape)
    assert 1 <= len(ts.shape)
    assert 1 <= len(vs.shape)

    P, Q = rs.shape[-2:]

    assert rs.shape[-2:] == (P, Q)
    assert ts.shape[-1] == P
    assert vs.shape[-1] == Q

    return (rs @ vs.unsqueeze(-1)).squeeze(-1) + ts


@typechecked
def MergeRT(
    a_rs: torch.Tensor,  # [..., P, Q],
    a_ts: torch.Tensor,  # [..., P],
    b_rs: torch.Tensor,  # [..., Q, R],
    b_ts: torch.Tensor,  # [..., Q],
) -> tuple[torch.Tensor, torch.Tensor]:  # [..., P, R], [..., P]
    assert 2 <= len(a_rs.shape)
    assert 1 <= len(a_ts.shape)
    assert 2 <= len(b_rs.shape)
    assert 1 <= len(b_ts.shape)

    P, Q = a_rs.shape[-2:]
    R = b_rs.shape[-1]

    assert a_rs.shape[-2:] == (P, Q)
    assert a_ts.shape[-1] == P
    assert b_rs.shape[-2:] == (Q, R)
    assert b_ts.shape[-1] == Q

    ret_rs = a_rs @ b_rs
    # [..., P, R]

    ret_ts = (a_rs @ b_ts.unsqueeze(-1)).squeeze(-1) + a_ts
    # [..., P]

    return ret_rs, ret_ts


@typechecked
def GetInvRT(
    rs: torch.Tensor,  # [..., D, D],
    ts: torch.Tensor,  # [..., D],
) -> tuple[torch.Tensor, torch.Tensor]:  # [..., D, D], [..., D]
    assert 2 <= len(rs.shape)
    assert 1 <= len(ts.shape)

    D = rs.shape[-1]

    assert rs.shape[-2:] == (D, D)
    assert ts.shape[-1] == D

    inv_rs = torch.inverse(rs)
    # [..., D, D]

    inv_ts = (inv_rs @ -ts.unsqueeze(-1)).squeeze(-1)
    # [..., D]

    return inv_rs, inv_ts
