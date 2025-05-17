from __future__ import annotations

import atexit
import collections
import datetime
import functools
import gc
import inspect
import itertools
import math
import mmap
import os
import pathlib
import pickle
import random
import shutil
import sys
import tempfile
import time
import traceback
import types
import typing
import weakref

import dateutil
import numpy as np
import torch
from beartype import beartype

from . import config

EPS = {
    np.float16: 1e-3,
    np.float32: 1e-5,
    np.float64: 1e-8,

    torch.float16: 1e-3,
    torch.float32: 1e-5,
    torch.float64: 1e-8,
}

RAD = 1.0
DEG = math.pi / 180.0

BYTE = 1
KiBYTE = 1024 * BYTE
MiBYTE = 1024 * KiBYTE
GiBYTE = 1024 * MiBYTE
TiBYTE = 1024 * GiBYTE

INT = torch.int32
FLOAT = torch.float32

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")

ORIGIN = torch.tensor([0, 0, 0], dtype=FLOAT)
X_AXIS = torch.tensor([1, 0, 0], dtype=FLOAT)
Y_AXIS = torch.tensor([0, 1, 0], dtype=FLOAT)
Z_AXIS = torch.tensor([0, 0, 1], dtype=FLOAT)


DEPTH_NEAR = 0.01
DEPTH_FAR = 100.0


class Empty:
    def __init__(self, **kwargs: object):
        for key, value in kwargs.items():
            setattr(self, key, value)


@beartype
class ArgPack:
    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        self.args = args
        self.kwargs = kwargs


@beartype
def print_pos(
    time: datetime.datetime,
    filename: str,
    lineno: int,
    funcname: str,
):
    timestr = serialize_datetime(time, "SEC")
    print(f"{timestr}\t{filename}:{lineno}\t\t{funcname}")


@beartype
def print_cur_pos() -> None:
    time = datetime.datetime.now()

    frame = inspect.currentframe().f_back

    print_pos(
        time,
        frame.f_code.co_filename,
        frame.f_lineno,
        frame.f_code.co_name,
    )


@beartype
def timestamp_sec() -> int:
    return int(time.time())


@beartype
def timestamp_msec() -> int:
    return int(time.time() * 1000)


@beartype
def timestamp_usec() -> int:
    return int(time.time() * 1000000)


@beartype
def set_add(s: set[object], obj: object) -> bool:
    old_size = len(s)
    s.add(obj)
    return old_size != len(s)


@beartype
def set_discard(s: set[object], obj: object) -> bool:
    old_size = len(s)
    s.discard(obj)
    return old_size != len(s)


@beartype
def dict_insert(
    d: dict[object, object],
    key: object,
    value: object,
) -> bool:
    old_size = len(d)
    value = d.setdefault(key, value)
    return key, value, old_size != len(d)


@beartype
def dict_pop(
    d: dict[object, object],
    key: object,
) -> bool:
    old_size = len(d)
    d.pop(key, None)
    return old_size != len(d)


@beartype
def min_max(x: typing.Any, y: typing.Any) -> tuple[typing.Any, typing.Any]:
    return (x, y) if x <= y else (y, x)


@beartype
def clamp(x: typing.Any, lb: typing.Any, ub: typing.Any) -> typing.Any:
    assert lb <= ub
    return max(lb, min(x, ub))


@beartype
def rand_int(lb: int, rb: int) -> int:
    assert lb <= rb
    return random.randint(lb, rb)


@beartype
def rand_float(lb: float, rb: float) -> float:
    assert lb <= rb
    return random.random() * (rb - lb) + lb


@beartype
def serialize_datetime(
    dt: typing.Optional[datetime.datetime],
    precision: str,  # "MIN", "SEC"
) -> str:
    if dt is None:
        return None

    precision = precision.upper()
    assert precision in ("MIN", "SEC", "USEC")

    dt = dt.replace(tzinfo=datetime.timezone.utc)

    match precision:
        case "MIN":
            format_str = "%Y-%m-%d %H:%M"

        case "SEC":
            format_str = "%Y-%m-%d %H:%M:%S"

        case _:
            raise MismatchException()

    return dt.strftime(format_str)


@beartype
def deserialize_datetime(dt_str: typing.Optional[str]) -> datetime.datetime:
    return None if dt_str is None else dateutil.parser.parse(dt_str)


class MismatchException(Exception):
    pass


@beartype
def torch_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@beartype
def _mem_clear() -> None:
    print(end="", flush=True)

    gc.collect()

    if torch.cuda.is_available():
        torch_cuda_sync()
        torch.cuda.empty_cache()


mem_clear_func_param = typing.ParamSpec("ParamSpec")
mem_clear_func_return = typing.TypeVar("TypeVar")


@beartype
def mem_clear(
    func: typing.Optional[
        typing.Callable[mem_clear_func_param, mem_clear_func_return]] = None):
    if func is None:
        _mem_clear()
        return

    @functools.wraps(func)
    def wrapper(
        *args: deferable_func_param.args,
        **kwargs: deferable_func_param.kwargs,
    ):
        _mem_clear()
        ret = func(*args, **kwargs)
        _mem_clear()
        return ret

    return wrapper


deferable_func_param = typing.ParamSpec("ParamSpec")
deferable_func_return = typing.TypeVar("TypeVar")

defer_funcs_stack: list[list[typing.Callable[[], typing.Any]]] = [list()]


@beartype
def _pop_defer_funcs() -> None:
    defer_funcs = defer_funcs_stack.pop()

    for defer_func in reversed(defer_funcs):
        try:
            defer_func()
        except Exception as e:
            print(traceback.format_exc())
            print(f"{e=}")

            print(f"{defer_func=}")

            raise e


@beartype
def deferable(
    func: typing.Callable[deferable_func_param, deferable_func_return]
) -> typing.Callable[deferable_func_param, deferable_func_return]:
    @functools.wraps(func)
    def wrapper(
        *args: deferable_func_param.args,
        **kwargs: deferable_func_param.kwargs,
    ) -> deferable_func_return:
        defer_funcs_stack.append(list())

        ret = None

        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            print(f"{e=}")

        _pop_defer_funcs()

        return ret

    return wrapper


@beartype
def defer(func: typing.Callable[[], typing.Any]) -> None:
    defer_funcs_stack[-1].append(func)


@beartype
class Timer:
    def __init__(self):
        self.beg: typing.Optional[float] = None
        self.end: typing.Optional[float] = None

        self.filename: typing.Optional[str] = None
        self.line_num: typing.Optional[int] = None
        self.function: typing.Optional[str] = None

    @property
    def duration(self):
        return -1.0 if self.beg is None or self.end is None \
            else self.end - self.beg

    def start(self):
        torch_cuda_sync()

        self.beg = time.time()
        self.end = None

    def stop(self):
        assert self.beg is not None

        torch_cuda_sync()

        self.end = time.time()

    def __enter__(self) -> Timer:
        self.start()

        frame = inspect.currentframe().f_back.f_back

        self.filename = frame.f_code.co_filename
        self.line_num = frame.f_lineno
        self.function = frame.f_code.co_name

        return self

    def __exit__(self, type, value, traceback) -> None:
        self.stop()

        print(
            f"{self.filename}:{self.line_num}\t\t{self.function}\t\tduration: {self.duration * 1000:>18.6f} ms")


@beartype
class DisableStdOut:
    def __init__(self):
        self.original_stdout = None

    def __enter__(self) -> DisableStdOut:
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        return self

    def __exit__(self, type, value, traceback) -> None:
        sys.stdout.close()
        sys.stdout = self.original_stdout
        self.original_stdout = None


@beartype
def allocate_id(
    lb: int,
    rb: int,
    is_ok: typing.Optional[typing.Callable[[int], bool]] = None,
):
    if is_ok is None:
        return rand_int(lb, rb)

    while True:
        ret = rand_int(lb, rb)

        if is_ok(ret):
            return ret


@beartype
def all_same(*args: object):
    assert 0 < len(args)

    ret = args[0]

    for arg in args:
        assert ret == arg

    return ret


@beartype
def is_almost_zeros(x: torch.Tensor, eps: float = 5e-4) -> bool:
    return bool(x.abs().max() <= eps)


@beartype
def check_almost_zeros(x: torch.Tensor, eps: float = 5e-4) -> None:
    err = x.abs().max()
    assert err <= eps, f"{err=}"


@beartype
def check_shapes(
    *args: typing.Any,
    set_zero_if_undet: bool = True,
) -> None | int | tuple[int, ...]:
    assert len(args) % 2 == 0

    undet_shapes: dict[int, int] = dict()

    for i in range(0, len(args), 2):
        obj = args[i]
        p = args[i + 1]

        t: typing.Optional[tuple[int, ...]] = None

        if obj is not None:
            if isinstance(obj, tuple):
                t = obj
            else:
                assert hasattr(obj, "shape"), f"{type(obj)}"
                t = obj.shape

            assert isinstance(t, tuple), f"{type(t)}"

        assert isinstance(p, tuple)

        p: typing.Sequence[types.EllipsisType | int]

        for p_val in p:
            assert p_val is ... or isinstance(p_val, int)

        ellipsis_cnt = p.count(...)

        assert ellipsis_cnt <= 1

        if t is None or math.prod(t) == 0:
            for p_val in p:
                if isinstance(p_val, int) and p_val < 0:
                    undet_shapes.setdefault(p_val, p_val)

            continue

        t = tuple(t) if isinstance(t, tuple) else t.shape

        if ellipsis_cnt == 0:
            assert len(p) == len(t), \
                f"Tensor dimenion {t} mismatches pattern {p}."

            t_idx_iter = range(len(t))
            p_idx_iter = range(len(p))
        else:
            assert len(p) - 1 <= len(t), \
                f"Tensor dimenion {t} mismatches pattern {p}."

            ellipsis_idx = p.index(...)

            t_idx_iter = itertools.chain(
                range(ellipsis_idx),
                range(len(t) - len(p) + ellipsis_idx + 1, len(t)))

            p_idx_iter = itertools.chain(
                range(ellipsis_idx), range(ellipsis_idx + 1, len(p)))

        for t_idx, p_idx in zip(t_idx_iter, p_idx_iter):
            t_val = t[t_idx]
            p_val = p[p_idx]

            assert isinstance(p_val, int)

            if 0 <= p_val:
                assert t_val == p_val, \
                    f"Tensor shape {t} mismatches pattern {p}."

                continue

            old_p_val = undet_shapes.setdefault(p_val, t_val)

            if old_p_val < 0:
                old_p_val = undet_shapes[p_val] = t_val

            assert old_p_val == t_val, \
                f"Tensor shape {old_p_val} and {t_val} are inconsistant."

    ret = tuple(
        max(0, undet_shape) if set_zero_if_undet else undet_shape
        for _, undet_shape in sorted(undet_shapes.items(), reverse=True)
    )

    match len(ret):
        case 0: return
        case 1: return ret[0]

    return ret


@beartype
def print_cuda_mem_usage(device: typing.Optional[torch.device] = None) -> None:
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
class DiskMemoty:
    def __init__(self, size: int):
        assert 0 <= size

        if size == 0:
            self.file = None
            self.size = 0

        self.file = tempfile.TemporaryFile(mode="w+b")
        self.size = max(4 * 1024, size)

        self.file.truncate(self.size)
        self.file.seek(0)

        self.mmap = mmap.mmap(self.file.fileno(), self.size)

    def close(self) -> None:
        if self.file is None:
            return

        self.file.close()
        self.file = None

        self.size = 0

    def __enter__(self) -> DiskMemoty:
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.close()


@beartype
def disk_empty(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    buffer_size: int = 16 * 1024,
):
    size = max(
        4 * 1024,
        math.prod(shape) * torch.tensor([], dtype=dtype).element_size(),
    )

    f = tempfile.TemporaryFile(mode="w+b", buffering=buffer_size)
    f.truncate(size)
    f.seek(0)

    m = mmap.mmap(f.fileno(), size)

    def _on_gc(x):
        del x

    weakref.finalize(m, _on_gc, f)

    return torch.frombuffer(m, dtype=dtype).view(shape)


@beartype
def empty_like(
    x: torch.Tensor,
    *,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
):
    if shape is None:
        shape = x.shape

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return torch.empty(shape, dtype=dtype, device=device)


@beartype
def zeros_like(
    x: torch.Tensor,
    *,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
):
    if shape is None:
        shape = x.shape

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return torch.zeros(shape, dtype=dtype, device=device)


@beartype
def ones_like(
    x: torch.Tensor,
    *,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
):
    if shape is None:
        shape = x.shape

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return torch.ones(shape, dtype=dtype, device=device)


@beartype
def full_like(
    x: torch.Tensor,
    full_value: object,
    *,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
):
    if shape is None:
        shape = x.shape

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return torch.full(shape, full_value, dtype=dtype, device=device)


@beartype
def eye_like(
    x: torch.Tensor,
    *,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
):
    if shape is None:
        shape = x.shape

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return eye(shape, dtype=dtype, device=device)


@beartype
def check_quaternion_order(order: str) -> str:
    assert len(order) == 4
    order = order.upper()
    assert all(d in order for d in "WXYZ")
    return order


@beartype
def get_quaternion_wxyz(
    q: torch.Tensor,  # [..., 4]
    order: str,  # wxyz
) -> tuple[
    torch.Tensor,  # w
    torch.Tensor,  # x
    torch.Tensor,  # y
    torch.Tensor,  # z
]:
    check_shapes(q, (..., 4))

    for i, k in enumerate(check_quaternion_order(order)):
        match k:
            case "W": w = q[..., i]
            case "X": x = q[..., i]
            case "Y": y = q[..., i]
            case "Z": z = q[..., i]

    return w, x, y, z


@beartype
def set_quaternion_wxyz(
    w: torch.Tensor,  # [...]
    x: torch.Tensor,  # [...]
    y: torch.Tensor,  # [...]
    z: torch.Tensor,  # [...]
    order: str,  # wxyz
    dst: torch.Tensor,  # [..., 4]
) -> None:
    check_shapes(dst, (..., 4))

    for i, k in enumerate(check_quaternion_order(order)):
        match k:
            case "W": dst[..., i] = w
            case "X": dst[..., i] = x
            case "Y": dst[..., i] = y
            case "Z": dst[..., i] = z


@beartype
def sph_to_cart(radius: float, theta: float, phi: float) \
        -> tuple[float, float, float]:
    xy_radius = radius * math.sin(theta)

    x = xy_radius * math.cos(phi)
    y = xy_radius * math.sin(phi)
    z = radius * math.cos(theta)

    return x, y, z


@beartype
def cart_to_sph(x: float, y: float, z: float) \
        -> tuple[float, float, float]:
    xy_radius_sq = x**2 + y**2
    xy_radius = math.sqrt(xy_radius_sq)

    radius = math.sqrt(xy_radius_sq + z**2)
    theta = math.atan2(xy_radius, z)
    phi = math.atan2(y, x)

    return radius, theta, phi


@beartype
def normed_idx(idx: int, length: int) -> int:
    assert -length <= idx
    assert idx < length

    return idx % length


@beartype
def broadcast_shapes(*args: typing.Any) \
        -> torch.Size:
    shapes = list()

    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, tuple):
            t = arg
        else:
            assert hasattr(arg, "shape"), f"{type(arg)}"
            t = arg.shape

        shapes.append([int(d) for d in t])

    k = max(len(shape) for shape in shapes)

    ret = torch.Size(
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
def try_get_batch_shape(x: typing.Optional[torch.Tensor], dim: int):
    return torch.Size() if x is None else x.shape[:dim]


@beartype
def get_batch_idxes(shape: typing.Iterable[int]) \
        -> typing.Iterable[tuple[int, ...]]:
    return itertools.product(*(range(s) for s in shape))


@beartype
def try_expand(x, batch_shape: tuple[int, ...]):
    return None if x is None else x.expand(batch_shape)


@beartype
def batch_expand(
    x: torch.Tensor,
    batch_shape: tuple[int, ...],
    cdim: int,
):
    ndim = len(x.shape)

    assert 0 <= cdim
    assert cdim <= ndim

    return x.expand(*batch_shape + x.shape[ndim - cdim:])


@beartype
def try_batch_expand(
    x: typing.Optional[torch.Tensor],
    batch_shape: tuple[int, ...],
    cdim: int,
) -> typing.Optional[torch.Tensor]:
    return None if x is None else batch_expand(x, batch_shape, cdim)


@beartype
def batch_indexing(
    x: typing.Any,
    batch_shape: typing.Optional[tuple[int, ...]],
    cdim: int,
    idx: typing.Any,
) -> typing.Any:
    assert x is not None

    ndim = len(x.shape)

    assert 0 <= cdim
    assert cdim <= ndim

    if batch_shape is not None:
        x = x.expand(batch_shape + x.shape[ndim - cdim:])

    if isinstance(idx, tuple):
        batch_idx = idx
    else:
        batch_idx = (idx,)

    data_idx = (slice(None) for _ in range(-cdim))

    return x[*batch_idx, ...,  *data_idx]


@beartype
def try_batch_indexing(
    x: object,
    batch_shape: typing.Optional[tuple[int, ...]],
    dims_cnt: int,
    idx,
) -> typing.Optional[torch.Tensor]:
    return None if x is None else batch_indexing(x, batch_shape, dims_cnt, idx)


@beartype
def batch_shrink(
    x: torch.Tensor,
    dim: typing.Optional[int] = None,
) -> typing.Optional[torch.Tensor]:
    dim = x.ndim if dim is None else normed_idx(dim, x.ndim)

    idx = [None for _ in range(x.ndim)]

    is_first = True

    for i in range(x.ndim):
        if dim <= i or x.shape[i] == 0 or x.stride(i) != 0:
            is_first = False
            idx[i] = slice(None)
        else:
            idx[i] = 0 if is_first else slice(0, 1)

    return x[*idx]


@beartype
def try_batch_shrink(
    x: typing.Optional[torch.Tensor],
    dim: int,
) -> typing.Optional[torch.Tensor]:
    return None if x is None else batch_shrink(x, dim)


@beartype
def ravel_idxes(
    batch_idxes: tuple[torch.Tensor, ...],
    shape,
) -> torch.Tensor:
    shape = tuple(shape)

    assert len(batch_idxes) == len(shape)

    ret = batch_idxes[-1].clone()

    m = 1

    for d in range(len(shape) - 2, -1, -1):
        m *= shape[d + 1]
        ret += m * batch_idxes[d]

    return ret


@beartype
def promote_dtypes(*args: typing.Any) -> torch.dtype:
    def _promote_dtype(
        x: typing.Optional[torch.dtype],
        y: typing.Optional[torch.dtype],
    ):
        if x is None:
            return y

        if y is None:
            return x

        return torch.promote_types(x, y)

    return functools.reduce(
        _promote_dtype,
        (arg if isinstance(arg, torch.dtype) else arg.dtype
         for arg in args if arg is not None),
        None,
    )


@beartype
def to_promoted_dtype(*args: object) -> tuple[object]:
    t = promote_dtypes(*args)
    return tuple(None if arg is None else arg.to(t) for arg in args)


@beartype
def check_devices(*args: object) -> torch.device:
    ret = None

    for arg in args:
        if arg is None:
            continue

        device = arg if isinstance(arg, torch.device) else arg.device

        if ret is None:
            ret = device
        else:
            assert ret == device

    return ret


@beartype
def rct(
    x: torch.Tensor,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return x.round().clamp(torch.iinfo(dtype).min, torch.iinfo(dtype).max).to(device, dtype)


@beartype
def sum(
    args: typing.Iterable[typing.Any],
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> typing.Any:
    if not isinstance(args, typing.Sequence):
        args = list(args)

    assert 0 < len(args)

    if dtype is None:
        dtype = promote_dtypes(
            *(arg.dtype if hasattr(arg, "dtype") else None for arg in args))

    acc = args[0].to(device, dtype)

    for i in range(1, len(args)):
        acc = acc + args[i].to(device, dtype)

    return acc


@beartype
def einsum(
    expr: str,
    *args: torch.Tensor,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = promote_dtypes(*args)

    return torch.einsum(expr, *(arg.to(device, dtype) for arg in args))


@beartype
def get_param_groups(module: object, base_lr: float):
    if hasattr(module, "get_param_groups"):
        return module.get_param_groups(base_lr)

    assert hasattr(module, "parameters")

    return [{"params": list(module.parameters()), "lr": base_lr}]


@beartype
def eye(
    shape,  # [..., N, N]
    *,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:  # [..., n, m]
    N, M = check_shapes(shape, (..., -1, -2))
    x = torch.zeros(shape, dtype=dtype, device=device)
    x.diagonal(0, -2, -1).fill_(1)
    return x


@beartype
def make_diag(
    diag_elems: torch.Tensor,  # [..., D, ...]
    dim: int = -1,
    shape: typing.Optional[tuple[int, int]] = None,  # (N, M)
) -> torch.Tensor:  # [..., N, M, ...]
    dim = normed_idx(dim, diag_elems.ndim)

    sa = diag_elems.shape[:dim]
    d = diag_elems.shape[dim]
    sb = diag_elems.shape[dim + 1:]

    if shape is None:
        shape = (d, d)

    assert 0 < shape[0]
    assert 0 < shape[1]

    zeros = zeros_like(diag_elems, shape=(1,)).expand(*sa, *sb)

    l: list[torch.Tensor] = [zeros] * (shape[0] * shape[1])

    ia = [slice(None) for _ in range(len(sa))]
    ib = [slice(None) for _ in range(len(sb))]

    for i in range(min(*shape, diag_elems.shape[-1])):
        l[(shape[1] + 1) * i] = diag_elems[*ia, i, *ib]

    return torch.stack(l, dim).reshape(*sa, *shape, *sb)


@beartype
def idx_grid(
    shape: typing.Iterable[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:  # [..., D] [i, j, 2] [i, j, k, 3], ...
    shape = tuple(shape)

    return torch.cartesian_prod(*(
        torch.arange(s, dtype=dtype, device=device)
        for s in shape
    )).reshape(*shape, len(shape))


@beartype
def rand_unit(
    size,  # [..., D]
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:  # [..., D]
    v = torch.normal(mean=0, std=1, size=size, dtype=dtype, device=device)
    return v / (vec_norm(v, -1, True) + EPS[v.dtype])


@beartype
def rand_quaternion(
    size,  # [...]
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:  # [..., 4]
    return rand_unit((*size, 4), dtype=dtype, device=device)


@beartype
def rand_rot_vec(
    size,  # [...]
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:  # [...]
    return rand_unit(size, dtype=dtype, device=device) * \
        torch.rand(size, dtype=dtype, device=device) * math.pi


@beartype
def vec_norm(
    x: torch.Tensor,  # [...]
    dim: int = -1,
    keepdim: bool = False,
    *,
    out: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:  # [...]
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim, out=out)


@beartype
def vec_sq_norm(
    x: torch.Tensor,  # [...]
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:  # [...]
    return x.square().sum(dim, keepdim)


@beartype
def vec_dot(
    x: torch.Tensor,  # [...]
    y: torch.Tensor,  # [...]
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:  # [...]
    ret: torch.Tensor = torch.linalg.vecdot(x, y, dim=dim)

    if keepdim:
        ret = ret.unsqueeze(dim)

    return ret


@beartype
def vec_cross(
    x: torch.Tensor,  # [...]
    y: torch.Tensor,  # [...]
    dim: int = -1,
) -> torch.Tensor:  # [...]
    return torch.linalg.cross(x, y, dim=dim)


@beartype
def vec_normed(
    x: torch.Tensor,  # [...]
    dim: int = -1,
    length: typing.Optional[int | float | torch.Tensor] = None,
) -> torch.Tensor:  # [...]
    x_norm = vec_norm(x, dim, True) + EPS[x.dtype]
    return x / x_norm if length is None else x * (length / x_norm)


@beartype
def get_diff(
    x: torch.Tensor,  # [...]
    y: torch.Tensor,  # [...]
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:  # [...]
    return vec_norm(x - y, dim, keepdim)


@beartype
def get_cos_angle(
    x: torch.Tensor,  # [..., D]
    y: torch.Tensor,  # [..., D]
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:  # [...] or [..., 1]
    x_norm = vec_norm(x, dim, keepdim)
    y_norm = vec_norm(y, dim, keepdim)

    z = x_norm * y_norm

    return vec_dot(x, y, dim, keepdim) / (z + EPS[z.dtype])


@beartype
def get_angle(
    x: torch.Tensor,  # [..., D]
    y: torch.Tensor,  # [..., D]
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:  # [...] or [..., 1]
    return get_cos_angle(x, y, dim, keepdim).acos()


@beartype
def axis_angle_to_quaternion(
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor] = None,  # [...]
    *,
    order: str,  # permutation of "wxyz"
) -> torch.Tensor:  # [..., 4]
    check_shapes(axis, (..., 3))

    order = check_quaternion_order(order)

    axis_norm = vec_norm(axis)

    if angle is None:
        angle = axis_norm

    unit_axis = axis / (axis_norm[..., None] + EPS[axis_norm.dtype])

    half_angle = angle / 2

    c = half_angle.cos()
    s = half_angle.sin()

    w = c
    x = unit_axis[..., 0] * s
    y = unit_axis[..., 1] * s
    z = unit_axis[..., 2] * s

    ret = empty_like(unit_axis, shape=(*x.shape, 4))

    set_quaternion_wxyz(w, x, y, z, order, ret)

    return ret


@beartype
def quaternion_to_axis_angle(
    quaternion: torch.Tensor,  # [..., 4]
    *,
    order: str,  # permutation of "wxyz"
) -> tuple[
    torch.Tensor,  # [..., 3]
    torch.Tensor,  # [...]
]:
    check_shapes(quaternion, (..., 4))

    order = check_quaternion_order(order)

    w, x, y, z = get_quaternion_wxyz(quaternion, order)

    k = 1 / vec_norm(quaternion)

    w = w * k
    x = x * k
    y = y * k
    z = z * k

    p = ((1 + EPS[w.dtype]) - w.square()).rsqrt()

    axis = empty_like(quaternion, shape=(*quaternion.shape[:-1], 3))

    torch.mul(x, p, out=axis[..., 0])
    torch.mul(y, p, out=axis[..., 1])
    torch.mul(z, p, out=axis[..., 2])

    return axis, w.acos() * 2


@beartype
def _axis_angle_to_rot_mat(
    *,
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor],  # [...]
    out: torch.Tensor,  # [..., 3, 3]
):
    check_shapes(axis, (..., 3))

    axis_norm = vec_norm(axis)

    if angle is None:
        angle = axis_norm

    unit_axis = axis / (axis_norm[..., None] + EPS[axis_norm.dtype])

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
def axis_angle_to_rot_mat(
    axis: torch.Tensor,  # [..., 3]
    angle: typing.Optional[torch.Tensor] = None,  # [...]
    *,
    out_shape: typing.Optional[tuple[int, int]] = None,  # (3/4, 3/4)
    out: typing.Optional[torch.Tensor] = None,  # [..., 3/4, 3/4]
) -> torch.Tensor:  # [..., 3/4, 3/4]
    check_shapes(axis, (..., 3))

    if out_shape is None:
        assert out is not None
        out_shape = check_shapes(out, (..., -1, -2))

    assert 3 <= out_shape[0] <= 4
    assert 3 <= out_shape[1] <= 4

    batch_shape = broadcast_shapes(axis.shape[:-1], angle)

    if out is None:
        out = torch.empty(
            batch_shape + out_shape,
            dtype=axis.dtype, device=axis.device
        )
    else:
        check_shapes(out, [..., *out_shape])

    _axis_angle_to_rot_mat(
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
def rot_mat_to_axis_angle(
    rot_mat: torch.Tensor  # [..., 3, 3]
) -> tuple[
    torch.Tensor,  # axis[..., 3]
    torch.Tensor,  # angle[..., 3]
]:
    check_shapes(rot_mat, (..., 3, 3))

    tr = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]

    k = 0.5 * (4 - (tr - 1).square()).clamp(1e-6, None).rsqrt()

    axis = torch.empty(rot_mat.shape[:-1],
                       dtype=rot_mat.dtype, device=rot_mat.device)

    axis[..., 0] = (rot_mat[..., 2, 1] - rot_mat[..., 1, 2]) * k
    axis[..., 1] = (rot_mat[..., 0, 2] - rot_mat[..., 2, 0]) * k
    axis[..., 2] = (rot_mat[..., 1, 0] - rot_mat[..., 0, 1]) * k

    return axis, ((tr - 1) / 2).acos()


@beartype
def quaternion_to_rot_mat_(
    *,
    quaternion: torch.Tensor,  # [..., 4]
    order: str,  # permutation of "wxyz"
    out: torch.Tensor,  # [..., 3, 3]
):
    check_shapes(quaternion, (..., 4))

    order = check_quaternion_order(order)

    w, x, y, z = get_quaternion_wxyz(quaternion, order)

    k = math.sqrt(2) / vec_norm(quaternion)

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
def quaternion_to_rot_mat(
    quaternion: torch.Tensor,  # [..., 4]
    *,
    order: str,  # permutation of "wxyz"
    out_shape: typing.Optional[tuple[int, int]] = None,  # (3/4, 3/4)
    out: typing.Optional[torch.Tensor] = None,  # [..., 3/4, 3/4]
) -> torch.Tensor:  # [..., 3/4, 3/4]
    check_shapes(quaternion, (..., 4))

    if out_shape is None:
        out_shape = check_shapes(out, (..., -1, -2))

    assert 3 <= out_shape[0] <= 4
    assert 3 <= out_shape[1] <= 4

    if out is None:
        out = torch.empty(
            quaternion.shape[:-1] + out_shape,
            dtype=quaternion.dtype, device=quaternion.device
        )
    else:
        check_shapes(out, [..., *out_shape])

    quaternion_to_rot_mat_(
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
def rot_mat_to_quaternion(
    rot_mat: torch.Tensor,  # [..., 3, 3]
    *,
    order: str,  # permutation of "wxyz"
    out: typing.Optional[torch.Tensor] = None,  # [..., 4]
) -> torch.Tensor:  # [..., 4]
    check_shapes(rot_mat, (..., 3, 3))

    order = check_quaternion_order(order)

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
        (*rot_mat.shape[:-2], 4),
        dtype=rot_mat.dtype, device=rot_mat.device)

    a0 = a_mat[..., 0] = (1 + tr)
    a1 = a_mat[..., 1] = (tr_n + m00 * 2)
    a2 = a_mat[..., 2] = (tr_n + m11 * 2)
    a3 = a_mat[..., 3] = (tr_n + m22 * 2)

    s0 = a0.clamp(1e-6, None).sqrt() * 2
    s1 = a1.clamp(1e-6, None).sqrt() * 2
    s2 = a2.clamp(1e-6, None).sqrt() * 2
    s3 = a3.clamp(1e-6, None).sqrt() * 2

    q_mat = torch.empty(
        (*rot_mat.shape[:-2], 4, 4),
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

    a_idxes = a_mat.argmax(-1, True)[..., None]
    a_idxes = a_idxes.expand(*a_idxes.shape[:-2], 4, 1)
    # [..., 4, 1]

    ret = q_mat.gather(-1, a_idxes)
    # [..., 4, 1]

    ret = ret[..., 0]
    # [..., 4]

    if out is None:
        out = ret
    else:
        out.copy_(ret)

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

    return out


@beartype
def quaternion_mul(
    q1: torch.Tensor,  # [..., 4]
    q2: torch.Tensor,  # [..., 4]
    *,
    order_1: str,  # wxyz
    order_2: str,  # wxyz
    order_out: str,  # wxyz

    out: typing.Optional[torch.Tensor] = None,  # [..., 4]
) -> torch.Tensor:  # [..., 4]
    batch_shape = broadcast_shapes(q1, q2)

    q1 = q1.expand(batch_shape)
    q2 = q2.expand(batch_shape)

    q1w, q1x, q1y, q1z = get_quaternion_wxyz(q1, order_1)
    q2w, q2x, q2y, q2z = get_quaternion_wxyz(q2, order_2)

    out_w = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
    out_x = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
    out_y = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
    out_z = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

    if out is None:
        out = torch.empty(batch_shape, dtype=out_w.dtype, device=out_w.device)

    set_quaternion_wxyz(out_w, out_x, out_y, out_z, order_out, out)

    return out


@beartype
def make_homo(
    x: torch.Tensor,  # [...]
    dim: int = -1,
) -> torch.Tensor:  # [...]
    dim = normed_idx(dim, x.ndim)

    shape = list(x.shape)
    shape[dim] += 1

    ret = torch.empty(shape, dtype=x.dtype, device=x.device)

    idxes = [slice(None)] * x.ndim

    idxes[dim] = slice(None, -1)
    ret[tuple(idxes)] = x

    idxes[dim] = -1
    ret[tuple(idxes)] = 1

    return ret


@beartype
def homo_normalize(
    x: torch.Tensor,  # [...]
    dim: int = -1,
) -> torch.Tensor:  # [...]
    dim = normed_idx(dim, x.ndim)

    idxes = [slice(None)] * x.ndim
    idxes[dim] = slice(-1, None)

    return x / x[tuple(idxes)]


@beartype
def mat_mul(
    *args: torch.Tensor,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
    out: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = len(args)

    assert 0 < n

    if dtype is None:
        dtype = promote_dtypes(*args)

    acc = args[0].to(device, dtype)

    if n == 1:
        return acc if out is None else out.copy_(acc.to(device, dtype))

    for i in range(1, n - 1):
        acc = torch.matmul(acc, args[i].to(device, dtype))

    return torch.matmul(acc, args[-1].to(device, dtype), out=out)


@beartype
def do_rt(
    r: torch.Tensor,  # [..., P, Q]
    t: torch.Tensor,  # [..., P]
    v: torch.Tensor,  # [..., Q]
    *,
    out: typing.Optional[torch.Tensor] = None,  # [..., P]
) -> torch.Tensor:  # out[..., P]
    P, Q = -1, -2

    P, Q = check_shapes(
        r, (..., P, Q),
        t, (..., P),
        v, (..., Q),
    )

    return torch.add(mat_mul(r, v[..., None])[..., 0], t, out=out)


@beartype
def merge_rt(
    a_r: torch.Tensor,  # [..., P, Q]
    a_t: torch.Tensor,  # [..., P]
    b_r: torch.Tensor,  # [..., Q, R]
    b_t: torch.Tensor,  # [..., Q]
    *,
    out_r: typing.Optional[torch.Tensor] = None,  # [..., P, R]
    out_t: typing.Optional[torch.Tensor] = None,  # [..., P]
) -> tuple[
    torch.Tensor,  # out_rs[..., P, R]
    torch.Tensor,  # out_ts[..., P]
]:
    P, Q, R = -1, -2, -3

    P, Q, R = check_shapes(
        a_r, (..., P, Q),
        a_t, (..., P),
        b_r, (..., Q, R),
        b_t, (..., Q),
    )

    dtype = promote_dtypes(a_r, a_t, b_r, b_t)

    a_r = a_r.to(dtype)
    a_t = a_t.to(dtype)
    b_r = b_r.to(dtype)
    b_t = b_t.to(dtype)

    out_r = mat_mul(a_r, b_r, out=out_r)
    # [..., P, R]

    out_t = torch.add(mat_mul(a_r, b_t[..., None])[..., 0], a_t, out=out_t)
    # [..., P]

    return out_r, out_t


@beartype
def get_inv_rt(
    r: torch.Tensor,  # [..., D, D]
    t: torch.Tensor,  # [..., D]
    *,
    out_r: typing.Optional[torch.Tensor] = None,  # [..., D, D]
    out_t: typing.Optional[torch.Tensor] = None,  # [..., D]
) -> tuple[
    torch.Tensor,  # out_rs[..., D, D]
    torch.Tensor,  # out_ts[..., D]
]:
    D = check_shapes(
        r, (..., -1, -1),
        t, (..., -1),
    )

    if out_r is None:
        out_r = empty_like(r)

    if out_t is None:
        out_t = torch.empty(
            (*t.shape, 1),
            dtype=promote_dtypes(r, t),
            device=check_devices(out_r, t)
        )

    torch.inverse(r, out=out_r)
    # [..., D, D]

    torch.matmul(
        out_r,
        -t[..., None],
        out=out_t,
    )
    # [..., D, 1]

    return out_r, out_t[..., 0]


@beartype
def do_homo(
    h: torch.Tensor,  # [..., P, Q]
    v: torch.Tensor,  # [..., Q]
) -> torch.Tensor:  # [..., P]
    P, Q = check_shapes(
        h, (..., -1, -2),
        v, (..., -2),
    )

    return homo_normalize(mat_mul(h, v[..., None])[..., 0])


@beartype
def get_l2_rms(
    x: torch.Tensor,  # [...]
    dim: int = -1,
):
    n = x.numel() // x.shape[dim]
    return (x.square().sum() / n).sqrt()


def dlt(
    src: torch.Tensor,  # [N, P]
    dst: torch.Tensor,  # [N, Q]
    normalize: bool = False,
    calc_err: bool = False,
) -> tuple[
    torch.Tensor,  # H[Q, P]
    float,  # err
]:
    def _get_normalize_h(
        points: torch.Tensor,  # [N, D]
        dist: float,
    ) -> torch.Tensor:  # [D+1, D+1]
        N, D = check_shapes(points, (-1, -2))

        mean = points.mean(0, True)
        # [1, D]

        odist = vec_norm(points - mean).mean()

        k = dist / odist

        h = torch.eye(D + 1, dtype=points.dtype, device=points.device) * k
        h[:-1, -1] = -k * mean[..., 0]
        h[-1, -1] = 1

        return h

    N, P, Q = -1, -2, -3

    N, P, Q = check_shapes(
        src, (N, P),
        dst, (N, Q),
    )

    assert 2 <= P
    assert 2 <= Q

    if normalize:
        src_h = _get_normalize_h(src[:, :-1], math.sqrt(P-1))
        # src_h[P, P]

        dst_h = _get_normalize_h(dst[:, :-1], math.sqrt(Q - 1))
        # dst_h[Q, Q]

        rep_src = mat_mul(src_h, src[..., None])[..., 0]
        rep_dst = mat_mul(dst_h, dst[..., None])[..., 0]
    else:
        rep_src = src
        rep_dst = dst

    A = torch.empty([N * (Q - 1), Q * P],
                    dtype=torch.promote_types(rep_src.dtype, rep_dst.dtype))

    A[:, :-P] = 0

    for q in range(Q - 1):
        A[q::Q - 1, P * q:P * q + P] = rep_src

    A[:, -P:] = (rep_src[..., None, :] * -rep_dst[:, :-1, None]) \
        .reshape(N * (Q - 1), P)
    # [N*(Q - 1), P] = (N, 1, P) * (N, Q - 1, 1) = (N, Q - 1, P) = (N*(Q - 1), P)

    Vh: torch.Tensor = torch.linalg.svd(A)[2]

    H = Vh[-1, :].reshape(Q, P)

    if normalize:
        H = mat_mul(torch.inverse(dst_h), H, src_h)

    if calc_err:
        err = math.sqrt((do_homo(H, src) - dst).square().sum() / N)
    else:
        err = -1.0

    return H, err


@beartype
def to_pathlib_path(path: os.PathLike) -> pathlib.Path:
    return path if isinstance(path, pathlib.Path) else pathlib.Path(path)


@beartype
def create_file(path: os.PathLike, mode="w", *args, **kwargs):
    path = to_pathlib_path(path)

    path.parents[0].mkdir(parents=True, exist_ok=True)

    return open(path, mode=mode, *args, **kwargs)


@beartype
def read_file(
    path: os.PathLike,
    mode: str,
):
    with open(path, mode) as f:
        return f.read()


@beartype
def write_file(
    path: os.PathLike,
    mode: str,
    data: typing.Any,
):
    with create_file(path, mode) as f:
        f.write(data)


@beartype
class PickleReader:
    def __init__(
        self,
        path: os.PathLike,
    ):
        self.f = open(path, mode="rb", buffering=128 * MiBYTE)

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> PickleReader:
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.close()

    @property
    def is_opened(self) -> bool:
        return not self.f.closed

    @property
    def mode(self) -> str:
        return self.f.mode

    def read(self):
        return pickle.load(self.f, encoding="latin1") \
            if self.is_opened else None

    def close(self) -> None:
        self.f.close()


@beartype
class PickleWriter:
    def __init__(
        self,
        path: os.PathLike,
        mode: str = "wb+",
    ):
        self.f = create_file(path, mode=mode, buffering=128 * MiBYTE)
        assert "b" in self.f.mode

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> PickleWriter:
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.close()

    @property
    def is_opened(self) -> bool:
        return not self.f.closed

    @property
    def mode(self) -> str:
        return self.f.mode

    def write(self, data: typing.Any) -> None:
        assert self.is_opened
        return pickle.dump(data, self.f)

    def close(self) -> None:
        self.f.close()


@beartype
def read_pickle(
    path: os.PathLike,
):
    with PickleReader(path) as reader:
        return reader.read()


@beartype
def write_pickle(
    path: os.PathLike,
    data: object,
    *,
    mode: str = "wb+",
) -> None:
    print(f"Writing pickle to \"{path=}\".")

    with PickleWriter(path, mode) as writer:
        writer.write(data)


@beartype
def write_tensor_to_file(path: os.PathLike, x: torch.Tensor) -> None:
    path = to_pathlib_path(path)

    path.parents[0].mkdir(parents=True, exist_ok=True)

    if x.dtype.is_floating_point:
        def to_str(val): return f"{float(val):+.7e}"
    else:
        def to_str(val): return str(int(val))

    print(f"Writing tensor to \"{path=}\".")

    with create_file(path) as f:
        def _write_tensor(cur_x: torch.Tensor) -> bytes:
            if cur_x.ndim == 0:
                f.write(to_str(cur_x.item()))
                return

            f.write("[ ")

            k = cur_x.shape[0]

            for i in range(k):
                _write_tensor(cur_x[i])

                if i < k - 1:
                    f.write(", ")

            f.write(" ]")

        _write_tensor(x)


@beartype
class edge_padding:
    def __init__(
        self,
        it: typing.Iterable[object],
        pre_n: int,
        post_n: int,
    ):
        self.it = it
        self.pre_n = pre_n
        self.post_n = post_n

        try:
            self.length = self.pre_n + len(self.it) + self.post_n
        except:
            self.length = None

    def __len__(self) -> int:
        if self.length is None:
            raise TypeError()

        return self.length

    def __iter__(self) -> typing.Iterable[object]:
        it = iter(self.it)

        try:
            item = next(it)
        except StopIteration:
            return

        for i in range(self.pre_n):
            yield item

        yield item

        for i in it:
            item = i
            yield item

        for i in range(self.post_n):
            yield item


@beartype
class slide_window:
    def __init__(
        self,
        gen: typing.Iterable[object],
        n: int,
    ):
        self.gen = gen
        self.n = n

        try:
            self.length = max(0, len(self.gen) - self.n + 1)
        except:
            self.length = None

    @beartype
    def __len__(self) -> int:
        if self.length is None:
            raise TypeError()

        return self.length

    def __iter__(self) -> typing.Iterable[list[object]]:
        it = iter(self.gen)

        d = collections.deque()

        for i in it:
            d.append(i)

            if len(d) < self.n:
                continue

            if self.n < len(d):
                d.popleft()

            yield list(d)


@beartype
def slide_window_with_padding(
    gen: typing.Iterable[object],
    n: int,
):
    r = n // 2

    return slide_window(edge_padding(gen, r, r), n)


@beartype
def smooth_clamp(
    x: torch.Tensor,
    lb: float,
    rb: float,
) -> torch.Tensor:
    assert lb < rb

    d = rb - lb

    return (x / d).sigmoid() * d + lb


_tensor_serialize_np_dtype_table = {
    torch.bool: np.bool_,

    torch.int8: np.int8,
    torch.uint8: np.uint8,

    torch.int16: np.int16,
    torch.uint16: np.uint16,

    torch.int32: np.int32,
    torch.uint32: np.uint32,

    torch.int64: np.int64,
    torch.uint64: np.uint64,

    torch.float16: np.float16,
    torch.bfloat16: np.float32,
    torch.float32: np.float32,
    torch.float64: np.float64,

    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


@beartype
def tensor_serialize(
    x: typing.Optional[torch.Tensor],
    dtype: typing.Optional[object] = None,
) -> typing.Optional[np.ndarray]:
    return None if x is None \
        else np.array(x.numpy(force=True), dtype=dtype, copy=True)


@beartype
def tensor_deserialize(
    x: typing.Optional[np.ndarray],
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> typing.Optional[torch.Tensor]:
    return None if x is None \
        else torch.from_numpy(x).to(device, dtype, copy=True)
