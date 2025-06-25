from __future__ import annotations

import collections
import colorsys
import contextlib
import datetime
import functools
import gc
import inspect
import itertools
import json
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
import tabulate
import torch
from beartype import beartype

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


DISK_DEVICE = torch.device("cpu")
CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")

ORIGIN = torch.tensor([0, 0, 0], dtype=torch.float32)
X_AXIS = torch.tensor([1, 0, 0], dtype=torch.float32)
Y_AXIS = torch.tensor([0, 1, 0], dtype=torch.float32)
Z_AXIS = torch.tensor([0, 0, 1], dtype=torch.float32)


DEPTH_NEAR = 0.01
DEPTH_FAR = 100.0


class Empty:
    def __init__(self, **kwargs: object):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MismatchException(Exception):
    pass


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
def print_cur_pos(delta: int = 0) -> None:
    time = datetime.datetime.now()

    frame = inspect.currentframe().f_back.f_back

    for _ in range(delta):
        frame = frame.f_back

    print_pos(
        time,
        frame.f_code.co_filename,
        frame.f_lineno,
        frame.f_code.co_name,
    )


@beartype
def pause() -> None:
    print_cur_pos(2)
    # include this function and beartype decorator

    input("pause...")


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
def fallback_if_none(
    x: typing.Any,
    y: typing.Any,
) -> typing.Any:
    return y if x is None else x


@beartype
def rand_int(lb: int, rb: int) -> int:
    assert lb <= rb
    return random.randint(lb, rb)


@beartype
def rand_float(lb: float, rb: float) -> float:
    assert lb <= rb
    return random.random() * (rb - lb) + lb


@beartype
def make_nd_list(
    value: typing.Any,
    shape: tuple[int, ...],
) -> list:
    assert 0 < len(shape)

    if len(shape) == 1:
        return [value] * shape[0]

    sub_shape = shape[1:]

    return [make_nd_list(value, sub_shape) for _ in range(shape[0])]


_serialize_datetime_format_str_table = {
    "MIN": "%Y-%m-%d %H:%M",
    "SEC": "%Y-%m-%d %H:%M:%S",
}


@beartype
def serialize_datetime(
    dt: typing.Optional[datetime.datetime],
    precision: str,  # "MIN", "SEC"
) -> str:
    if dt is None:
        return None

    precision = precision.upper()
    assert precision in _serialize_datetime_format_str_table

    dt = dt.replace(tzinfo=datetime.timezone.utc)

    return dt.strftime(_serialize_datetime_format_str_table[precision])


@beartype
def deserialize_datetime(dt_str: typing.Optional[str]) -> datetime.datetime:
    return None if dt_str is None else dateutil.parser.parse(dt_str)


@beartype
def generate_color(
    *,
    init_hue: typing.Optional[float] = None,
    saturation: float = 0.5,
    value: float = 1.0,
) -> typing.Generator[tuple[int, int, int], None, None]:
    golden_ratio_conjugate = 0.61803398875

    hue = random.random() if init_hue is None else init_hue

    while True:
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        yield tuple(clamp(int(round(c * 255)), 0, 255) for c in rgb)
        hue = (hue + golden_ratio_conjugate) % 1.0


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
class TorchDetectAnomaly(contextlib.ContextDecorator):
    def __init__(self, *args, **kwargs):
        self.cm = torch.autograd.set_detect_anomaly(*args, **kwargs)

    def __enter__(self) -> TorchDetectAnomaly:
        self.cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.cm.__exit__(exc_type, exc_value, traceback)


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

        if t is None:
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

    if len(ret) == 0:
        return

    if len(ret) == 1:
        return ret[0]

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

    print(f"Cuda Mem Usage ({device}): {mem:.3g} {unit}")


_disk_storages: weakref.WeakSet[torch.TypedStorage] = weakref.WeakSet()


@beartype
def is_disk_tensor(x: torch.Tensor) -> bool:
    return x.untyped_storage() in _disk_storages


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

    x = torch.frombuffer(m, dtype=dtype).view(shape)

    _disk_storages.add(x.untyped_storage())

    def _on_gc(x):
        del x

    weakref.finalize(m, _on_gc, f)

    return x


@beartype
def get_sddr(
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
    requires_grad: typing.Optional[bool] = None,
) -> tuple[
    tuple[int, ...],  # shape
    torch.dtype,  # dtype
    torch.device,  # device
    bool,  # requires_grad
]:
    if shape is None:
        assert like is not None
        shape = like.shape

    if dtype is None:
        assert like is not None
        dtype = like.dtype

    if device is None:
        assert like is not None
        device = like.device

    if requires_grad is None:
        assert like is not None
        requires_grad = like.requires_grad

    return shape, dtype, device, requires_grad


@beartype
def empty(
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
    requires_grad: typing.Optional[bool] = None,
):
    shape, dtype, device, requires_grad = get_sddr(
        like, shape, dtype, device, requires_grad)

    return torch.empty(
        shape, dtype=dtype, device=device, requires_grad=requires_grad)


@beartype
def zeros(
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
    requires_grad: typing.Optional[bool] = None,
) -> torch.Tensor:
    shape, dtype, device, requires_grad = get_sddr(
        like, shape, dtype, device, requires_grad)

    return torch.zeros(
        shape, dtype=dtype, device=device, requires_grad=requires_grad)


_dummy_zeros_cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = \
    dict()


@beartype
def dummy_zeros(
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    shape, dtype, device, _ = get_sddr(
        like, shape, dtype, device, False)

    dd = (dtype, device)

    t = _dummy_zeros_cache.get(dd)

    if t is None:
        t = torch.zeros((1,), dtype=dtype, device=device)
        _dummy_zeros_cache[dd] = t

    return t.expand(shape)


@beartype
def ones(
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
    requires_grad: typing.Optional[bool] = None,
) -> torch.Tensor:
    shape, dtype, device, requires_grad = get_sddr(
        like, shape, dtype, device, requires_grad)

    return torch.ones(
        shape, dtype=dtype, device=device, requires_grad=requires_grad)


_dummy_ones_cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = \
    dict()


@beartype
def dummy_ones(
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    shape, dtype, device, _ = get_sddr(like, shape, dtype, device, False)

    dd = (dtype, device)

    t = _dummy_ones_cache.get(dd)

    if t is None:
        t = torch.ones((1,), dtype=dtype, device=device)
        _dummy_ones_cache[dd] = t

    return t.expand(shape)


@beartype
def full(
    val: object,
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
    requires_grad: typing.Optional[bool] = None,
) -> torch.Tensor:
    shape, dtype, device, requires_grad = get_sddr(
        like, shape, dtype, device, requires_grad)

    return torch.full(
        shape, val, dtype=dtype, device=device, requires_grad=requires_grad)


@beartype
def dummy_full(
    val: object,
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    shape, dtype, device, _ = get_sddr(like, shape, dtype, device, False)
    return torch.full((1,), val, dtype=dtype, device=device).expand(shape)


@beartype
def eye(
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
    requires_grad: typing.Optional[bool] = None,
) -> torch.Tensor:
    shape, dtype, device, requires_grad = get_sddr(
        like, shape, dtype, device, requires_grad)

    assert 2 <= len(shape)

    N, M = shape[-2:]
    assert N == M

    l = [0] * (N * N)

    for i in range(N):
        l[i * N + i] = 1

    t: torch.Tensor = \
        torch.tensor(l, dtype=dtype).view(N, N).expand(shape).to(device)

    if 0 in t.stride():
        t = t.clone()

    return t.requires_grad_(requires_grad)


@beartype
def dummy_eye(
    *,
    like: typing.Optional[torch.Tensor] = None,
    shape: typing.Optional[tuple[int, ...]] = None,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    shape, dtype, device, _ = get_sddr(like, shape, dtype, device, False)

    return eye(
        shape=shape[-2:], dtype=dtype, device=device, requires_grad=False
    ).expand(shape)


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

    order = check_quaternion_order(order)

    return \
        q[..., order.index("W")], \
        q[..., order.index("X")], \
        q[..., order.index("Y")], \
        q[..., order.index("Z")], \



@beartype
def set_quaternion_wxyz(
    w: torch.Tensor,  # [...]
    x: torch.Tensor,  # [...]
    y: torch.Tensor,  # [...]
    z: torch.Tensor,  # [...]
    order: str,  # wxyz
) -> torch.Tensor:  # [..., 4]
    order = check_quaternion_order(order)

    l = [None, None, None, None]

    l[order.index("W")] = w
    l[order.index("X")] = x
    l[order.index("Y")] = y
    l[order.index("Z")] = z

    return torch.stack(l, -1)


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
def try_get_batch_shape(
    x: typing.Optional[torch.Tensor],
    cdim: int
) -> torch.Size:
    if x is None:
        return torch.Size()

    shape = x.shape

    assert 0 <= cdim
    assert cdim <= len(shape)

    return shape[:len(shape) - cdim]


@beartype
def get_batch_idxes(shape: typing.Iterable[int]) \
        -> typing.Iterable[tuple[int, tuple[int, ...]]]:
    return enumerate(itertools.product(*(range(s) for s in shape)))


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

    data_idx = (slice(None) for _ in range(cdim))

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
def make_diag(
    diag_elem: torch.Tensor,  # [..., D]
    shape: typing.Optional[tuple[int, int]] = None,  # (N, M)
) -> torch.Tensor:  # [..., N, M]
    D = check_shapes(diag_elem, (..., -1))

    if shape is None:
        shape = (D, D)

    assert len(shape) == 2
    assert 0 < shape[0]
    assert 0 < shape[1]

    l: list[torch.Tensor] = \
        [dummy_zeros(like=diag_elem, shape=diag_elem.shape[:-1])] * \
        (shape[0] * shape[1])

    for i in range(min(*shape, D)):
        l[(shape[1] + 1) * i] = diag_elem[..., i]

    t = torch.stack(l, -1)

    return t.view(*t.shape[:-1], *shape)


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
) -> torch.Tensor:  # [...]
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)


@beartype
def vec_sq_norm(
    x: torch.Tensor,  # [...]
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:  # [...]
    return vec_dot(x, x, dim, keepdim)


@beartype
def vec_dot(
    x: torch.Tensor,  # [...]
    y: torch.Tensor,  # [...]
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:  # [...]
    dtype = promote_dtypes(x, y)

    x = x.to(dtype)
    y = y.to(dtype)

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
    dtype = promote_dtypes(x, y)
    device = check_devices(x, y)

    x = x.to(dtype)
    y = y.to(dtype)

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
def c_vec_sq_norm(
    x: torch.Tensor,  # [..., ...(xdim), D]
    y: torch.Tensor,  # [..., ...(ydim), D]
    xdim: int,
    ydim: int,
) -> torch.Tensor:  # [..., ...(xdim), ...(ydim)]
    assert 0 <= xdim
    assert 0 <= ydim

    D = check_shapes(x, (..., -1), y, (..., -1))

    assert xdim + 1 <= x.ndim
    assert ydim + 1 <= y.ndim

    x = x.view(*x.shape[:-xdim-1], *x.shape[-xdim-1:-1], *((1,) * ydim), D)
    y = y.view(*y.shape[:-ydim-1], *((1,) * xdim), *y.shape[-ydim-1:-1], D)
    # x[..., D]
    # y[..., D]

    return vec_sq_norm(x) + vec_sq_norm(y) - 2 * vec_dot(x, y)


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

    return vec_dot(x, y, dim, keepdim) / z.clamp(EPS[z.dtype], None)


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
    axis_angle: torch.Tensor,  # [..., 3]
    order: str,  # permutation of "wxyz"
) -> torch.Tensor:  # [..., 4]
    check_shapes(axis_angle, (..., 3))

    order = check_quaternion_order(order)

    eps = EPS[axis_angle.dtype]

    angle_sq = vec_sq_norm(axis_angle)

    small_angle_mask = angle_sq < eps
    save_angle_sq = angle_sq.clamp(eps, None)

    angle = save_angle_sq.sqrt()

    half_angle = angle / 2

    ks = torch.where(
        small_angle_mask,
        0.5 - angle_sq / 48 + angle_sq.square() / 3840,
        half_angle.sin() / angle,
    )
    # [...]

    w = half_angle.cos()
    x = axis_angle[..., 0] * ks
    y = axis_angle[..., 1] * ks
    z = axis_angle[..., 2] * ks

    return set_quaternion_wxyz(w, x, y, z, order)


@beartype
def quaternion_to_axis_angle(
    quaternion: torch.Tensor,  # [..., 4]
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

    axis = torch.stack([x * p, y * p, z * p], -1)

    return axis, w.acos() * 2


@beartype
def axis_angle_to_rot_mat(
    axis_angle: torch.Tensor,  # [..., 3]
    out_shape: tuple[int, int],  # (3/4, 3/4)
) -> torch.Tensor:  # [..., 3/4, 3/4]
    check_shapes(axis_angle, (..., 3))

    assert 3 <= out_shape[0] <= 4
    assert 3 <= out_shape[1] <= 4

    eps = EPS[axis_angle.dtype]

    buffer: list[torch.Tensor] = [
        [None for _ in range(out_shape[1])] for _ in range(out_shape[0])]

    angle_sq = vec_sq_norm(axis_angle)

    small_angle_mask = angle_sq < eps
    save_angle_sq = angle_sq.clamp(eps, None)

    angle = save_angle_sq.sqrt()
    angle_sq_sq = angle_sq.square()

    angle_cos = angle.cos()

    ks = torch.where(
        small_angle_mask,
        1 - angle_sq / 6 + angle_sq_sq / 120,
        angle.sin() / angle,
    )
    # [...]

    kc = torch.where(
        small_angle_mask,
        0.5 - angle_sq / 24 + angle_sq_sq / 720,
        (1 - angle_cos) / save_angle_sq,
    )
    # [...]

    vx = axis_angle[..., 0]
    vy = axis_angle[..., 1]
    vz = axis_angle[..., 2]

    vxx_nc = vx.square() * kc
    vyy_nc = vy.square() * kc
    vzz_nc = vz.square() * kc

    vxy_nc = vyx_nc = vx * vy * kc
    vyz_nc = vzy_nc = vy * vz * kc
    vzx_nc = vxz_nc = vz * vx * kc

    vxs = vx * ks
    vys = vy * ks
    vzs = vz * ks

    buffer[0][0] = vxx_nc + angle_cos
    buffer[0][1] = vxy_nc - vzs
    buffer[0][2] = vxz_nc + vys

    buffer[1][0] = vyx_nc + vzs
    buffer[1][1] = vyy_nc + angle_cos
    buffer[1][2] = vyz_nc - vxs

    buffer[2][0] = vzx_nc - vys
    buffer[2][1] = vzy_nc + vxs
    buffer[2][2] = vzz_nc + angle_cos

    if 4 in out_shape:
        z = dummy_zeros(like=buffer[0][0])

        if out_shape == (4, 4):
            buffer[3][3] = dummy_ones(like=buffer[0][0])

        if out_shape[0] == 4:
            buffer[3][0] = z
            buffer[3][1] = z
            buffer[3][2] = z

        if out_shape[1] == 4:
            buffer[0][3] = z
            buffer[1][3] = z
            buffer[2][3] = z

    t = torch.stack([item for row in buffer for item in row], -1)

    return t.view(*t.shape[:-1], *out_shape)


@beartype
def rot_mat_to_axis_angle(
    rot_mat: torch.Tensor  # [..., 3, 3]
) -> torch.Tensor:  # axis_angle[..., 3]
    check_shapes(rot_mat, (..., 3, 3))

    tr = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]

    k = ((tr - 1) / 2).acos() * \
        (4 - (tr - 1).square()).clamp(EPS[rot_mat.dtype], None).rsqrt()

    axis_angle = torch.stack([
        (rot_mat[..., 2, 1] - rot_mat[..., 1, 2]) * k,
        (rot_mat[..., 0, 2] - rot_mat[..., 2, 0]) * k,
        (rot_mat[..., 1, 0] - rot_mat[..., 0, 1]) * k,
    ], -1)
    # [..., 3]

    return axis_angle


@beartype
def quaternion_to_rot_mat(
    quaternion: torch.Tensor,  # [..., 4]
    order: str,  # permutation of "wxyz"
    out_shape: typing.Optional[tuple[int, int]] = None,  # (3/4, 3/4)
) -> torch.Tensor:  # [..., 3/4, 3/4]
    check_shapes(quaternion, (..., 4))

    order = check_quaternion_order(order)

    assert 3 <= out_shape[0] <= 4
    assert 3 <= out_shape[1] <= 4

    buffer: list[torch.Tensor] = [
        [None for _ in range(out_shape[1])] for _ in range(out_shape[0])]

    w, x, y, z = get_quaternion_wxyz(quaternion, order)

    k = math.sqrt(2) / vec_norm(quaternion).clamp(EPS[quaternion.dtype], None)

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

    buffer[0][0] = 1 - yy - zz
    buffer[0][1] = xy - wz
    buffer[0][2] = xz + wy

    buffer[1][0] = yx + wz
    buffer[1][1] = 1 - zz - xx
    buffer[1][2] = yz - wx

    buffer[2][0] = zx - wy
    buffer[2][1] = zy + wx
    buffer[2][2] = 1 - xx - yy

    if 4 in out_shape:
        z = dummy_zeros(like=buffer[0][0])

        if out_shape == (4, 4):
            buffer[3][3] = dummy_ones(like=buffer[0][0])

        if out_shape[0] == 4:
            buffer[3][0] = z
            buffer[3][1] = z
            buffer[3][2] = z

        if out_shape[1] == 4:
            buffer[0][3] = z
            buffer[1][3] = z
            buffer[2][3] = z

    t = torch.stack([item for row in buffer for item in row], -1)

    return t.view(*t.shape[:-1], *out_shape)


@beartype
def rot_mat_to_quaternion(
    rot_mat: torch.Tensor,  # [..., 3, 3]
    order: str,  # permutation of "wxyz"
) -> torch.Tensor:  # [..., 4]
    check_shapes(rot_mat, (..., 3, 3))

    order = check_quaternion_order(order)

    wi = order.index("W")
    xi = order.index("X")
    yi = order.index("Y")
    zi = order.index("Z")

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

    eps = EPS[rot_mat.dtype]

    s0 = a0.clamp(eps, None).sqrt() * 2
    s1 = a1.clamp(eps, None).sqrt() * 2
    s2 = a2.clamp(eps, None).sqrt() * 2
    s3 = a3.clamp(eps, None).sqrt() * 2

    r_s0 = 1 / s0
    r_s1 = 1 / s1
    r_s2 = 1 / s2
    r_s3 = 1 / s3

    q_mat_buffer = [None] * 16

    q_mat_buffer[wi * 4 + 0] = s0 / 4
    q_mat_buffer[xi * 4 + 0] = m21_sub_m12 * r_s0
    q_mat_buffer[yi * 4 + 0] = m02_sub_m20 * r_s0
    q_mat_buffer[zi * 4 + 0] = m10_sub_m01 * r_s0

    q_mat_buffer[wi * 4 + 1] = m21_sub_m12 * r_s1
    q_mat_buffer[xi * 4 + 1] = s1 / 4
    q_mat_buffer[yi * 4 + 1] = m10_add_m01 * r_s1
    q_mat_buffer[zi * 4 + 1] = m02_add_m20 * r_s1

    q_mat_buffer[wi * 4 + 2] = m02_sub_m20 * r_s2
    q_mat_buffer[xi * 4 + 2] = m10_add_m01 * r_s2
    q_mat_buffer[yi * 4 + 2] = s2 / 4
    q_mat_buffer[zi * 4 + 2] = m21_add_m12 * r_s2

    q_mat_buffer[wi * 4 + 3] = m10_sub_m01 * r_s3
    q_mat_buffer[xi * 4 + 3] = m02_add_m20 * r_s3
    q_mat_buffer[yi * 4 + 3] = m21_add_m12 * r_s3
    q_mat_buffer[zi * 4 + 3] = s3 / 4

    q_mat = torch.stack(q_mat_buffer, -1)
    # [..., 16]

    q_mat = q_mat.view(*q_mat.shape[:-1], 4, 4)
    # [..., 4, 4]

    a_idxes = a_mat.argmax(-1, True)[..., None]
    # [..., 4] -> [..., 1, 1]

    a_idxes = a_idxes.expand(*a_idxes.shape[:-2], 4, 1)
    # [..., 1, 1] -> [..., 4, 1]

    ret = q_mat.gather(-1, a_idxes)[..., 0]
    # [..., 4]

    """

    q_mat[..., q channel 4, choices 4]

    a_idxes[..., dummy 4, dummy 1]

    q_mat[
        ...,
        q channel 4,
        a_idxes[..., dummy 4, dummy 1],
    ]

    """

    return ret


@beartype
def quaternion_mul(
    q1: torch.Tensor,  # [..., 4]
    q2: torch.Tensor,  # [..., 4]
    *,
    order_1: str,  # wxyz
    order_2: str,  # wxyz
    order_out: str,  # wxyz
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

    return set_quaternion_wxyz(out_w, out_x, out_y, out_z, order_out)


@beartype
def make_homo_rt(
    r: torch.Tensor,  # [..., P, Q]
    t: torch.Tensor,  # [..., P]
    b: bool,
) -> torch.Tensor:  # [..., P / P + 1, Q + 1]
    P, Q = -1, -2

    P, Q = check_shapes(
        r, (..., P, Q),
        t, (..., P),
    )

    dtype = promote_dtypes(r, t)
    device = all_same(r.device, t.device)

    batch_shape = broadcast_shapes(r.shape[:-2], t.shape[:-1])

    r = r.expand(*batch_shape, P, Q)
    t = t.expand(*batch_shape, P)

    buffer = [[None for _ in range(Q + 1)] for _ in range(P + b)]

    for p in range(P):
        for q in range(Q):
            buffer[p][q] = r[..., p, q]

        buffer[p][Q] = t[..., p]

    if b:
        a0 = dummy_zeros(shape=batch_shape, dtype=dtype, device=device)
        a1 = dummy_ones(shape=batch_shape, dtype=dtype, device=device)

        for q in range(Q):
            buffer[P][q] = a0

        buffer[P][Q] = a1

    return torch.stack(
        [item for row in buffer for item in row], -1
    ).view(*batch_shape, P + b, Q + 1)


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
) -> torch.Tensor:
    n = len(args)

    assert 0 < n

    if dtype is None:
        dtype = promote_dtypes(*args)

    acc = args[0].to(device, dtype)

    for i in range(1, n):
        acc = acc @ args[i].to(device, dtype)

    return acc


@beartype
def do_rt(
    r: torch.Tensor,  # [..., P, Q]
    t: torch.Tensor,  # [..., P]
    v: torch.Tensor,  # [..., Q]
) -> torch.Tensor:  # out[..., P]
    P, Q = -1, -2

    P, Q = check_shapes(
        r, (..., P, Q),
        t, (..., P),
        v, (..., Q),
    )

    dtype = promote_dtypes(r, t, v)

    r = r.to(dtype)
    t = t.to(dtype)
    v = v.to(dtype)

    return (r @ v[..., None])[..., 0] + t


@beartype
def merge_rt(
    a_r: torch.Tensor,  # [..., P, Q]
    a_t: torch.Tensor,  # [..., P]
    b_r: torch.Tensor,  # [..., Q, R]
    b_t: torch.Tensor,  # [..., Q]
) -> tuple[
    torch.Tensor,  # out_r[..., P, R]
    torch.Tensor,  # out_t[..., P]
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

    out_r = a_r @ b_r
    # [..., P, R]

    out_t = (a_r @ b_t[..., None])[..., 0] + a_t
    # [..., P]

    return out_r, out_t


@beartype
def get_inv_rt(
    r: torch.Tensor,  # [..., D, D]
    t: torch.Tensor,  # [..., D]
) -> tuple[
    torch.Tensor,  # out_rs[..., D, D]
    torch.Tensor,  # out_ts[..., D]
]:
    D = check_shapes(
        r, (..., -1, -1),
        t, (..., -1),
    )

    inv_r = r.inverse()
    # [..., D, D]

    inv_t = (inv_r @ -t[..., None])[..., 0]
    # [..., D]

    return inv_r, inv_t


@beartype
def do_homo(
    h: torch.Tensor,  # [..., P, Q]
    v: torch.Tensor,  # [..., Q]
) -> torch.Tensor:  # [..., P]
    P, Q = check_shapes(
        h, (..., -1, -2),
        v, (..., -2),
    )

    dtype = promote_dtypes(h, v)

    h = h.to(dtype)
    v = v.to(dtype)

    return homo_normalize((h @ v[..., None])[..., 0])


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

    dtype = promote_dtypes(src, dst)

    src = src.to(dtype)
    dst = dst.to(dtype)

    if normalize:
        src_h = _get_normalize_h(src[:, :-1], math.sqrt(P-1))
        # src_h[P, P]

        dst_h = _get_normalize_h(dst[:, :-1], math.sqrt(Q - 1))
        # dst_h[Q, Q]

        rep_src = (src_h @ src[..., None])[..., 0]
        rep_dst = (dst_h @ dst[..., None])[..., 0]
    else:
        rep_src = src
        rep_dst = dst

    A = torch.empty([N * (Q - 1), Q * P], dtype=dtype)

    A[:, :-P] = 0

    for q in range(Q - 1):
        A[q::Q - 1, P * q:P * q + P] = rep_src

    A[:, -P:] = (rep_src[..., None, :] * -rep_dst[:, :-1, None]) \
        .reshape(N * (Q - 1), P)
    # [N*(Q - 1), P] = (N, 1, P) * (N, Q - 1, 1) = (N, Q - 1, P) = (N*(Q - 1), P)

    Vh: torch.Tensor = torch.linalg.svd(A)[2]

    H = Vh[-1, :].reshape(Q, P)

    if normalize:
        H = dst_h.inverse() @ H @ src_h

    if calc_err:
        err = math.sqrt((do_homo(H, src) - dst).square().sum() / N)
    else:
        err = -1.0

    return H, err


PathLike = str | bytes | pathlib.Path | os.PathLike


@beartype
def to_pathlib_path(path: PathLike) -> pathlib.Path:
    return path if isinstance(path, pathlib.Path) else pathlib.Path(path)


@beartype
def create_file(path: PathLike, mode="w", *args, **kwargs):
    path = to_pathlib_path(path)

    path.parents[0].mkdir(parents=True, exist_ok=True)

    return open(path, mode=mode, *args, **kwargs)


@beartype
def read_file(
    path: PathLike,
    mode: str,
) -> typing.Any:
    with open(path, mode) as f:
        return f.read()


@beartype
def write_file(
    path: PathLike,
    mode: str,
    data: typing.Any,
) -> None:
    with create_file(path, mode) as f:
        f.write(data)


@beartype
def create_dir(dir: PathLike, clear: bool = False) -> pathlib.Path:
    dir = to_pathlib_path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    if clear:
        remove_dir(dir, False)

    return dir


@beartype
def remove_dir(dir: PathLike, rm_dir_self: bool = True) -> None:
    dir = to_pathlib_path(dir)

    if not dir.exists():
        return

    assert dir.is_dir()

    if rm_dir_self:
        shutil.rmtree(dir)
        return

    for name in os.listdir(dir):
        child = dir / name

        if os.path.isfile(child) or os.path.islink(child):
            os.unlink(child)
        else:
            shutil.rmtree(child)


@beartype
def read_json(path: PathLike) -> typing.Any:
    with open(path, "r", encoding="utf-8") as f:
        ret = json.load(f)

    return ret


@beartype
def write_json(path: PathLike, data: typing.Any) -> None:
    with create_file(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


@beartype
class PickleReader:
    def __init__(
        self,
        path: PathLike,
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
        path: PathLike,
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
def read_pickle(path: PathLike) -> typing.Any:
    with PickleReader(path) as reader:
        return reader.read()


@beartype
def write_pickle(
    path: PathLike,
    data: object,
    *,
    mode: str = "wb+",
) -> None:
    print(f"Writing pickle to \"{path=}\".")

    with PickleWriter(path, mode) as writer:
        writer.write(data)


@beartype
def write_tensor_to_file(path: PathLike, x: torch.Tensor) -> None:
    path = to_pathlib_path(path)

    path.parents[0].mkdir(parents=True, exist_ok=True)

    if x.is_floating_point():
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


@beartype
def serialize_tensor(
    x: typing.Optional[torch.Tensor],
    dtype: typing.Optional[object] = None,
) -> typing.Optional[np.ndarray]:
    return None if x is None \
        else np.array(x.numpy(force=True), dtype=dtype, copy=True)


@beartype
def deserialize_tensor(
    x: typing.Optional[np.ndarray],
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> typing.Optional[torch.Tensor]:
    return None if x is None \
        else torch.from_numpy(x).to(device, dtype, copy=True)


@beartype
def show_tensor_info(
    x: typing.Iterable[tuple[str, torch.Tensor]],
) -> None:
    def _f(x):
        return None if x is None else f"{x:+.4e}"

    tab = [(
        "name",
        "dtype",
        "shape",
        "device",
        "requires_grad",
        "grad_min",
        "grad_max",
        "grad_mean",
        "grad_sum",
        "grad_std",
    )]

    for name, t in x:
        dtype = t.dtype
        shape = t.shape
        device = t.device
        requires_grad = t.requires_grad

        grad = t.grad

        if grad is None:
            grad_min = None
            grad_max = None
            grad_mean = None
            grad_sum = None
            grad_std = None
        else:
            grad_min = grad.min().item()
            grad_max = grad.max().item()
            grad_mean = grad.mean().item()
            grad_sum = grad.sum().item()
            grad_std = grad.std().item()

        tab.append((
            name,
            dtype,
            shape,
            device,
            requires_grad,
            _f(grad_min),
            _f(grad_max),
            _f(grad_mean),
            _f(grad_sum),
            _f(grad_std),
        ))

    print(tabulate.tabulate(tab, tablefmt="grid"))


@beartype
def show_point(
    point: torch.Tensor,  # [..., 3]
) -> None:
    import trimesh

    pc = trimesh.points.PointCloud(
        point.detach().to(CPU_DEVICE).reshape(-1, 3))

    pc.show()


@beartype
class LossTable:
    def __init__(self):
        self.table: dict[tuple[torch.Tensor, torch.Tensor]] = dict()

    def add(
        self,
        name: str,
        loss: torch.Tensor,
        weighted_loss: torch.Tensor,
    ) -> None:
        assert name not in self.table

        assert loss.isfinite().all()
        assert weighted_loss.isfinite().all()

        self.table[name] = (loss, weighted_loss)

    def get_weighted_sum_loss(self) -> torch.Tensor:
        return sum(w_loss for loss, w_loss in self.table.values())

    def show(self) -> None:
        def _f(x):
            return None if x is None else f"{x.item():+.4e}"

        print(tabulate.tabulate(zip(
            ("", "loss", "weighted loss"),
            *(
                (loss_name, _f(loss), _f(w_loss))
                for loss_name, (loss, w_loss) in self.table.items()
            )
        ), tablefmt="grid"))
