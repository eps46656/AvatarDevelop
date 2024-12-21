import os
import pathlib
import time
import typing

import cv2 as cv
import numpy as np
from typeguard import typechecked


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
def ReadImage(path: object):
    return cv.cvtColor(cv.imdecode(np.fromfile(
        path, dtype=np.uint8), -1), cv.COLOR_BGR2RGB)


@typechecked
def WriteImage(path: object, img: np.array):
    path = pathlib.Path(path)

    os.makedirs(path.parents[0], exist_ok=True)

    cv.imencode(os.path.splitext(path)[1], cv.cvtColor(
        img, cv.COLOR_RGB2BGR))[1].tofile(path)


@typechecked
class Timer:
    def __init__(self):
        self.beg: typing.Optional[float] = None
        self.end: typing.Optional[float] = None

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
        return self

    def __exit__(self, type, value, traceback):
        self.Stop()
        print(f"duration: {self.duration()}")


@typechecked
def Union(*iters: typing.Iterable):
    s = set()

    for iter in iters:
        for o in iter:
            if SetAdd(s, o):
                yield o
