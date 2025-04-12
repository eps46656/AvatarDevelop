from __future__ import annotations

import os
import typing
import enum

import cv2 as cv
import einops
import numpy as np
import torch
import torchvision
from beartype import beartype

from . import utils


class ColorType(enum.StrEnum):
    GRAY = "gray"
    RGB = "rgb"


channels_cnt_table = {
    ColorType.GRAY: 1,
    ColorType.RGB: 3,
}


@beartype
def _frame_to_image(
    frame: np.ndarray,  # [... H, W, C]
    color_type: ColorType,
) -> torch.Tensor:
    assert 3 <= frame.ndim

    match (frame.shape[-1], color_type):
        case (1, ColorType.GRAY):
            arr = frame
        case (1, ColorType.RGB):
            arr = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        case (3, ColorType.GRAY):
            arr = np.expand_dims(cv.cvtColor(
                frame, cv.COLOR_BGR2GRAY), axis=-1)
        case (3, ColorType.RGB):
            arr = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        case _:
            raise utils.MismatchException()

    return torch.from_numpy(einops.rearrange(
        arr, "... h w c -> ... c h w"))


class VideoGenerator:
    @property
    def height(self) -> int:
        raise NotImplementedError()

    @property
    def width(self) -> int:
        raise NotImplementedError()

    @property
    def fps(self) -> float:
        raise NotImplementedError()

    def __iter__(self) -> VideoGenerator:
        return self

    def __next__(self) -> torch.Tensor:
        img = self.read()

        if img is None:
            raise StopIteration()

        return img

    def read(self):
        raise NotImplementedError()


class SeqVideoGenerator(VideoGenerator):
    def __init__(
        self,
        imgs: list[torch.Tensor],  # [T][C, H, W]
        fps: float,
    ):
        self.frames_cnt = len(imgs)

        if self.frames_cnt == 0:
            self.__height = -1
            self.__width = -1
        else:
            self.__height, self.__width = \
                utils.all_same(*(img.shape[1:] for img in imgs))

        self.__imgs = imgs
        self.__fps = fps

        self.__img_idx = 0

    @property
    def height(self) -> int:
        return self.__height

    @property
    def width(self) -> int:
        return self.__width

    @property
    def fps(self) -> float:
        return self.__fps

    @property
    def img_idx(self) -> int:
        return self.__img_idx

    def __len__(self) -> int:
        return self.frames_cnt

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def set_img_idx(self, img_idx: int) -> None:
        assert -self.frames_cnt <= img_idx
        assert img_idx <= self.frames_cnt

        self.__img_idx = \
            (img_idx + self.frames_cnt + 1) % (self.frames_cnt + 1)

    def read(self) -> torch.Tensor:
        if self.frames_cnt <= self.__img_idx:
            return None

        ret = self.__imgs[self.__img_idx]

        self.__img_idx += 1

        return ret


class VideoReader(VideoGenerator):
    def __init__(
        self,
        path: os.PathLike,
        color_type: ColorType = ColorType.RGB,
    ):
        self.reader = cv.VideoCapture(path)
        self.color_type = color_type

    def __del__(self):
        return self.close()

    @property
    def height(self) -> int:
        return int(self.reader.get(cv.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self) -> int:
        return int(self.reader.get(cv.CAP_PROP_FRAME_WIDTH))

    @property
    def fps(self) -> float:
        return float(self.reader.get(cv.CAP_PROP_FPS))

    @property
    def is_opened(self) -> bool:
        return self.reader.isOpened()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def grab(self) -> bool:
        return self.reader.grab()

    def retrieve(self) -> tuple[bool, torch.Tensor]:
        b, frame = self.reader.retrieve()
        return _frame_to_image(frame, self.color_type) if b else None

    def read(self) -> torch.Tensor:
        b, frame = self.reader.read()
        return _frame_to_image(frame, self.color_type) if b else None

    def close(self) -> None:
        self.reader.release()


class VideoWriter:
    def __init__(
        self,
        path: os.PathLike,
        height: int,
        width: int,
        color_type: ColorType,
        fps: float,
        fourcc: str = "XVID",
    ):
        path = utils.to_pathlib_path(path)
        os.makedirs(path.parents[0], exist_ok=True)

        self.height = height
        self.width = width
        self.color_type = color_type
        self.fps = fps

        fourcc = cv.VideoWriter_fourcc(*fourcc)
        self.writer = cv.VideoWriter(
            path, fourcc, fps, (self.width, self.height))

    def __del__(self) -> None:
        self.close()

    @property
    def is_opened(self) -> bool:
        return self.writer.isOpened()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def write(
        self,
        img: torch.Tensor,  # [C, H, W]
    ) -> None:
        assert img.dtype == torch.uint8

        if img.ndim == 2:
            img = img.unsqueeze(0)

        match (img.shape[0], self.color_type):
            case (1, ColorType.GRAY):
                img = img
            case (1, ColorType.RGB):
                img = img.expand(3, self.height, self.width)
            case (3, ColorType.GRAY):
                img = torch.div(
                    img.to(torch.int).sum(0, True),
                    3,
                    rounding_mode="trunc",
                ).to(torch.uint8)
            case (3, ColorType.RGB):
                img = img
            case _:
                raise utils.MismatchException()

        self.writer.write(cv.cvtColor(einops.rearrange(
            img.cpu().numpy(), "c h w -> h w c"), cv.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


@beartype
def read_video(
    path: os.PathLike,
    color_type: ColorType,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> tuple[
    typing.Optional[torch.Tensor],  # imgs
    float,  # fps
]:
    reader = VideoReader(path, color_type)

    first_frame = reader.read()

    fps = reader.fps

    if first_frame is None:
        return None, fps

    C, H, W = first_frame.shape

    frames_cnt = 1

    while reader.grab():
        frames_cnt += 1

    reader = VideoReader(path, color_type)

    ret = torch.empty(
        (frames_cnt, channels_cnt_table[color_type], H, W),
        dtype=first_frame.dtype if dtype is None else dtype,
        device=device,
    )

    for frame_i, frame in enumerate(reader):
        ret[frame_i] = frame

    return ret, fps


@beartype
def read_video_mask(
    path: os.PathLike,
    *,
    dtype: typing.Optional[torch.dtype] = torch.float16,
    device: typing.Optional[torch.device] = None,
) -> tuple[
    typing.Optional[torch.Tensor],  # imgs
    float,  # fps
]:
    reader = VideoReader(path, ColorType.GRAY)

    first_frame = reader.read()

    fps = reader.fps

    if first_frame is None:
        print(f"Not found {path}")
        return None, fps

    C, H, W = first_frame.shape

    frames_cnt = 1

    while reader.grab():
        frames_cnt += 1

    reader = VideoReader(path, ColorType.GRAY)

    ret = torch.empty(
        (frames_cnt, 1, H, W),
        dtype=dtype,
        device=device,
    )

    for frame_i, frame in enumerate(reader):
        ret[frame_i] = frame.to(dtype=dtype) / 255

    return ret, fps
