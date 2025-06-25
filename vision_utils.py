from __future__ import annotations

import enum
import math
import typing

import cv2 as cv
import einops
import numpy as np
import PIL
import PIL.Image
import torch
import torchvision
import torchvision.transforms.functional
from beartype import beartype

from . import utils


class ColorType(enum.StrEnum):
    GRAY = "GRAY"
    RGB = "RGB"
    RGBA = "RGBA"


ColorTypeLike = str | ColorType

channels_cnt_table = {
    ColorType.GRAY: 1,
    ColorType.RGB: 3,
    ColorType.RGBA: 4,
}


DEFUALT_FOURCC = "FFV1"


class ImageWithColorType(typing.NamedTuple):
    image: torch.Tensor  # [..., C, H, W]
    color_type: ColorType


@beartype
def to_color_type(
    color_type: ColorTypeLike,
) -> ColorType:
    if isinstance(color_type, ColorType):
        return color_type

    match color_type.upper():
        case "GRAY" | "G" | "L": return ColorType.GRAY
        case "RGB": return ColorType.RGB
        case "RGBA": return ColorType.RGBA
        case _: raise utils.MismatchException()


@beartype
def read_image(
    path: utils.PathLike,
    color_type: typing.Optional[ColorTypeLike] = None,
) -> ImageWithColorType:
    return from_pillow_image(PIL.Image.open(path), color_type)


@beartype
def write_image(
    path: utils.PathLike,
    img: torch.Tensor,  # [C, H, W]
    color_type: typing.Optional[ColorTypeLike] = None,
) -> None:
    path = utils.to_pathlib_path(path)

    path.parents[0].mkdir(parents=True, exist_ok=True)

    print(f"Writing image to \"{path}\".")

    to_pillow_image(img, color_type).save(path)


@beartype
def derive_color_type(
    img: torch.Tensor,  # [C, H, W]
) -> ColorType:
    C, H, W = utils.check_shapes(img, (-1, -2, -3))

    match C:
        case 1: return ColorType.GRAY
        case 3: return ColorType.RGB
        case 4: return ColorType.RGBA
        case _: raise utils.MismatchException()


@beartype
def change_color_type(
    img: torch.Tensor,  # [..., C, H, W]
    src_color_type: typing.Optional[ColorTypeLike],
    dst_color_type: ColorTypeLike,
) -> torch.Tensor:
    C, H, W = utils.check_shapes(img, (..., -1, -2, -3))

    shape = img.shape[:-3]

    assert not img.is_floating_point()

    if src_color_type is None:
        src_color_type = derive_color_type(img)

    src_color_type = to_color_type(src_color_type)
    dst_color_type = to_color_type(dst_color_type)

    assert C == channels_cnt_table[src_color_type]

    if src_color_type == dst_color_type:
        return img

    match (src_color_type, dst_color_type):
        case (ColorType.GRAY, ColorType.RGB):
            return img.expand(*shape, 3, H, W)

        case (ColorType.GRAY, ColorType.RGBA):
            return torch.cat([img, img, img, utils.full(255, like=img)], -3)

        case (ColorType.RGB, ColorType.GRAY):
            return (img.sum(
                -3, True, utils.promote_dtypes(img.dtype, torch.uint16)
            ) + 1) // 3

        case (ColorType.RGB, ColorType.RGBA):
            return torch.cat([img, utils.full(255, like=img)], -3)

        case (ColorType.RGBA, ColorType.RGB):
            return img[..., :3, :, :]

        case (ColorType.RGBA, ColorType.GRAY):
            return (img[..., :3, :, :].sum(
                -3, True, utils.promote_dtypes(img.dtype, torch.uint16)
            ) + 1) // 3

        case _:
            raise utils.MismatchException()


@beartype
def make_gaussian_kernel(
    sigma: float,
    kernel_radius: int,
    make_mean: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ax = torch.arange(
        -kernel_radius, kernel_radius + 1,
        dtype=dtype,
        device=device,
    )

    xx, yy = torch.meshgrid(ax, ax, indexing='ij')

    kernel = (-(xx**2 + yy**2) / (2 * sigma**2)).exp()

    if make_mean:
        kernel = kernel / kernel.sum()

    return kernel


@beartype
def get_sigma(
    img_h: int,
    img_w: int,
    ratio_sigma: float,
) -> float:
    assert 0 < img_h
    assert 0 < img_w
    assert 0 < ratio_sigma

    return math.sqrt(img_h ** 2 + img_w ** 2) * ratio_sigma


@beartype
def gaussian_blur(
    img: torch.Tensor,  # [..., H, W]
    *,
    sigma: typing.Optional[float] = None,
    ratio_sigma: typing.Optional[float] = None,
) -> torch.Tensor:  # [..., H, W]
    H, W = utils.check_shapes(img, (..., -1, -2))

    if sigma is None:
        assert ratio_sigma is not None
        assert 0 < ratio_sigma

        sigma = get_sigma(H, W, ratio_sigma)

    assert 0 < sigma

    kernel_radius = max(1, math.ceil(sigma * 3))

    blurred_img = torchvision.transforms.functional.gaussian_blur(
        img.reshape(-1, H, W),
        kernel_size=kernel_radius * 2 + 1,
        sigma=sigma,
    )

    return blurred_img.view(img.shape)


@beartype
def dilate(
    img: torch.Tensor,  # [..., H, W]
    *,
    sigma: typing.Optional[float] = None,
    ratio_sigma: typing.Optional[float] = None,
):
    H, W = utils.check_shapes(img, (..., -1, -2))

    if sigma is None:
        assert ratio_sigma is not None
        assert 0 < ratio_sigma

        sigma = get_sigma(H, W, ratio_sigma)

    assert 0 < sigma

    kernel_radius = max(1, math.ceil(sigma * 3))

    kernel = make_gaussian_kernel(
        sigma=sigma,
        kernel_radius=kernel_radius,
        make_mean=False,
        dtype=img.dtype,
        device=img.device,
    )

    ret_img = torch.nn.functional.conv2d(
        input=img.reshape(-1, 1, H, W),
        weight=kernel[None, None, :, :],
        padding=kernel_radius,
    ).view(img.shape)
    # [B, 1, H, W]

    return ret_img


@beartype
def show_image(
    title: str,
    img: torch.Tensor,  # [C, H, W]
    color_type: typing.Optional[ColorTypeLike] = None,
    *,
    pause: bool = False,
) -> None:
    if img.ndim == 2:
        img = img[None, :, :]

    if color_type is None:
        color_type = derive_color_type(img)

    cv.imshow(title, to_opencv_image(img, color_type))
    cv.waitKey(0 if pause else 1)


@beartype
def to_opencv_image(
    img: torch.Tensor,  # [..., C, H, W]
    color_type: typing.Optional[ColorTypeLike] = None,
) -> np.ndarray:  # [..., H, W, C]
    C, H, W = utils.check_shapes(img, (..., -1, -2, -3))

    if color_type is None:
        color_type = derive_color_type(img)

    assert C == channels_cnt_table[color_type]

    img = utils.rct(img.detach(), dtype=torch.uint8, device=utils.CPU_DEVICE)

    if color_type == ColorType.GRAY:
        return img[..., 0, :, :].numpy(force=True)

    if color_type == ColorType.RGB:
        r = img[..., 0, :, :]
        g = img[..., 1, :, :]
        b = img[..., 2, :, :]

        return torch.stack([b, g, r], -1).numpy(force=True)

    if color_type == ColorType.RGBA:
        r = img[..., 0, :, :]
        g = img[..., 1, :, :]
        b = img[..., 2, :, :]
        a = img[..., 3, :, :]

        return torch.stack([b, g, r, a], -1).numpy(force=True)

    raise utils.MismatchException()


@beartype
def from_opencv_image(
    img: np.ndarray,  # [..., H, W, (C)]
    color_type: ColorTypeLike,
) -> torch.Tensor:  # [..., C, H, W]
    assert img.dtype == np.uint8

    img = torch.from_numpy(img)

    color_type = to_color_type(color_type)

    if color_type == ColorType.GRAY:
        utils.check_shapes(img, (..., -1, -2))
        return img[..., None, :, :]

    H, W, C = utils.check_shapes(img, (..., -1, -2, -3))

    assert C == channels_cnt_table[color_type]

    if color_type == ColorType.RGB:
        b = img[..., :, :, 0]
        g = img[..., :, :, 1]
        r = img[..., :, :, 2]

        return torch.stack([r, g, b], -3)

    if color_type == ColorType.RGBA:
        b = img[..., :, :, 0]
        g = img[..., :, :, 1]
        r = img[..., :, :, 2]
        a = img[..., :, :, 3]

        return torch.stack([r, g, b, a], -3)

    raise utils.MismatchException()


@beartype
def to_pillow_image(
    img: torch.Tensor,  # [C, H, W]
    color_type: typing.Optional[ColorTypeLike] = None,
) -> PIL.Image.Image:
    C, H, W = utils.check_shapes(img, (-1, -2, -3))

    color_type = to_color_type(
        derive_color_type(img) if color_type is None else color_type)

    if C == 1:
        np_img = img[0].numpy(force=True)
    else:
        np_img = einops.rearrange(img, "c h w -> h w c").numpy(force=True)

    return PIL.Image.fromarray(
        np_img,
        {
            ColorType.GRAY: "L",
            ColorType.RGB: "RGB",
            ColorType.RGBA: "RGBA",
        }[color_type]
    )


@beartype
def from_pillow_image(
    img: PIL.Image.Image,
    color_type: typing.Optional[ColorTypeLike] = None,
) -> ImageWithColorType:
    x_color_type = {
        "L": ColorType.GRAY,
        "RGB": ColorType.RGB,
        "RGBA": ColorType.RGBA,
    }[img.mode]

    x = torch.from_numpy(np.array(img))
    # [H, W] or [H, W, C]

    if x.ndim == 2:
        x = x[:, :, None]

    x = einops.rearrange(x, "h w c -> c h w")

    if color_type is not None:
        color_type = to_color_type(color_type)
        x = change_color_type(x, x_color_type, color_type)
        x_color_type = color_type

    return ImageWithColorType(
        image=x,
        color_type=x_color_type
    )


@beartype
def _frame_to_image(
    frame: np.ndarray,  # [... H, W, C]
    color_type: ColorTypeLike,
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


@beartype
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


@beartype
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

    def read(self) -> typing.Optional[torch.Tensor]:
        if self.frames_cnt <= self.__img_idx:
            return None

        ret = self.__imgs[self.__img_idx]

        self.__img_idx += 1

        return ret


@beartype
class VideoReader(VideoGenerator):
    def __init__(
        self,
        path: utils.PathLike,
        color_type: ColorTypeLike = ColorType.RGB,
    ):
        path = utils.to_pathlib_path(path)

        assert path.is_file()

        self.reader = cv.VideoCapture(path)
        self.color_type = to_color_type(color_type)

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

    def read(self) -> typing.Optional[torch.Tensor]:
        b, frame = self.reader.read()
        return _frame_to_image(frame, self.color_type) if b else None

    def read_many(
        self,
        max_num: int = -1,
    ) -> typing.Optional[torch.Tensor]:  # [T, C, H, W]
        l: list[torch.Tensor] = list()
        # [][C, H, W]

        while max_num != len(l):
            image = self.read()

            if image is None:
                break

            l.append(image)

        return None if len(l) == 0 else torch.stack(l, 0)

    def close(self) -> None:
        self.reader.release()


@beartype
class VideoWriter:
    def __init__(
        self,
        path: utils.PathLike,
        height: int,
        width: int,
        color_type: ColorTypeLike,
        fps: float,
        fourcc: str = DEFUALT_FOURCC,
    ):
        path = utils.to_pathlib_path(path)

        path.parents[0].mkdir(parents=True, exist_ok=True)

        self.path = path
        self.height = height
        self.width = width
        self.color_type = to_color_type(color_type)
        self.fps = fps

        fourcc = cv.VideoWriter_fourcc(*fourcc)

        self.writer = cv.VideoWriter(
            filename=path,
            fourcc=fourcc,
            fps=fps,
            frameSize=(self.width, self.height),
            isColor=self.color_type != ColorType.GRAY,
        )

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
        color_type: typing.Optional[ColorTypeLike] = None,
    ) -> None:
        self.writer.write(to_opencv_image(change_color_type(
            img, color_type, self.color_type), self.color_type))

    def write_all(self, img: typing.Iterable[torch.Tensor]) -> None:
        for i in img:
            self.write(i)

    def close(self) -> None:
        if not self.is_opened:
            return

        print(f"Write video to \"{self.path}\".")
        self.writer.release()


@beartype
def read_video(
    path: utils.PathLike,
    color_type: ColorTypeLike,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: torch.device,
) -> tuple[
    typing.Optional[torch.Tensor],  # img
    float,  # fps
]:
    reader = VideoReader(path, color_type)

    return torch.stack([
        frame.to(device, dtype)
        for frame in reader
    ]), reader.fps


@beartype
def read_video_mask(
    path: utils.PathLike,
    *,
    dtype: typing.Optional[torch.dtype] = torch.float32,
    device: torch.device,
) -> tuple[
    typing.Optional[torch.Tensor],  # mask[T, 1, H, W]
    float,  # fps
]:
    reader = VideoReader(path, ColorType.GRAY)

    return torch.stack([
        frame.to(device, dtype) / 255
        for frame in reader
    ]), reader.fps


@beartype
@utils.mem_clear
def write_video(
    path: utils.PathLike,
    video: torch.Tensor,  # [T, C, H, W]
    fps: float,
    fourcc: str = DEFUALT_FOURCC,
) -> None:
    T, C, H, W = -1, -2, -3, -4

    T, C, H, W = utils.check_shapes(video, (T, C, H, W))

    if T == 0:
        return

    with VideoWriter(
        path,
        height=H,
        width=W,
        color_type=ColorType.RGB,
        fps=fps,
        fourcc=fourcc,
    ) as writer:
        for t in range(T):
            writer.write(video[t])
