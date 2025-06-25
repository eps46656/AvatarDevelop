import itertools
import math
import typing

import torch
import tqdm
from beartype import beartype

from . import utils, vision_utils

SRC_FILENAME = "src.avi"

PERSON_MASK_FILENAME = "person_mask.avi"
REFINED_PERSON_MASK_FILENAME = "refined_person_mask.avi"

PERSON_MASKED_FILENAME = "person_masked.avi"


@beartype
def get_obj_mask_filename(obj_name: str) -> str:
    return f"obj_mask_[{obj_name}].avi"


@beartype
def get_refined_obj_mask_filename(obj_name: str) -> str:
    return f"obj_mask_[{obj_name}]_refined.avi"


PREDICT_BATCH_SIZE = 8


@beartype
def make_blur(
    img: torch.Tensor,  # [..., C, H, W]
) -> torch.Tensor:  # [..., C, H, W]
    return vision_utils.gaussian_blur(img, ratio_sigma=0.01)


@beartype
class make_temporal_blur:
    def __init__(
        self,
        img: typing.Iterable[torch.Tensor],  # [C, H, W]
        fps: float,
    ):
        self.img = img
        self.fps = fps

        try:
            self.length = len(img)
        except:
            self.length = None

    def __len__(self) -> typing.Optional[int]:
        return self.length

    def __iter__(self) -> typing.Iterable[torch.Tensor]:
        std = 0.1

        r = math.ceil(std * 3 * self.fps)

        weight = [
            math.exp(-0.5 * (i / self.fps) ** 2 / std ** 2)
            for i in range(-r, r + 1)
        ]

        sum_weight = sum(weight)

        weight = [w / sum_weight for w in weight]

        for cur_img in utils.slide_window_with_padding(self.img, len(weight)):
            yield utils.sum([w * i for w, i in zip(weight, cur_img)])


@beartype
def refine(
    img: typing.Iterable[torch.Tensor],  # [C, H, W]
    fps: float,
) -> typing.Iterable[torch.Tensor]:  # [C, H, W]
    return make_temporal_blur(map(make_blur, img), fps)


@beartype
def read_src(out_dir: utils.PathLike) \
        -> vision_utils.VideoReader:  # [C, H, W]

    out_dir = utils.to_pathlib_path(out_dir)

    return vision_utils.VideoReader(
        out_dir / SRC_FILENAME,
        color_type="RGB",
    )


@beartype
def _read_refined_mask(
    name: str,
    refined_path: utils.PathLike,
    mask: vision_utils.VideoReader,
):
    def read_and_return():
        return vision_utils.VideoReader(
            refined_path, "GRAY")

    if refined_path.exists():
        return read_and_return()

    video_writer = vision_utils.VideoWriter(
        refined_path,
        height=mask.height,
        width=mask.width,
        color_type="GRAY",
        fps=mask.fps,
    )

    with mask, video_writer:
        for i in tqdm.tqdm(
            refine(mask, mask.fps),
            desc=f"processing refined_{name}_mask",
        ):
            frame = utils.rct(i, dtype=torch.uint8)

            # vision_utils.show_image(f"refined_{name}_mask", frame)

            video_writer.write(frame)

    return read_and_return()


@beartype
@utils.mem_clear
def read_person_mask(out_dir: utils.PathLike) \
        -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    person_mask_path = out_dir / PERSON_MASK_FILENAME

    def read_and_return():
        return vision_utils.VideoReader(
            person_mask_path, "GRAY")

    if person_mask_path.exists():
        return read_and_return()

    from . import lang_sam_utils

    src = read_src(out_dir)

    gen = lang_sam_utils.predict(
        sam_type=lang_sam_utils.SAMType.LARGE,
        img=map(vision_utils.to_pillow_image, src, "RGB"),
        text_prompt=itertools.repeat("the main person"),
        mask_strategy=lang_sam_utils.MaskStrategy.MAX_SCORE,
        batch_size=PREDICT_BATCH_SIZE,
    )

    with vision_utils.VideoWriter(
        person_mask_path,
        height=src.height,
        width=src.width,
        color_type="GRAY",
        fps=src.fps,
    ) as video_writer:
        for i in tqdm.tqdm(gen, desc="processing person_mask"):
            frame = utils.rct(i * 255, dtype=torch.uint8)

            # vision_utils.show_image("person_mask", frame)

            video_writer.write(frame)

    return read_and_return()


@beartype
def read_refined_person_mask(out_dir: utils.PathLike) \
        -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    return _read_refined_mask(
        "person",
        out_dir / REFINED_PERSON_MASK_FILENAME,
        read_person_mask(out_dir),
    )


@beartype
def read_person_masked(out_dir: utils.PathLike):
    out_dir = utils.to_pathlib_path(out_dir)

    person_masked_path = out_dir / PERSON_MASKED_FILENAME

    def read_and_return():
        return vision_utils.VideoReader(
            person_masked_path, "RGB")

    if person_masked_path.exists():
        return read_and_return()

    src = read_src(out_dir)

    refined_person_mask = read_refined_person_mask(out_dir)

    video_writer = vision_utils.VideoWriter(
        person_masked_path,
        height=refined_person_mask.height,
        width=refined_person_mask.width,
        color_type="RGB",
        fps=refined_person_mask.fps,
    )

    with refined_person_mask, video_writer:
        for cur_img, cur_refined_person_mask in tqdm.tqdm(
            zip(src, refined_person_mask),
            desc="processing person_masked",
        ):
            m = cur_refined_person_mask.to(torch.float64) / 255

            frame = utils.rct(
                255 * (1 - m) + cur_img * m, dtype=torch.uint8)

            # vision_utils.show_image("person_masked", frame)

            video_writer.write(frame)

    return read_and_return()


@beartype
@utils.mem_clear
def read_obj_mask(
    out_dir: utils.PathLike,
    obj_name: str,
    text_prompt: typing.Optional[str],
) -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    obj_mask_path = out_dir / get_obj_mask_filename(obj_name)

    def read_and_return():
        return vision_utils.VideoReader(obj_mask_path, "GRAY")

    if obj_mask_path.exists():
        return read_and_return()

    from . import lang_sam_utils

    src = read_src(out_dir)

    gen = lang_sam_utils.predict(
        sam_type=lang_sam_utils.SAMType.LARGE,
        img=(vision_utils.to_pillow_image(img, "RGB")
             for img in src),
        text_prompt=itertools.repeat(text_prompt),
        mask_strategy=lang_sam_utils.MaskStrategy.INTERSECTION,
        batch_size=PREDICT_BATCH_SIZE,
    )  # [T][H, W]

    video_writer = vision_utils.VideoWriter(
        obj_mask_path,
        height=src.height,
        width=src.width,
        color_type="GRAY",
        fps=src.fps,
    )

    with video_writer:
        for obj_mask in tqdm.tqdm(gen, desc=f"processing obj_mask_{obj_name}"):
            frame = utils.rct(obj_mask * 255, dtype=torch.uint8)

            # vision_utils.show_image(f"{obj_name}_mask", frame)

            video_writer.write(frame)

    return read_and_return()


@beartype
def read_refined_obj_mask(
    out_dir: utils.PathLike,
    obj_name: str,
    text_prompt: typing.Optional[str],
) -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    return _read_refined_mask(
        obj_name,
        out_dir / get_refined_obj_mask_filename(obj_name),
        read_obj_mask(out_dir, obj_name, text_prompt),
    )


@beartype
def read_skin_mask(
    out_dir: utils.PathLike,
    obj_text_prompt: typing.Optional[typing.Mapping[str, str]],
) -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    skin_mask_path = out_dir / get_obj_mask_filename("skin")

    def read_and_return():
        return vision_utils.VideoReader(
            skin_mask_path, "GRAY")

    if skin_mask_path.exists():
        return read_and_return()

    assert obj_text_prompt is not None

    person_mask = read_person_mask(out_dir)

    obj_masks = {
        obj_name: read_obj_mask(out_dir, obj_name, text_prompt)
        for obj_name, text_prompt in obj_text_prompt.items()
    }

    video_writer = vision_utils.VideoWriter(
        skin_mask_path,
        height=person_mask.height,
        width=person_mask.width,
        color_type="GRAY",
        fps=person_mask.fps,
    )

    while True:
        person_mask_frame = person_mask.read()
        obj_mask_frames = [o.read() for o in obj_masks.values()]

        if person_mask_frame is None:
            break

        acc_mask = utils.zeros(like=person_mask_frame)

        for frame in obj_mask_frames:
            assert frame is not None
            acc_mask = frame if acc_mask is None else \
                torch.maximum(acc_mask, frame)

        acc_mask = (torch.minimum(
            person_mask_frame, 255 - acc_mask) / 255).pow(0.5)

        frame = utils.rct(acc_mask * 255, dtype=torch.uint8)

        # vision_utils.show_image(f"skin_mask", frame)

        video_writer.write(frame)

    return read_and_return()


@beartype
def read_refined_skin_mask(
    out_dir: utils.PathLike,
    obj_text_prompt: typing.Optional[typing.Mapping[str, str]],
) -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    return _read_refined_mask(
        "skin",
        out_dir / get_refined_obj_mask_filename("skin"),
        read_skin_mask(out_dir, obj_text_prompt),
    )


@beartype
def segment(
    src: vision_utils.VideoGenerator,  # [T][C, H, W]
    out_dir: utils.PathLike,
    obj_text_prompt: typing.Mapping[str, str],
    en_obj_mask: bool = True,
    en_refined_obj_mask: bool = True,
    en_skin_mask: bool = True,
    en_refined_skin_mask: bool = True,
) -> None:
    out_dir = utils.to_pathlib_path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    assert out_dir.is_dir()

    H, W = src.height, src.width
    fps = src.fps

    src_video_path = out_dir / SRC_FILENAME

    if not src_video_path.exists():
        src_video_writer = vision_utils.VideoWriter(
            out_dir / SRC_FILENAME,
            height=H,
            width=W,
            color_type="RGB",
            fps=fps,
        )

        with src_video_writer:
            for i in tqdm.tqdm(src):
                src_video_writer.write(utils.rct(i, dtype=torch.uint8))

    read_refined_person_mask(out_dir)

    if en_obj_mask:
        for obj_name, text_prompt in obj_text_prompt.items():
            read_obj_mask(out_dir, obj_name, text_prompt)

    if en_refined_obj_mask:
        for obj_name, text_prompt in obj_text_prompt.items():
            read_refined_obj_mask(out_dir, obj_name, text_prompt)

    if "skin" in obj_text_prompt:
        return

    if en_skin_mask:
        read_skin_mask(out_dir, obj_text_prompt)

    if en_refined_skin_mask:
        read_refined_skin_mask(out_dir, obj_text_prompt)
