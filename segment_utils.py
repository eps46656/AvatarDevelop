import enum
import itertools
import math
import os

import torch
import torchvision
import torchvision.transforms.functional
import tqdm
from beartype import beartype

from . import utils, vision_utils


class ObjectType(enum.Enum):
    # head
    HAT = "HAT"
    HAIR = "HAIR"

    # face
    EYEGLASS = "EYEGLASS"
    EARRING = "EARRING"

    # neck
    NECK_WEAR = "NECK_WEAR"

    # body
    UPPER_GARMENT = "UPPER_GARMENT"
    LOWER_GARMENT = "LOWER_GARMENT"
    ONE_PIECE_GARMENT = "ONE_PIECE_GARMENT"

    # waist
    BELT = "BELT"

    # hand
    WRIST_WEARING = "WRIST_WEARING"
    GLOVES = "GLOVES"

    # foot
    SOCKS = "SOCKS"
    FOOTWEAR = "FOOTWEAR"

    BAG = "BAG"


object_segment_prompts = {
    ObjectType.HAT: "hat",
    ObjectType.HAIR: "hair",

    ObjectType.EYEGLASS: "eye glass",
    ObjectType.EARRING: "earring",

    ObjectType.NECK_WEAR: "neck_wear",

    ObjectType.UPPER_GARMENT: "upper garment",
    ObjectType.LOWER_GARMENT: "lower garment",
    ObjectType.ONE_PIECE_GARMENT: "one-piece garment",

    ObjectType.BELT: "belt",

    ObjectType.WRIST_WEARING: "wrist wearing",
    ObjectType.GLOVES: "gloves",

    ObjectType.SOCKS: "socks",
    ObjectType.FOOTWEAR: "footwear",

    ObjectType.BAG: "bag",
}


PERSON_MASK_FILENAME = "mask_PERSON.avi"
BLURRED_PERSON_MASK_FILENAME = "mask_PERSON_blurred.avi"


@beartype
def object_mask_filename(object_type: ObjectType) -> str:
    return f"mask_{object_type.name}.avi"


@beartype
def blurred_object_mask_filename(object_type: ObjectType) -> str:
    return f"mask_{object_type.name}_blurred.avi"


PREDICT_BATCH_SIZE = 8


@beartype
def make_blur(
    img: torch.Tensor,  # [..., C, H, W]
) -> torch.Tensor:  # [..., C, H, W]
    C, H, W = utils.check_shapes(img, (..., -1, -2, -3))

    diag = math.sqrt(H * H + W * W)

    blur_sigma = diag * 0.01
    kernel_radius = round(blur_sigma * 3)

    blurred_img = torchvision.transforms.functional.gaussian_blur(
        img,
        kernel_size=kernel_radius * 2 + 1,
        sigma=blur_sigma,
    )

    return blurred_img


@beartype
@utils.mem_clear
def _read_person_mask(
    imgs: vision_utils.VideoGenerator,  # [C, H, W]
    out_dir: os.PathLike,
) -> vision_utils.VideoReader:  # [1, H, W]
    from . import lang_sam_utils

    out_dir = utils.to_pathlib_path(out_dir)

    person_mask_path = out_dir / PERSON_MASK_FILENAME

    def read_and_return():
        return vision_utils.VideoReader(
            person_mask_path, vision_utils.ColorType.GRAY)

    if person_mask_path.exists():
        return read_and_return()

    gen = lang_sam_utils.predict(
        sam_type=lang_sam_utils.SAMType.LARGE,
        imgs=(utils.to_pillow_image(img) for img in imgs),
        prompts=itertools.repeat("the main person"),
        mask_strategy=lang_sam_utils.MaskStrategy.MAX_SCORE,
        batch_size=PREDICT_BATCH_SIZE,
    )

    video_writer = vision_utils.VideoWriter(
        person_mask_path,
        height=imgs.height,
        width=imgs.width,
        color_type=vision_utils.ColorType.GRAY,
        fps=imgs.fps,
    )

    with video_writer:
        for person_mask in tqdm.tqdm(gen):
            video_writer.write(vision_utils.denormalize_image(person_mask))

    return read_and_return()


@beartype
def _read_blurred_person_mask(
    masked_imgs: vision_utils.VideoGenerator,  # [C, H, W]
    out_dir: os.PathLike,
) -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    blurred_person_mask_path = out_dir / BLURRED_PERSON_MASK_FILENAME

    def read_and_return():
        return vision_utils.VideoReader(
            blurred_person_mask_path, vision_utils.ColorType.GRAY)

    if blurred_person_mask_path.exists():
        return read_and_return()

    person_masks: vision_utils.VideoReader = \
        _read_person_mask(masked_imgs, out_dir)

    video_writer = vision_utils.VideoWriter(
        blurred_person_mask_path,
        height=person_masks.height,
        width=person_masks.width,
        color_type=vision_utils.ColorType.GRAY,
        fps=person_masks.fps,
    )

    with person_masks, video_writer:
        for cur_person_mask in tqdm.tqdm(person_masks):
            video_writer.write(make_blur(cur_person_mask))

    return read_and_return()


@beartype
@utils.mem_clear
def _read_object_mask(
    masked_imgs: vision_utils.VideoGenerator,  # [C, H, W]
    out_dir: os.PathLike,
    object_type: ObjectType,
) -> vision_utils.VideoReader:  # [1, H, W]
    from . import lang_sam_utils

    out_dir = utils.to_pathlib_path(out_dir)

    object_mask_path = out_dir / object_mask_filename(object_type)

    def read_and_return():
        return vision_utils.VideoReader(
            object_mask_path, vision_utils.ColorType.GRAY)

    if object_mask_path.exists():
        return read_and_return()

    object_segment_prompt = object_segment_prompts[object_type]

    gen = lang_sam_utils.predict(
        sam_type=lang_sam_utils.SAMType.LARGE,
        imgs=(utils.to_pillow_image(img) for img in masked_imgs),
        prompts=itertools.repeat(
            f"the {object_segment_prompt} of the main person"),
        mask_strategy=lang_sam_utils.MaskStrategy.MIN_AREA,
        batch_size=PREDICT_BATCH_SIZE,
    )  # [T][H, W]

    video_writer = vision_utils.VideoWriter(
        object_mask_path,
        height=masked_imgs.height,
        width=masked_imgs.width,
        color_type=vision_utils.ColorType.GRAY,
        fps=masked_imgs.fps,
    )

    with video_writer:
        for object_mask in tqdm.tqdm(gen):
            video_writer.write(vision_utils.denormalize_image(object_mask))

    return read_and_return()


@beartype
def _read_blurred_object_mask(
    masked_imgs: vision_utils.VideoGenerator,  # [C, H, W]
    out_dir: os.PathLike,
    object_type: ObjectType,
) -> vision_utils.VideoReader:  # [1, H, W]
    out_dir = utils.to_pathlib_path(out_dir)

    blurred_object_mask_path = \
        out_dir / blurred_object_mask_filename(object_type)

    def read_and_return():
        return vision_utils.VideoReader(
            blurred_object_mask_path, vision_utils.ColorType.GRAY)

    if blurred_object_mask_path.exists():
        return read_and_return()

    object_masks: vision_utils.VideoReader = \
        _read_object_mask(masked_imgs, out_dir, object_type)

    video_writer = vision_utils.VideoWriter(
        blurred_object_mask_path,
        height=object_masks.height,
        width=object_masks.width,
        color_type=vision_utils.ColorType.GRAY,
        fps=object_masks.fps,
    )

    with object_masks, video_writer:
        for cur_object_mask in tqdm.tqdm(object_masks):
            video_writer.write(make_blur(cur_object_mask))

    return read_and_return()


@beartype
def segment(
    imgs: list[torch.Tensor],  # [T][C, H, W]
    fps: float,
    out_dir: os.PathLike,
):
    out_dir = utils.to_pathlib_path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    assert out_dir.is_dir()

    person_masks: vision_utils.VideoGenerator = _read_person_mask(
        vision_utils.SeqVideoGenerator(imgs, fps), out_dir)

    _read_blurred_person_mask(
        vision_utils.SeqVideoGenerator(imgs, fps), out_dir)

    masked_imgs = [
        torch.where(
            128 <= person_mask,
            img,
            255,
        )
        for img, person_mask in zip(imgs, person_masks)
    ]

    for object_type in tqdm.tqdm(ObjectType):
        _read_blurred_object_mask(
            vision_utils.SeqVideoGenerator(masked_imgs, fps),
            out_dir,
            object_type,
        )
