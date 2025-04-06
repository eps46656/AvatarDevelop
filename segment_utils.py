import enum
import os
import pathlib

import PIL
import torch
import tqdm
from beartype import beartype

from . import lang_sam_utils, utils


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


PERSON_MASK_FILENAME = "mask<PERSON>.mp4"

BATCH_SIZE = 8


@beartype
def _read_person_mask(
    imgs: torch.Tensor,  # [T, C, H, W]
    out_dir: pathlib.Path,
    out_fps: int,
) -> torch.Tensor:  # [T, 1, H, W]
    T, C, H, W = utils.check_shapes(imgs, (-1, -2, -3, -4))

    person_mask_path = out_dir / PERSON_MASK_FILENAME

    if person_mask_path.exists():
        return utils.read_video(person_mask_path)[0].mean(1, keepdim=True)

    person_masks = lang_sam_utils.predict(
        sam_type=lang_sam_utils.SAMType.LARGE,
        imgs=utils.to_pillow_image([img for img in imgs]),
        prompt=["the main person"] * T,
        mask_strategy=lang_sam_utils.MaskStrategy.MAX_SCORE,
        batch_size=BATCH_SIZE,
    )  # [T][H, W]

    person_masks = torch.stack(person_masks).unsqueeze(1)
    # [T, 1, H, W]

    utils.write_video(
        out_dir / PERSON_MASK_FILENAME,
        person_masks.expand(T, 3, H, W),
        out_fps,
    )

    return person_masks


@beartype
def _read_object_mask(
    masked_imgs: list[PIL.Image.Image],  # [T][C, H, W]
    out_dir: pathlib.Path,
    out_fps: int,
    object_type: ObjectType
) -> torch.Tensor:  # [T, 1, H, W]

    T = len(masked_imgs)

    object_mask_path = out_dir / f"mask<{object_type.name}>.mp4"

    if object_mask_path.exists():
        return utils.read_video(object_mask_path)[0].mean(1, keepdim=True)

    object_segment_prompt = object_segment_prompts[object_type]

    object_masks = lang_sam_utils.predict(
        sam_type=lang_sam_utils.SAMType.LARGE,
        imgs=masked_imgs,
        prompt=[f"the {object_segment_prompt} of the main person"] * T,
        mask_strategy=lang_sam_utils.MaskStrategy.MIN_AREA,
        batch_size=BATCH_SIZE,
    )  # [T][H, W]

    object_masks = torch.stack(object_masks).unsqueeze(1)
    # [T, 1, H, W]

    H, W = object_masks.shape[-2:]

    utils.write_video(
        out_dir / f"mask<{object_type.name}>.mp4",
        object_masks.expand(T, 3, H, W),
        out_fps,
    )

    return object_masks


@beartype
def segment(
    imgs: torch.Tensor,  # [T, C, H, W]
    out_dir: os.PathLike,
    out_fps: int,
):
    T, C, H, W = utils.check_shapes(imgs, (-1, -2, -3, -4))

    out_dir = pathlib.Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    assert out_dir.is_dir()

    person_masks = _read_person_mask(imgs, out_dir, out_fps)

    bg = torch.ones((C, H, W), dtype=utils.FLOAT, device=imgs.device)

    masked_imgs = utils.to_pillow_image([
        bg + (imgs[frame_i] - bg) * person_masks[frame_i]
        for frame_i in range(T)
    ])

    for object_type in tqdm.tqdm(ObjectType):
        _read_object_mask(masked_imgs, out_dir, out_fps, object_type)
