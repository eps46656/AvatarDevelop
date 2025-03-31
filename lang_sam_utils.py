import enum

import lang_sam
import numpy as np
import PIL
import torch
import tqdm
from beartype import beartype

from . import utils

lang_sam_modules = dict()


@beartype
class SAMType(enum.StrEnum):
    TINY = "sam2.1_hiera_tiny"
    SMALL = "sam2.1_hiera_small"
    BASE_PLUS = "sam2.1_hiera_base_plus"
    LARGE = "sam2.1_hiera_large"


@beartype
class MaskStrategy(enum.Enum):
    MAX_SCORE = 1
    MIN_AREA = 2


@beartype
def get_lang_sam_module(sam_type: SAMType):
    if sam_type in lang_sam_modules:
        ret = lang_sam_modules[sam_type]
    else:
        ret = lang_sam_modules[sam_type] = lang_sam.LangSAM(
            sam_type=str(sam_type))

    return ret


@beartype
def predict(
    sam_type: SAMType,
    imgs: list[PIL.Image.Image],  # [C, H, W]
    prompt: list[str],
    mask_strategy: MaskStrategy,
    batch_size: int,
) -> list[torch.Tensor]:  # [H, W]
    T = len(imgs)
    assert len(prompt) == T

    lang_sam_module = get_lang_sam_module(sam_type)

    ret: list[torch.Tensor] = [None for _ in range(T)]

    with torch.no_grad():
        for frame_beg in tqdm.tqdm(range(0, T, batch_size)):
            frame_end = min(frame_beg + batch_size, T)

            with utils.DisableStdOut():
                result = lang_sam_module.predict(
                    imgs[frame_beg:frame_end],
                    prompt[frame_beg:frame_end],
                )

            for frame_i in range(frame_beg, frame_end):
                cur_result = result[frame_i - frame_beg]

                masks = cur_result["masks"]
                # [K, 1, H, W]

                mask_scores = cur_result["mask_scores"]
                # [K]

                match mask_strategy:
                    case MaskStrategy.MAX_SCORE:
                        target_idx = mask_scores.argmax()
                    case MaskStrategy.MIN_AREA:
                        target_idx = \
                            np.array([masks[k].sum()
                                      for k in range(masks.shape[0])]).argmin()

                ret[frame_i] = torch.from_numpy(masks[target_idx])

    return ret
