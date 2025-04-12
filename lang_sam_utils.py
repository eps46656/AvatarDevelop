import enum
import itertools
import typing

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
class MaskStrategy(enum.StrEnum):
    MAX_SCORE = "max score"
    MIN_AREA = "min area"


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
    imgs: typing.Iterable[PIL.Image.Image],  # [C, H, W]
    prompts: typing.Iterable[str],
    mask_strategy: MaskStrategy,
    batch_size: int,
) -> typing.Iterable[torch.Tensor]:  # [C, H, W]
    assert 0 < batch_size

    lang_sam_module = get_lang_sam_module(sam_type)

    with torch.no_grad():
        for cur_batch in tqdm.tqdm(
                itertools.batched(zip(imgs, prompts), batch_size)):
            with utils.DisableStdOut():
                result = lang_sam_module.predict(
                    [k[0] for k in cur_batch],
                    [k[1] for k in cur_batch],
                )

            for i in range(len(cur_batch)):
                cur_result = result[i]

                masks = cur_result["masks"]
                # [K, H, W]

                mask_scores = cur_result["mask_scores"].reshape(masks.shape[0])
                # [K]

                match mask_strategy:
                    case MaskStrategy.MAX_SCORE:
                        target_idx = mask_scores.argmax()
                    case MaskStrategy.MIN_AREA:
                        target_idx = masks.sum(axis=(1, 2)).argmin()

                yield torch.from_numpy(masks[target_idx])
