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
class MaskStrategy(enum.Enum):
    MAX_SCORE = enum.auto()
    MIN_AREA = enum.auto()
    MEAN = enum.auto()


@beartype
def get_lang_sam_module(sam_type: SAMType):
    if sam_type in lang_sam_modules:
        ret = lang_sam_modules[sam_type]
    else:
        ret = lang_sam_modules[sam_type] = lang_sam.LangSAM(
            sam_type=str(sam_type))

    return ret


@beartype
class predict:
    def __init__(
        self,
        sam_type: SAMType,
        img: typing.Iterable[PIL.Image.Image],  # [C, H, W]
        text_prompt: typing.Iterable[str],
        mask_strategy: MaskStrategy,
        batch_size: int,
    ):
        self.sam_type = sam_type
        self.imgs = img
        self.prompts = text_prompt
        self.mask_strategy = mask_strategy
        self.batch_size = batch_size

        try:
            self.length = (min(len(img), len(text_prompt)) +
                           batch_size - 1) // batch_size
        except:
            self.length = None

    @beartype
    def __len__(self) -> typing.Optional[int]:
        return self.length

    @beartype
    def __iter__(self) -> typing.Iterable[torch.Tensor]:  # [C, H, W]
        assert 0 < self.batch_size

        lang_sam_module = get_lang_sam_module(self.sam_type)

        with torch.no_grad():
            for cur_batch in tqdm.tqdm(itertools.batched(
                    zip(self.imgs, self.prompts), self.batch_size)):
                with utils.DisableStdOut():
                    result = lang_sam_module.predict(
                        [k[0] for k in cur_batch],
                        [k[1] for k in cur_batch],
                    )

                for i in range(len(cur_batch)):
                    cur_result = result[i]

                    masks = cur_result["masks"]
                    # [K, H, W]

                    mask_scores = cur_result["mask_scores"].reshape(
                        masks.shape[0])
                    # [K]

                    match self.mask_strategy:
                        case MaskStrategy.MAX_SCORE:
                            mask = masks[mask_scores.argmax()]
                        case MaskStrategy.MIN_AREA:
                            mask = masks[masks.sum(axis=(1, 2)).argmin()]
                        case MaskStrategy.MEAN:
                            mask = masks.mean(axis=0)
                        case _:
                            raise utils.MismatchException()

                    yield torch.from_numpy(mask)
