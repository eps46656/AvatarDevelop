import math
import subprocess
import tempfile
import typing

import torch
from beartype import beartype

from . import config, utils, vision_utils


@beartype
class predict:
    def __init__(
        self,
        img: typing.Iterable[torch.Tensor],  # [C, H, W]
    ):
        self.img = img

        try:
            self.length = len(img)
        except:
            self.length = None

    @beartype
    def __len__(self) -> typing.Optional[int]:
        return self.length

    @beartype
    def __iter__(self) -> typing.Iterable[torch.Tensor]:  # [C, H, W]
        tmp_dir_obj = tempfile.TemporaryDirectory()

        dir = utils.to_pathlib_path(tmp_dir_obj.name)

        img_dir = dir / "image"
        result_dir = dir / "result"

        img_list_path = dir / "img_list.txt"

        img_list = utils.create_file(img_list_path, "w")

        imgs_cnt = 0

        for cur_img in self.img:
            cur_img_path = img_dir / f"{imgs_cnt:06d}.png"

            img_list.write(f"{cur_img_path.resolve()}\n")

            vision_utils.write_image(cur_img_path, cur_img)

            imgs_cnt += 1

        img_list.close()

        subprocess.run([
            "python",
            "inference_acc.py",

            "--loadmodel",
            "../pretrained/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth",

            "--img_list", str(img_list_path.resolve()),

            "--output_dir", str(result_dir.resolve()),
        ],
            cwd=str((config.DD_HUMAN_PARSING_DIR / "inference").resolve()),
            check=True,
        )

        real_result_dir = result_dir / "train_parsing/image"

        for t in range(imgs_cnt):
            yield vision_utils.read_image(
                real_result_dir / f"{t:06d}_label.png", "RGB").image
