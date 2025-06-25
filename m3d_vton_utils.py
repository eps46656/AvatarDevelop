

import os
import subprocess

import torch
import torchvision
from beartype import beartype

from . import config, utils, vision_utils


@beartype
def make_dataset(
    dataset_dir: os.PathLike,
    img: torch.Tensor,  # [T, IMG_C, IMG_H, CLOTH_W]
    img_parse: torch.Tensor,  # [T, 1, IMG_H, CLOTH_W]
    pose: list,  # [T]
    cloth: torch.Tensor,  # [T, CLOTH_C, CLOTH_H, CLOTH_W]
    cloth_mask: torch.Tensor,  # [T, 1, CLOTH_H, CLOTH_W]
) -> None:
    dataset_dir = utils.to_pathlib_path(dataset_dir)

    T, IMG_C, IMG_H, IMG_W, CLOTH_C, CLOTH_H, CLOTH_W = \
        -1, -2, -3, -4, -5, -6, -7

    T, IMG_C, IMG_H, IMG_W, CLOTH_C, CLOTH_H, CLOTH_W = utils.check_shapes(
        img, (T, IMG_C, IMG_H, IMG_W),
        img_parse, (T, 1, IMG_H, IMG_W),
        cloth, (T, CLOTH_C, CLOTH_H, CLOTH_W),
        cloth_mask, (T, 1, CLOTH_H, CLOTH_W),
    )

    img_dir = utils.create_dir(dataset_dir / "image", True)
    img_parse_dir = utils.create_dir(dataset_dir / "image-parse", True)
    pose_dir = utils.create_dir(dataset_dir / "pose", True)
    cloth_dir = utils.create_dir(dataset_dir / "cloth", True)
    cloth_mask_dir = utils.create_dir(dataset_dir / "cloth-mask", True)

    test_pairs_f = utils.create_file(dataset_dir / "train_pairs.txt", "w")
    train_pairs_f = utils.create_file(dataset_dir / "test_pairs.txt", "w")

    def _f(x):
        return torchvision.transforms.functional.resize(x, (512, 320))

    for t in range(T):
        cur_img = img[t]
        cur_img_parse = img_parse[t]
        cur_pose = pose[t]
        cur_cloth = cloth[t]
        cur_cloth_mask = cloth_mask[t]

        name = f"{t:06d}"

        vision_utils.write_image(
            img_dir / f"{name}.png",
            _f(cur_img))

        vision_utils.write_image(
            img_parse_dir / f"{name}_label.png",
            _f(cur_img_parse))

        utils.write_json(
            pose_dir / f"{name}_keypoints.json",
            cur_pose)

        vision_utils.write_image(
            cloth_dir / f"{name}.jpg",
            _f(cur_cloth))

        cur_cloth_mask = _f(cur_cloth_mask)

        vision_utils.write_image(
            cloth_mask_dir / f"{name}_mask.jpg",
            cur_cloth_mask.expand(3, *cur_cloth_mask.shape[1:]))

        test_pairs_f.write(f"{name}.png {name}.jpg\n")
        train_pairs_f.write(f"{name}.png {name}.jpg\n")

    train_pairs_f.close()
    test_pairs_f.close()

    subprocess.run([
        "python",
        "util/data_preprocessing.py",
        "--MPV3D_root", str(dataset_dir.resolve()),
    ],
        cwd=config.M3D_VTON_DIR,
        check=True,
    )
