import torch
from beartype import beartype

from . import utils, vision_utils


@beartype
def make_dataset(
    dataset_dir: utils.PathLike,
    img: torch.Tensor,  # [..., IMG_C, IMG_H, CLOTH_W]
    img_mask: torch.Tensor,  # [..., 1, IMG_H, CLOTH_W]
    img_densepose: torch.Tensor,  # [..., IMG_C, IMG_H, CLOTH_W]
    cloth: torch.Tensor,  # [..., CLOTH_C, CLOTH_H, CLOTH_W]
    cloth_mask: torch.Tensor,  # [..., 1, CLOTH_H, CLOTH_W]
) -> None:
    dataset_dir = utils.to_pathlib_path(dataset_dir)

    IMG_C, IMG_H, IMG_W, CLOTH_C, CLOTH_H, CLOTH_W = \
        -1, -2, -3, -4, -5, -6

    IMG_C, IMG_H, IMG_W, CLOTH_C, CLOTH_H, CLOTH_W = utils.check_shapes(
        img, (..., IMG_C, IMG_H, IMG_W),
        img_mask, (..., 1, IMG_H, IMG_W),
        img_densepose, (..., IMG_C, IMG_H, IMG_W),
        cloth, (..., CLOTH_C, CLOTH_H, CLOTH_W),
        cloth_mask, (..., 1, CLOTH_H, CLOTH_W),
    )

    shape = utils.broadcast_shapes(
        img.shape[:-3],
        img_mask.shape[:-3],
        img_densepose.shape[:-3],
        cloth.shape[:-3],
        cloth_mask.shape[:-3],
    )

    img = utils.batch_expand(img, shape, 3)
    img_mask = utils.batch_expand(img_mask, shape, 3)
    img_densepose = utils.batch_expand(img_densepose, shape, 3)
    cloth = utils.batch_expand(cloth, shape, 3)
    cloth_mask = utils.batch_expand(cloth_mask, shape, 3)

    dataset_test_dir = dataset_dir / "test"
    utils.create_dir(dataset_test_dir, False)

    img_dir = dataset_test_dir / "image"
    utils.create_dir(img_dir, True)

    img_densepose_dir = dataset_test_dir / "image-densepose"
    utils.create_dir(img_densepose_dir, True)

    img_mask_dir = dataset_test_dir / "agnostic-mask"
    utils.create_dir(img_mask_dir, True)

    masked_img_dir = dataset_test_dir / "agnostic-v3.2"
    utils.create_dir(masked_img_dir, True)

    cloth_dir = dataset_test_dir / "cloth"
    utils.create_dir(cloth_dir, True)

    cloth_mask_dir = dataset_test_dir / "cloth-mask"
    utils.create_dir(cloth_mask_dir, True)

    test_pairs_f = utils.create_file(dataset_dir / "test_pairs.txt", "w")

    for flatten_batch_idx, batch_idx in utils.get_batch_idxes(shape):
        cur_img = img[batch_idx]
        cur_img_densepose = img_densepose[batch_idx]
        cur_img_mask = img_mask[batch_idx]
        cur_cloth = cloth[batch_idx]
        cur_cloth_mask = cloth_mask[batch_idx]

        cur_masked_img = torch.where(
            (128 <= cur_img_mask).expand_as(cur_img),
            128,
            cur_img,
        )

        name = f"{flatten_batch_idx:06d}"

        vision_utils.write_image(
            img_dir / f"{name}.jpg",
            cur_img)

        vision_utils.write_image(
            img_densepose_dir / f"{name}.jpg",
            cur_img_densepose)

        vision_utils.write_image(
            img_mask_dir / f"{name}_mask.png",
            cur_img_mask)

        vision_utils.write_image(
            masked_img_dir / f"{name}.jpg",
            cur_masked_img)

        vision_utils.write_image(
            cloth_dir / f"{name}.jpg",
            torch.where(
                (128 <= cur_cloth_mask).expand_as(cur_cloth),
                cur_cloth,
                255,
            )
        )

        vision_utils.write_image(
            cloth_mask_dir / f"{name}.jpg",
            cur_cloth_mask)

        test_pairs_f.write(f"{name}.jpg {name}.jpg\n")

    test_pairs_f.close()
