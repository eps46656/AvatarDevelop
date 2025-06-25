import torch
import copy
from beartype import beartype

from . import (config, smplx_utils, tex_avatar_utils, training_utils,
               transform_utils, utils, vision_utils)

DTYPE = torch.float64
DEVICE = torch.device("cuda")


def main1():
    temp_model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )

    blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
        temp_model_data)

    model_data = copy.copy(temp_model_data)

    model_data.vert_pos = model_data.vert_pos + torch.randn(
        size=model_data.vert_pos.shape,
        dtype=model_data.vert_pos.dtype,
        device=model_data.vert_pos.device
    ) * 5e-3
    # [V, 3]

    closest_vert_idx = utils.vec_sq_norm(
        model_data.vert_pos[:, None, :] - temp_model_data.vert_pos[None, :, :]
    ).argmin(-2)
    # [V]

    closest_vert_pos = temp_model_data.vert_pos[closest_vert_idx]
    # [V, 3]

    closest_body_shape_vert_dir = temp_model_data. \
        body_shape_vert_dir[closest_vert_idx]
    closest_pose_vert_dir = temp_model_data. \
        pose_vert_dir[closest_vert_idx]
    closest_lbs_weight = temp_model_data. \
        lbs_weight[closest_vert_idx]

    diff_vert_pos = model_data.vert_pos - closest_vert_pos
    print(f"{diff_vert_pos.abs().max()=}")
    print(f"{diff_vert_pos.abs().mean()=}")

    model_data = blending_coeff_field.query_model_data(model_data)

    new_body_shape_vert_dir = blending_coeff_field.query_body_shape_vert_dir(
        temp_model_data.vert_pos,
    )

    print(f"{model_data.vert_pos.shape=}")
    print(f"{temp_model_data.vert_pos.shape=}")

    diff_vert_pos = model_data.vert_pos - closest_vert_pos
    print(f"{diff_vert_pos.abs().max()=}")
    print(f"{diff_vert_pos.abs().mean()=}")

    print(f"{model_data.body_shape_vert_dir.shape=}")
    print(f"{closest_body_shape_vert_dir.shape=}")

    diff_body_shape_vert_dir = (
        new_body_shape_vert_dir - closest_body_shape_vert_dir
    )
    print(f"{diff_body_shape_vert_dir.abs().max()=}")
    print(f"{diff_body_shape_vert_dir.abs().mean()=}")

    diff_pose_vert_dir = (
        model_data.pose_vert_dir - closest_pose_vert_dir
    )
    print(f"{diff_pose_vert_dir.abs().max()=}")
    print(f"{diff_pose_vert_dir.abs().mean()=}")

    temp_model_data_lbs_weight_sum = closest_lbs_weight.sum(dim=-1)
    model_data_lbs_weight_sum = model_data.lbs_weight.sum(dim=-1)

    print(f"{closest_lbs_weight.min()=}")
    print(f"{closest_lbs_weight.max()=}")
    print(f"{temp_model_data_lbs_weight_sum.min()=}")
    print(f"{temp_model_data_lbs_weight_sum.max()=}")

    print(f"{model_data.lbs_weight.min()=}")
    print(f"{model_data.lbs_weight.max()=}")
    print(f"{model_data_lbs_weight_sum.min()=}")
    print(f"{model_data_lbs_weight_sum.max()=}")

    diff_lbs_weight = (
        model_data.lbs_weight - closest_lbs_weight
    )
    print(f"{diff_lbs_weight.abs().max()=}")
    print(f"{diff_lbs_weight.abs().mean()=}")


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()

    print("ok")
