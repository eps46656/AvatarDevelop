
import pathlib

import matplotlib.pyplot as plt
import torch

from . import (config, people_snapshot_utils, smplx_utils,
               utils, video_seg_utils, vision_utils)


DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-1-casual"


def main1():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=utils.FLOAT,
        device=DEVICE,
    )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data, DEVICE)

    video, _ = vision_utils.read_video(
        config.DIR / "nor_1747289760.avi",
        "RGB",
        device=DEVICE,
    )

    mask_dir = config.DIR / "video_seg_2025_0514_1"

    mask = vision_utils.read_video_mask(
        mask_dir / video_seg_utils.get_obj_mask_filename(
            video_seg_utils.ObjectType.UPPER_GARMENT),
        dtype=torch.float32,
        device=DEVICE,
    )[0]

    mask_color = torch.tensor([255, 0, 0], dtype=utils.FLOAT, device=DEVICE)

    mask_color = mask_color[:, None, None]  # [3, 1, 1]

    with vision_utils.VideoWriter(
        path=config.DIR / f"output_{utils.timestamp_sec()}.avi",
        fps=subject_data.fps,
        height=video.shape[2],
        width=video.shape[3],
        color_type="RGB",
    ) as out_video_writer:
        for i in range(video.shape[0]):
            img = video[i]
            mask_img = mask[i]

            out_img = img * (1 - mask_img) + mask_color * mask_img

            out_video_writer.write(out_img.clamp(0, 255).to(torch.uint8))


if __name__ == "__main__":
    main1()
