import torch
import tqdm

from . import (config, dense_pose_utils, people_snapshot_utils, utils,
               vision_utils)

DTYPE = torch.float64
DEVICE = utils.CPU_DEVICE

SUBJECT_NAME = "female-4-casual"
SUBJECT_SHORT_NAME = "f4c"


def main1():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        None,
        DTYPE,
        DEVICE,
    )

    img = subject_data.video  # [T, C, H, W]

    T, C, H, W = img.shape

    with vision_utils.VideoWriter(
        config.DIR / f"people_snapshot_densepose/{SUBJECT_SHORT_NAME}.avi",
        H,
        W,
        "RGB",
        subject_data.fps,
    ) as video_writer:
        for seg_img in tqdm.tqdm(dense_pose_utils.predict(img[t] for t in range(T))):
            video_writer.write(seg_img)


if __name__ == "__main__":
    main1()
