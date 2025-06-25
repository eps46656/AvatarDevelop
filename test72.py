import torch

from . import config, dense_pose_utils, people_snapshot_utils, vision_utils

DTYPE = torch.float64
DEVICE = config.CPU_DEVICE


def main1():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / "female-3-casual",
        None,
        DTYPE,
        DEVICE,
    )

    img = subject_data.video  # [T, C, H, W]

    T, C, H, W = img.shape

    dense_pose_utils.predict(img[0])


if __name__ == "__main__":
    main1()
