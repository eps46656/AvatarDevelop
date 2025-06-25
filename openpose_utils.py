import os

import openpose
import torch
from beartype import beartype

from . import config, utils, vision_utils

params = {
    "model_folder": "models/",
    "write_json": "output_json/",
    "display": 1,
    "render_pose": 1,
    "disable_blending": True,
    "hand": False,
    "face": False,
}

opWrapper = openpose.pyopenpose.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


@beartype
def predict(
    img: torch.Tensor  # [C, H, W]
):
    print(f"{img.shape=}")

    opencv_img = vision_utils.to_opencv_image(img)

    print(f"{opencv_img.shape=}")

    datum = openpose.pyopenpose.Datum()

    datum.cvInputData = opencv_img

    opWrapper.emplaceAndPop([datum])

    pose_keypoints = datum.poseKeypoints

    print(f"{pose_keypoints=}")
