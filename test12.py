
import math
import pathlib

import torch
import pytorch3d

import utils
import camera_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


DEVICE = torch.device("cpu")


def main1():
    view_mat = torch.eye(4, dtype=utils.FLOAT)

    cameras = pytorch3d.renderer.PerspectiveCameras(
        R=view_mat[:3, :3].transpose(0, 1).unsqueeze(0),
        T=view_mat[:3, 3].unsqueeze(0),
        focal_length=[camera_utils.GetFocalLengthByDiagFoV(
            img_shape, 45 * utils.DEG)],
        principal_point=[(img_shape[1] / 2, img_shape[0] / 2)],
        in_ndc=False,
        device=DEVICE,
    )


if __name__ == "__main__":
    main1()
