

import numpy as np
import torch

from . import cloth3d_utils, config, transform_utils, utils

DTYPE = torch.float64
DEVICE = utils.CPU_DEVICE


def main1():
    SUBJECT_DIR = config.CLOTH3D_DIR / "train" / "00001"

    """
    d = cloth3d_utils.load_info(SUBJECT_DIR)


    print(f"{d["trans"].shape=}")
    print(f"{d["shape"].shape=}")
    print(f"{d["poses"].shape=}")

    print(f"{d=}")
    """

    """

    d={
        'lights': {'data': {'pwr': 3.734491922953544, 'rot': array([0.77491479, 0.84303283, 0.        ])}, 'type': 'sun'},

        'camLoc': array([4.59356138, 0.        , 1.        ]),

        'shape': array([-2.71335335,  0.13721677, -0.96557967,  2.15569137, -2.20862524,
        2.02085724,  1.32704986, -0.18951267, -2.37554491,  0.31818445]),


        'gender': 0,

        'zrot': 0.8544361893345753,

        'outfit': {
            'Top': {'fabric': 'silk', 'texture': {'type': 'pattern'}},
            'Skirt': {'fabric': 'cotton', 'texture': {'type': 'pattern'}}},

        'trans': array([[-6.24258763e-01, -6.24288595e-01, -6.24638245e-01,


    """

    # camera_pos = d["camLoc"]

    camera_pos = np.array([1, 2, 3], dtype=np.float32)

    print(f"{camera_pos=}")

    print(cloth3d_utils.extrinsic(camera_pos))

    opencv_camera_view_transform = \
        transform_utils.ObjectTransform.from_matching("RDF")
    # camera <-> view

    my_camera_view_transform = cloth3d_utils.make_camera_transform(
        camera_pos, DTYPE, DEVICE)
    # camera <-> world

    my_extrinsic = my_camera_view_transform.get_trans_to(
        opencv_camera_view_transform)

    print(f"{my_extrinsic=}")


def main2():
    subject_data = cloth3d_utils.read_subject(
        subject_dir=config.CLOTH3D_DIR / "train" / "00001",
        dtype=DTYPE,
        device=DEVICE,
    )

    pass


if __name__ == "__main__":
    main2()
