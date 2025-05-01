
from . import utils

import torch


def main1():
    q = torch.tensor(
        [0.868254, 0.0224726, 0.474455, -0.143257])

    t = torch.tensor([-0.679221, 1.00351, 3.65061])

    print(f"{q=}")

    print(f"{t=}")

    rot_mat = utils.quaternion_to_rot_mat(q, order="WXYZ", out_shape=(3, 3))
    print(f"{rot_mat=}")

    inv_rot_mat = rot_mat.T
    inv_t = -inv_rot_mat @ t

    print(f"{inv_rot_mat=}")
    print(f"{inv_t=}")


if __name__ == "__main__":
    main1()
