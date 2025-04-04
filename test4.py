

import torch


def main1():
    axis_x = torch.tensor([0, 1, 2])
    axis_y = torch.tensor([3, 4, 5])
    axis_z = torch.tensor([6, 7, 8])

    m = torch.stack([axis_x, axis_y, axis_z], -1)

    print(f"{m=}")


if __name__ == "__main__":
    main1()
