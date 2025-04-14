
import torch
import matplotlib.pyplot as plt


from . import utils


def main1():
    x = torch.linspace(-1, 1, 100000)

    y = utils.smooth_clamp(
        x=x,  # clamp local scale
        x_lb=0.001,
        x_rb=0.010,

        slope_l=0.0,
        slope_r=0.005,
    )

    plt.plot(x.numpy(), y.numpy())  # 把 tensor 轉成 numpy，matplotlib 需要
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    main1()
