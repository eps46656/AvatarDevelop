
import numpy as np
import scipy
import sklearn
import torch

from . import pca_utils, utils


def my_pca(
    x: torch.Tensor,  # [N, C]
) -> tuple[torch.Tensor, torch.Tensor]:
    N, C = -1, -2

    N, C = utils.check_shapes(x, (N, C))

    cnt = torch.zeros((1,), dtype=torch.int, device=x.device)
    sum_x = torch.zeros((1, C), dtype=x.dtype, device=x.device)
    sum_xxt = torch.zeros((1, C, C), dtype=x.dtype, device=x.device)

    pca_utils.scatter_feed(
        idx=torch.zeros((N,), dtype=torch.int, device=x.device),
        x=x,
        inplace=True,
        dst_sum_w=cnt,
        dst_sum_w_x=sum_x,
        dst_sum_w_xxt=sum_xxt,
    )

    x_means, x_pcas, x_stds = pca_utils.get_pca(cnt, sum_x, sum_xxt)

    print(f"{x_means=}")
    print(f"{x_pcas=}")
    print(f"{x_stds=}")


def sci_pca(
    x: torch.Tensor,  # [N, C]
) -> tuple[torch.Tensor, torch.Tensor]:
    N, C = -1, -2

    N, C = utils.check_shapes(x, (N, C))

    pca = sklearn.decomposition.PCA(n_components=C)
    x_pca = pca.fit_transform(x)

    print("sci pca:")
    print(pca.components_)

    print("sci std:")
    print(np.sqrt(pca.explained_variance_))


def main1():
    B = 5
    C = 3

    x = torch.rand(B, C)
    # [B, 3]

    my_pca(x)

    sci_pca(x)


if __name__ == "__main__":
    main1()
