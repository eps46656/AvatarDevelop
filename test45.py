import torch
from beartype import beartype

from . import config, utils

DEVICE = utils.CUDA_DEVICE


def calc_face_normal(
    face_vert_pos: torch.Tensor,  # [..., 3, 3]
) -> torch.Tensor:  # [..., 3]
    utils.check_shapes(
        face_vert_pos, (..., 3, 3),
    )

    face_vert_pos_a = face_vert_pos[..., 0, :]
    face_vert_pos_b = face_vert_pos[..., 1, :]
    face_vert_pos_c = face_vert_pos[..., 2, :]

    face_normal = utils.vec_cross(
        face_vert_pos_b - face_vert_pos_a,
        face_vert_pos_c - face_vert_pos_a,
    )
    # [..., 3]

    return utils.vec_normed(face_normal)


@beartype
def calc_bary_coord_mat(
    face_vert_pos: torch.Tensor,  # [..., 3, 3]
    face_normal: torch.Tensor,  # [..., 3]
) -> torch.Tensor:  # [..., 4, 4]
    print(f"{face_vert_pos=}")

    utils.check_shapes(
        face_vert_pos, (..., 3, 3),
        face_normal, (..., 3),
    )

    shape = utils.broadcast_shapes(
        face_vert_pos.shape[:-2],
        face_normal.shape[:-1],
    )

    face_vert_pos = face_vert_pos.expand(*shape, 3, 3).detach()
    face_normal = face_normal.expand(*shape, 3).detach()

    A = utils.zeros_like(face_vert_pos, shape=(*shape, 20, 16))

    for i in range(4):
        p = 4 * i
        q = p + 3

        A[..., p:q, p:q] = face_vert_pos
        A[..., p:q, q] = 1
        A[..., q, p:q] = face_normal

        A[..., 16 + i, i:-4:4] = 1

    print(f"{A=}")

    b = torch.tensor([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        0, 0, 0, 1,
    ], dtype=face_vert_pos.dtype, device=face_vert_pos.device)

    s = torch.linalg.lstsq(A, b)

    M = s.solution.view(*shape, 4, 4)
    # [..., 4, 4]

    a = utils.empty_like(M, shape=(*shape, 4, 4))

    a[..., :3, :3] = face_vert_pos.transpose(-2, -1)
    a[..., :3, 3] = face_normal
    a[..., 3, :3] = 1
    a[..., 3, 3] = 0

    err = (M @ a)

    print(f"{err=}")

    print(f"{M[..., :3, :].sum(dim=-2)=}")

    print(f"{M=}")

    return M


def main1():
    torch.set_printoptions(linewidth=200)

    face_vert_pos = torch.rand((3, 3))

    calc_bary_coord_mat(
        face_vert_pos=face_vert_pos,
        face_normal=calc_face_normal(face_vert_pos),
    )


if __name__ == "__main__":
    main1()
