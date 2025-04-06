

import torch
from beartype import beartype

from . import utils


@beartype
def vote(
    pixel_to_face:  torch.Tensor,  # [..., H, W]
    masks: torch.Tensor,  # [..., H, W]
    face_ballot_box: torch.Tensor,  # [F]
) -> None:
    # H image height
    # W image width
    # F number of faces

    H, W, F = -1, -2, -3

    H, W, F = utils.check_shapes(
        pixel_to_face, (..., H, W),
        masks, (..., H, W),
        face_ballot_box, (F,),
    )

    batch_shape = utils.broadcast_shapes(
        pixel_to_face.shape[:-2],
        masks.shape[:-2],
    )

    pixel_to_face = pixel_to_face.expand(batch_shape + (H, W))
    masks = masks.expand(batch_shape + (H, W))
    face_ballot_box = face_ballot_box.expand(batch_shape + (F,))

    pixel_to_face = pixel_to_face.view(batch_shape + (H * W,))
    # [..., H * W]

    masks = masks.view(batch_shape + (H * W,))
    # [..., H * W]

    face_ballot_box.scatter_add_(-1, pixel_to_face, masks)

    """

    face_ballot_box[..., pixel_to_face[..., i]] += masks[..., i]

    """


@beartype
def batch_vote(
    pixel_to_face:  torch.Tensor,  # [H, W]
    masks: torch.Tensor,  # [B, H, W]
    face_ballot_box: torch.Tensor,  # [F, B]
):
    # B number of parts
    # H image height
    # W image width
    # F number of faces

    N, B, H, W, F = -1, -2, -3, -4

    N, B, H, W = utils.check_shapes(
        pixel_to_face, (N, H, W),
        masks, (N, B, H, W),
        face_ballot_box, (F, B),
    )

    for n in range(N):
        for b in range(B):
            vote(
                pixel_to_face,  # [H, W]
                masks[b],  # [H, W]
                b,
                face_ballot_box[:, b],  # [F]
            )


@beartype
def elect(
    pixel_to_face: list[torch.Tensor],  # [H, W]
    masks: torch.Tensor,  # [..., K, H, W]
    faces_cnt: int,
):
    assert 0 <= faces_cnt

    K, H, W = -1, -2, -3

    K, H, W = utils.check_shapes(
        pixel_to_face, (..., H, W),
        masks, (..., K, H, W),
    )

    batch_shape = utils.broadcast_shapes(
        pixel_to_face.shape[:-2],
        masks.shape[:-2],
    )

    pixel_to_face = pixel_to_face.expand(batch_shape + (H, W))
    masks = masks.expand(batch_shape + (K, H, W))

    face_ballot_box = torch.zeros(
        (K, faces_cnt + 1),  # F + 1 for -1 pixel to face index
        dtype=masks.dtype,
        device=masks.device,
    )

    for batch_idxes in utils.get_batch_idxes(batch_shape):
        for k in range(K):
            vote(
                pixel_to_face[batch_idxes],  # [H, W]
                masks[*batch_idxes, k],  # [H, W]
                face_ballot_box[k],
            )

    face_obj_idx = face_ballot_box.argmax(0)
    # [F]

    pass


@beartype
def assign(
    face_ballot_box: torch.Tensor,  # [..., K, F]
    threshold: float = 0.7,
    bg_idx: int = -1,
):
    assert 0 < threshold
    assert threshold <= 1

    K, F = -1, -2

    K, F = utils.check_shapes(face_ballot_box, (..., K, F))

    votes_cnt = face_ballot_box.sum(-2, True)
    # [..., 1, F]

    scores = face_ballot_box / (1 + votes_cnt)
    # [..., B, F]

    max_score, max_obj = scores.max(-2)
    # max_score[..., F]
    # max_obj[..., F]

    return torch.where(
        threshold <= max_score,
        max_obj,
        bg_idx,
    )
