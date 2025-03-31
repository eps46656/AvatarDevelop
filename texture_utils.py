import torch
from beartype import beartype

from . import rendering_utils, utils


def _tex_coord_to_img_coord(
    tex_coord: torch.Tensor,  # [..., 2]
    img_h: int,
    img_w: int,
):
    utils.check_shapes(tex_coord, (..., 2))

    assert 0 < img_h
    assert 0 < img_w

    ret = torch.empty_like(tex_coord)

    ret[:, 0] = (img_h - 1) * (1 - tex_coord[:, 1])
    ret[:, 1] = (img_w - 1) * tex_coord[:, 0]

    return ret


def _img_coord_to_tex_coord(
    img_coord: torch.Tensor,  # [..., 2]
    img_h: int,
    img_w: int,
):
    utils.check_shapes(img_coord, (..., 2))

    assert 0 < img_h
    assert 0 < img_w

    ret = torch.empty_like(img_coord)

    ret[:, 0] = img_coord[:, 1] / (img_w - 1)
    ret[:, 1] = 1 - img_coord[:, 0] / (img_h - 1)

    return ret


@beartype
def draw_face_color(
    texture_faces: torch.Tensor,  # [F, 3]
    texture_vertex_positions: torch.Tensor,  # [TV, 2]

    face_colors: torch.Tensor,  # [F, C]

    img_h: int,
    img_w: int,
):
    F, TV, C = -1, -2, -3

    utils.check_shapes(
        texture_faces, (F, 3),
        texture_vertex_positions, (TV, 2),
        face_colors, (F, C),
    )

    assert 0 < img_h
    assert 0 < img_w

    tfs = texture_faces
    tvps = _tex_coord_to_img_coord(
        texture_vertex_positions.to(device=utils.CPU_DEVICE), img_h, img_w)
    fcs = face_colors.to(device=utils.CPU_DEVICE)

    ret = torch.empty((C, img_h, img_w), dtype=face_colors.dtype)

    for fi in range(F):
        va, vb, vc = tfs[fi]

        tvpa, tvpb, tvpc = tvps[va], tvps[vb], tvps[vc]
        # [2]

        it = rendering_utils.rasterize_triangle(
            points=torch.tensor([tvpa, tvpb, tvpc], dtype=utils.FLOAT),
            x_range=(0, img_h),
            y_range=(0, img_w),
        )

        for (hi, wi), (ka, kb, kc) in it:
            ret[hi][wi] = fcs[fi]

    return ret


@beartype
def position_to_map(
    vertex_positions: torch.Tensor,  # [V, D]

    texture_faces: torch.Tensor,  # [TF, 3]
    texture_vertex_positions: torch.Tensor,  # [TV, 2]

    img_h: int,
    img_w: int,
) -> torch.Tensor:  # [img_h, img_w] None | list[tuple[int, torch.Tensor[D]]]
    """
    ret[pixel_i][pixel_j] = None | [
        (face_i, position)
    ]
    """

    assert 0 < img_h
    assert 0 < img_w

    V, D, TF, TV = -1, -2, -3, -4

    utils.check_shapes(
        vertex_positions, (V, D),
        texture_faces, (TF, 3),
        texture_vertex_positions, (TV, 2),
    )

    vps = vertex_positions.to(device=utils.CPU_DEVICE)
    tfs = texture_faces.to(device=utils.CPU_DEVICE)
    tvps = _tex_coord_to_img_coord(
        texture_vertex_positions.to(device=utils.CPU_DEVICE), img_h, img_w)

    ret = torch.full((img_h, img_w), None, dtype=object)

    for fi in range(TF):
        va, vb, vc = tfs[fi]

        vpa, vpb, vpc = vps[va], vps[vb], vps[vc]
        # [D]

        tvpa, tvpb, tvpc = tvps[va], tvps[vb], tvps[vc]
        # [2]

        it = rendering_utils.rasterize_triangle(
            points=torch.tensor([tvpa, tvpb, tvpc], dtype=utils.FLOAT),
            x_range=(0, img_h),
            y_range=(0, img_w),
        )

        for (hi, wi), (ka, kb, kc) in it:
            l = ret[hi][wi]

            if l is None:
                l = ret[hi][wi] = list()

            l.append((fi, vpa * ka + vpb * kb + vpc * kc))

    return ret
