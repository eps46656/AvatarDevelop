import torch
import tqdm
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

    ret[..., 0] = (img_h - 1) * (1 - tex_coord[..., 1])
    ret[..., 1] = (img_w - 1) * tex_coord[..., 0]

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
        texture_vertex_positions.to(utils.CPU_DEVICE), img_h, img_w)
    fcs = face_colors.to(utils.CPU_DEVICE)

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
    faces: torch.Tensor,  # [F, 3]

    texture_vertex_positions: torch.Tensor,  # [TV, 2]
    texture_faces: torch.Tensor,  # [F, 3]

    img_h: int,
    img_w: int,
) -> list[list[None | list[tuple[int, torch.Tensor]]]]:
    assert 0 < img_h
    assert 0 < img_w

    D, V, F, TV = -1, -2, -3, -4

    D, V, F, TV = utils.check_shapes(
        vertex_positions, (V, D),
        faces, (F, 3),

        texture_faces, (F, 3),
        texture_vertex_positions, (TV, 2),
    )

    vps = vertex_positions.to(utils.CPU_DEVICE)
    fs = faces.to(utils.CPU_DEVICE)

    tvps = _tex_coord_to_img_coord(
        texture_vertex_positions.to(utils.CPU_DEVICE), img_h, img_w)
    tfs = texture_faces.to(utils.CPU_DEVICE)

    ret = [[None] * img_w for pixel_i in range(img_h)]

    for fi in tqdm.tqdm(range(F)):
        va, vb, vc = fs[fi]
        tva, tvb, tvc = tfs[fi]

        vpa, vpb, vpc = vps[va], vps[vb], vps[vc]
        # [D]

        tvpa, tvpb, tvpc = tvps[tva], tvps[tvb], tvps[tvc]
        # [2]

        ras_result = rendering_utils.rasterize_triangle(
            points=torch.tensor([
                [tvpa[0], tvpa[1]],
                [tvpb[0], tvpb[1]],
                [tvpc[0], tvpc[1]],
            ], dtype=utils.FLOAT),
            x_range=(0, img_h),
            y_range=(0, img_w),
        )

        for (hi, wi), (ka, kb, kc) in ras_result:
            l = ret[hi][wi]

            if l is None:
                l = ret[hi][wi] = list()

            l.append((fi, vpa * ka + vpb * kb + vpc * kc))

    return ret
