import dataclasses
import typing

import torch
import tqdm
from beartype import beartype

from . import camera_utils, rendering_utils, transform_utils, utils


@beartype
def tex_coord_to_img_coord(
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


@beartype
def img_coord_to_tex_coord(
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
    tex_faces: torch.Tensor,  # [F, 3]
    tex_vert_pos: torch.Tensor,  # [TV, 2]

    face_colors: torch.Tensor,  # [F, C]

    img_h: int,
    img_w: int,
):
    F, TV, C = -1, -2, -3

    utils.check_shapes(
        tex_faces, (F, 3),
        tex_vert_pos, (TV, 2),
        face_colors, (F, C),
    )

    assert 0 < img_h
    assert 0 < img_w

    tf = tex_faces
    tvp = tex_coord_to_img_coord(
        tex_vert_pos.to(utils.CPU_DEVICE), img_h, img_w)
    fc = face_colors.to(utils.CPU_DEVICE)

    ret = torch.empty((C, img_h, img_w), dtype=face_colors.dtype)

    for fi in range(F):
        va, vb, vc = tf[fi]

        tvp_a, tvp_b, tvp_c = tvp[va], tvp[vb], tvp[vc]
        # [2]

        it = rendering_utils.rasterize_triangle(
            points=torch.tensor([tvp_a, tvp_b, tvp_c], dtype=utils.FLOAT),
            x_range=(0, img_h),
            y_range=(0, img_w),
        )

        for (hi, wi), (ka, kb, kc) in it:
            ret[hi][wi] = fc[fi]

    return ret


@beartype
def calc_face_idx(
    tex_vert_pos: torch.Tensor,  # [TV, 2]
    tex_faces: torch.Tensor,  # [F, 3]

    tex_h: int,
    tex_w: int,
):
    TV, F = -1, -2

    TV, F = utils.check_shapes(
        tex_vert_pos, (TV, 2),
        tex_faces, (F, 3),
    )

    assert 0 < tex_h
    assert 0 < tex_w

    length = 10

    tex_s = max(tex_h, tex_w) / length

    tex_hs = tex_h / tex_s
    tex_ws = tex_w / tex_s

    camera_config = camera_utils.CameraConfig.from_slope_udlr(
        slope_u=tex_hs / length / 2,
        slope_d=tex_hs / length / 2,
        slope_l=tex_ws / length / 2,
        slope_r=tex_ws / length / 2,
        depth_near=1e-2,
        depth_far=1e2,
        img_h=tex_h,
        img_w=tex_w,
    )

    """
    camera_config = camera_utils.CameraConfig.from_delta_hw(
        delta_h=tex_hs,
        delta_w=tex_hs,
        depth_near=1e-2,
        depth_far=1e2,
        img_h=tex_h,
        img_w=tex_w,
    )
    """

    camera_transform = transform_utils.ObjectTransform.from_matching(
        "RUB",
        pos=torch.tensor([0, 0, length], dtype=utils.FLOAT),
        device=utils.check_devices(tex_vert_pos, tex_faces),
    )

    oven_tex_vert_pos = torch.empty(
        (TV, 3), dtype=tex_vert_pos.dtype, device=tex_vert_pos.device)

    oven_tex_vert_pos[:, 0] = (tex_vert_pos[:, 0] - 0.5) * tex_ws
    oven_tex_vert_pos[:, 1] = (tex_vert_pos[:, 1] - 0.5) * tex_hs
    oven_tex_vert_pos[:, 2] = 0

    rasterize_mesh_result = rendering_utils.rasterize_mesh(
        oven_tex_vert_pos,
        tex_faces,
        camera_config,
        camera_transform,
        1,

        cull_backfaces=False,
    )

    return rasterize_mesh_result.pixel_to_faces


@beartype
def position_to_map(
    vert_pos: torch.Tensor,  # [V, D]
    faces: torch.Tensor,  # [F, 3]

    tex_vert_pos: torch.Tensor,  # [TV, 2]
    tex_faces: torch.Tensor,  # [F, 3]

    img_h: int,
    img_w: int,
) -> list[list[None | list[tuple[int, torch.Tensor]]]]:
    assert 0 < img_h
    assert 0 < img_w

    D, V, F, TV = -1, -2, -3, -4

    D, V, F, TV = utils.check_shapes(
        vert_pos, (V, D),
        faces, (F, 3),

        tex_faces, (F, 3),
        tex_vert_pos, (TV, 2),
    )

    vp = vert_pos.to(utils.CPU_DEVICE)
    faces = faces.to(utils.CPU_DEVICE)

    tvp = tex_coord_to_img_coord(
        tex_vert_pos.to(utils.CPU_DEVICE), img_h, img_w)
    tex_faces = tex_faces.to(utils.CPU_DEVICE)

    ret = [[None] * img_w for pixel_i in range(img_h)]

    for fi in tqdm.tqdm(range(F)):
        va, vb, vc = faces[fi]
        tva, tvb, tvc = tex_faces[fi]

        vp_a, vp_b, vp_c = vp[va], vp[vb], vp[vc]
        # [D]

        tvp_a, tvp_b, tvp_c = tvp[tva], tvp[tvb], tvp[tvc]
        # [2]

        ras_result = rendering_utils.rasterize_triangle(
            points=torch.tensor([
                [tvp_a[0], tvp_a[1]],
                [tvp_b[0], tvp_b[1]],
                [tvp_c[0], tvp_c[1]],
            ], dtype=utils.FLOAT),
            x_range=(0, img_h),
            y_range=(0, img_w),
        )

        for (hi, wi), (ka, kb, kc) in ras_result:
            l = ret[hi][wi]

            if l is None:
                l = ret[hi][wi] = list()

            l.append((fi, vp_a * ka + vp_b * kb + vp_c * kc))

    return ret


@dataclasses.dataclass
class TextureOven:
    camera_config: camera_utils.CameraConfig
    camera_transform: transform_utils.ObjectTransform
    tex_vert_pos: torch.Tensor  # [TV, 3]


@beartype
def make_texture_oven(
    tex_vert_pos: torch.Tensor,  # [TV, 2]
    tex_h: int,
    tex_w: int,
) -> TextureOven:
    TV = utils.check_shapes(
        tex_vert_pos, (-1, 2),
    )

    assert 0 < tex_h
    assert 0 < tex_w

    camera_config = camera_utils.CameraConfig.from_delta_hw(
        delta_h=float(tex_h),
        delta_w=float(tex_w),
        depth_near=1e-2,
        depth_far=1e2,
        img_h=tex_h,
        img_w=tex_w,
    )

    camera_transform = transform_utils.ObjectTransform.from_matching(
        "RUB",
        pos=torch.tensor([0, 0, 10], dtype=utils.FLOAT),
    )

    oven_tex_vert_pos = torch.empty(
        (TV, 3), dtype=tex_vert_pos.dtype, device=tex_vert_pos.device)

    tex_s = max(tex_h, tex_w)

    oven_tex_vert_pos[:, 0] = (tex_vert_pos[:, 0] - 0.5) * (tex_w / tex_s * 2)
    oven_tex_vert_pos[:, 1] = (tex_vert_pos[:, 1] - 0.5) * (tex_h / tex_s * 2)
    oven_tex_vert_pos[:, 2] = 0

    return TextureOven(
        camera_config=camera_config,
        camera_transform=camera_transform,
        tex_vert_pos=oven_tex_vert_pos,
    )
