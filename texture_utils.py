import dataclasses
import typing

import einops
import numpy as np
import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, dataset_utils, pca_utils,
               rendering_utils, transform_utils, utils)


@beartype
def tex_coord_to_img_coord(
    tex_coord: torch.Tensor,  # [..., 2]
    img_h: int,
    img_w: int,
) -> torch.Tensor:  # [..., 2]
    utils.check_shapes(tex_coord, (..., 2))

    assert 0 < img_h
    assert 0 < img_w

    ret = utils.empty_like(tex_coord)

    ret[..., 0] = (img_h - 1) * (1 - tex_coord[..., 1])
    ret[..., 1] = (img_w - 1) * tex_coord[..., 0]

    return ret


@beartype
def img_coord_to_tex_coord(
    img_coord: torch.Tensor,  # [..., 2]
    img_h: int,
    img_w: int,
) -> torch.Tensor:  # [..., 2]
    utils.check_shapes(img_coord, (..., 2))

    assert 0 < img_h
    assert 0 < img_w

    ret = utils.empty_like(img_coord)

    ret[:, 0] = img_coord[:, 1] / (img_w - 1)
    ret[:, 1] = 1 - img_coord[:, 0] / (img_h - 1)

    return ret


@beartype
def draw_face_color(
    tex_faces: torch.Tensor,  # [F, 3]
    tex_vert_pos: torch.Tensor,  # [TV, 2]

    face_color: torch.Tensor,  # [F, C]

    img_h: int,
    img_w: int,
):
    F, TV, C = -1, -2, -3

    F, TV, C = utils.check_shapes(
        tex_faces, (F, 3),
        tex_vert_pos, (TV, 2),
        face_color, (F, C),
    )

    assert 0 < img_h
    assert 0 < img_w

    tf = tex_faces
    tvp = tex_coord_to_img_coord(
        tex_vert_pos.to(utils.CPU_DEVICE), img_h, img_w)
    fc = face_color.to(utils.CPU_DEVICE)

    ret = torch.empty((C, img_h, img_w), dtype=face_color.dtype)

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
def rasterize_texture_map(
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

    ret = rendering_utils.rasterize_mesh(
        oven_tex_vert_pos,
        tex_faces,
        camera_config,
        camera_transform,
        1,
        False,
    )

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
        False,
    )

    ret = rasterize_mesh_result.pix_to_face
    # [1, H, W, 1]

    return ret[0, :, :, 0]


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


@beartype
@utils.deferable
@utils.mem_clear
@torch.no_grad()
def bake_texture(
    *,
    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]

    img: torch.Tensor,  # [..., C, H, W]

    mask: torch.Tensor,  # [..., 1, H, W]
    blending_param: typing.Any,  # [...]

    avatar_blender: avatar_utils.AvatarBlender,

    tex_h: int,
    tex_w: int,

    batch_size: int = 0,
    batches_cnt: int = 0,

    device: torch.device,
) -> torch.Tensor:  # [tex_h, tex_w, C]
    assert 0 < tex_h
    assert 0 < tex_w

    avatar_model: avatar_utils.AvatarModel = avatar_blender.get_avatar_model()

    C, H, W = img.shape[-3:]

    TV = avatar_model.tex_verts_cnt
    F = avatar_model.faces_cnt

    tex_vert_color_pca_calculator = pca_utils.PCACalculator(
        n=TV + 1,
        dim=C,
        dtype=torch.float64,
        device=device,
    )

    mesh_ras_result_tmp_path = utils.allocate_tmp_dir() / "mesh_ras_result.pkl"

    mesh_ras_result_tmp = utils.PickleWriter(
        mesh_ras_result_tmp_path)

    def exit():
        mesh_ras_result_tmp.close()
        mesh_ras_result_tmp_path.unlink(missing_ok=True)

    utils.defer(exit)

    tex_f_to_vvv = torch.cat([
        avatar_model.tex_mesh_graph.f_to_vvv,  # [F, 3]

        torch.tensor(
            [[TV, TV, TV]],
            dtype=avatar_model.tex_mesh_graph.f_to_vvv.dtype,
            device=avatar_model.tex_mesh_graph.f_to_vvv.device
        ),  # [1, 3]
    ], dim=0)
    # [F + 1, 3]

    tex_vert_pos = torch.cat([
        avatar_model.tex_vert_pos,  # [TV, 2]

        torch.tensor(
            [[0, 0]],
            dtype=avatar_model.tex_vert_pos.dtype,
            device=avatar_model.tex_vert_pos.device
        ),  # [1, 2]
    ], dim=0)
    # [TV + 1, 2]

    shape = utils.broadcast_shapes(
        camera_transform,
        img.shape[:-3],
        mask.shape[:-3],
        blending_param,
    )

    camera_transform = camera_transform.expand(shape)
    img = img.expand(*shape, C, H, W)
    mask = mask.expand(*shape, 1, H, W)
    blending_param = blending_param.expand(shape)

    for batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            shape, batch_size=batch_size, batches_cnt=batches_cnt, shuffle=False)):
        utils.mem_clear()

        cur_camera_transform = camera_transform[batch_idx]
        cur_blending_param = blending_param[batch_idx]
        cur_mask = mask[batch_idx]
        cur_img = img[batch_idx]

        B = batch_idx[0].shape[0]

        avatar_model = avatar_blender(cur_blending_param)

        frag = rendering_utils.rasterize_mesh(
            vert_pos=avatar_model.vert_pos,
            faces=avatar_model.mesh_graph.f_to_vvv,
            camera_config=camera_config,
            camera_transform=cur_camera_transform,
            faces_per_pixel=1,
        )

        pix_to_face: torch.Tensor = frag.pix_to_face.reshape(B, H, W)
        bary_coord: torch.Tensor = frag.bary_coords.reshape(B, H, W, 3)

        pix_to_face = torch.where(
            0.5 <= cur_mask.reshape(B, H, W),
            pix_to_face,
            -1,
        )
        # discard pixels not on person

        pix_to_face = (pix_to_face + (F + 1)) % (F + 1)

        for b in range(B):
            mesh_ras_result_tmp.write(
                utils.tensor_serialize(pix_to_face[b], np.int32))

            mesh_ras_result_tmp.write(
                utils.tensor_serialize(bary_coord[b]))

        ref_img = einops.rearrange(
            cur_img.reshape(B, C, H, W), "b c h w -> b h w c")
        # [B, H, W, C]

        utils.mem_clear()

        tex_vert_color_pca_calculator.scatter_feed(
            idx=tex_f_to_vvv[pix_to_face],  # [B, H, W, 3]
            w=bary_coord,  # [B, H, W, 3]
            x=ref_img[:, :, :, None, :],  # [B, H, W, 1, C]
        )

    mesh_ras_result_tmp.close()

    mesh_ras_result_tmp = utils.PickleReader(mesh_ras_result_tmp_path)

    tex_vert_color_mean, tex_vert_color_pca, tex_vert_color_std = \
        tex_vert_color_pca_calculator.get_pca(True)

    # tex_vert_color_mean[TV + 1, C]
    # tex_vert_color_pca[TV + 1, C, C]
    # tex_vert_color_std[TV + 1, C]

    use_pca_threshold = 10.0

    tex_vert_color_ell_mean = tex_vert_color_mean
    # [TV + 1, C]

    inv_tex_vert_color_ell_axis = torch.where(
        (use_pca_threshold <=
            tex_vert_color_pca_calculator.sum_w)[..., None, None]
        .expand(TV + 1, C, C),

        (tex_vert_color_pca *
            tex_vert_color_std[:, :, None]).transpose(-2, -1),

        torch.eye(
            C,
            dtype=tex_vert_color_pca.dtype,
            device=tex_vert_color_pca.device
        ).expand(TV + 1, C, C)
    ).inverse()
    # [F, C, C]

    tex_vert_inlier_color_pca_calculator = pca_utils.PCACalculator(
        n=TV + 1,
        dim=C,
        dtype=torch.float64,
        device=device,
    )

    for batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            shape, batch_size=1, shuffle=False)):
        utils.mem_clear()

        pix_to_face = utils.tensor_deserialize(
            mesh_ras_result_tmp.read(),
            device=device,
        )
        # [H, W]

        bary_coord = utils.tensor_deserialize(
            mesh_ras_result_tmp.read(),
            device=device,
        )
        # [H, W, 3]

        ref_img = einops.rearrange(
            img[batch_idx].reshape(C, H, W), "c h w -> h w c")
        # [H, W, C]

        tex_vert = tex_f_to_vvv[pix_to_face]
        # [H, W, 3]

        cur_tex_vert_color_ell_mean = tex_vert_color_ell_mean[tex_vert]
        # [H, W, 3, C]

        cur_inv_tex_vert_color_ell_axis = \
            inv_tex_vert_color_ell_axis[tex_vert]
        # [H, W, 3, C, C]

        ell_coord = (
            cur_inv_tex_vert_color_ell_axis  # [H, W, 3, C, C]
            @
            (ref_img[:, :, None, :] - cur_tex_vert_color_ell_mean)
            [..., None]  # [H, W, 3, C, 1]
        )[..., 0]
        # [H, W, 3, C]

        weight = bary_coord / (0.1 + utils.vec_sq_norm(ell_coord))
        # [H, W, 3]

        tex_vert_inlier_color_pca_calculator.scatter_feed(
            idx=tex_f_to_vvv[pix_to_face],  # [H, W, 3]
            w=weight,  # [H, W, 3]
            x=ref_img[:, :, None, :],  # [H, W, 1, C]
        )

    tex_vert_inlier_color_mean, tex_vert_inlier_color_pca, tex_vert_inlier_color_std = \
        tex_vert_inlier_color_pca_calculator.get_pca(True)

    tex_frags = rasterize_texture_map(
        tex_vert_pos=tex_vert_pos,
        tex_faces=tex_f_to_vvv,

        tex_h=tex_h,
        tex_w=tex_w,
    )

    tex_pix_to_face = tex_frags.pix_to_face.reshape(tex_h, tex_w)

    tex_bary_coord = tex_frags.bary_coords.reshape(tex_h, tex_w, 3)

    tex_vert_inlier_color_mean[TV] = 1.0

    tex = (tex_vert_inlier_color_mean[tex_f_to_vvv[
        (tex_pix_to_face + (F + 1)) % (F + 1)
    ]] * tex_bary_coord[..., None]).sum(dim=-2)
    # [tex_h, tex_w, C]

    print(f"{tex.shape=})")

    return tex
