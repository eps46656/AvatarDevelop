import enum
import typing

import einops
import numpy as np
import torch
import tqdm
from beartype import beartype

from .. import (avatar_utils, camera_utils, dataset_utils, pca_utils,
                transform_utils, utils)
from .ras_mesh import *


@beartype
def tex_coord_to_img_idx(
    tex_coord: torch.Tensor,  # [..., 2]
    img_h: int,
    img_w: int,
) -> torch.Tensor:  # [..., 2]
    utils.check_shapes(tex_coord, (..., 2))

    assert 0 < img_h
    assert 0 < img_w

    return torch.stack([
        (img_h - 1) * (1 - tex_coord[..., 1]),  # [....]
        (img_w - 1) * tex_coord[..., 0],  # [...]
    ], dim=-1)  # [..., 2]


@beartype
def img_idx_to_tex_coord(
    img_coord: torch.Tensor,  # [..., 2]
    img_h: int,
    img_w: int,
) -> torch.Tensor:  # [..., 2]
    utils.check_shapes(img_coord, (..., 2))

    assert 1 < img_h
    assert 1 < img_w

    return torch.stack([
        img_coord[:, 1] / (img_w - 1),  # [....]
        1 - img_coord[:, 0] / (img_h - 1),  # [...]
    ], dim=-1)  # [..., 2]


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

    camera_transform = transform_utils.ObjectTransform.from_matching(
        "RUB",
        pos=torch.tensor([0, 0, length], dtype=utils.FLOAT),
        device=utils.check_devices(tex_vert_pos, tex_faces),
    )

    oven_tex_vert_pos = torch.stack([
        (tex_vert_pos[:, 0] - 0.5) * tex_ws,  # [TV]
        (tex_vert_pos[:, 1] - 0.5) * tex_hs,  # [TV]
        utils.dummy_full(0, like=tex_vert_pos, shape=(TV,))
    ], dim=-1)  # [TV, 3]

    ret = rasterize_mesh(
        oven_tex_vert_pos,
        tex_faces,
        camera_config,
        camera_transform,
        1,
        False,
    )

    return ret


@beartype
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

    tmp_pix_to_face = utils.disk_empty((*shape, H, W), torch.int64)
    tmp_bary_coord = utils.disk_empty((*shape, H, W, 3), torch.float64)

    tex_vert_color_pca_calculator = pca_utils.PCACalculator(
        n=TV + 1,
        dim=C,
        dtype=torch.float64,
        device=device,
    )

    for B, batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            shape, batch_size=batch_size, batches_cnt=batches_cnt, shuffle=False)):
        utils.mem_clear()

        cur_camera_transform = camera_transform[batch_idx]
        cur_blending_param = blending_param[batch_idx]
        cur_mask = mask[batch_idx]
        cur_img = img[batch_idx]

        avatar_model = avatar_blender(cur_blending_param)

        mesh_ras_result = rasterize_mesh(
            vert_pos=avatar_model.vert_pos,
            faces=avatar_model.mesh_graph.f_to_vvv,
            camera_config=camera_config,
            camera_transform=cur_camera_transform,
            faces_per_pixel=1,
        )

        pix_to_face = mesh_ras_result.pix_to_face[..., 0]
        # [..., H, W, 1] -> [..., H, W]

        bary_coord: torch.Tensor = mesh_ras_result.bary_coord[..., 0, :]
        # [..., H, W, 1, 3] -> [..., H, W, 3]

        pix_to_face = torch.where(
            0.5 <= cur_mask.reshape(B, H, W),
            pix_to_face,
            -1,
        )
        # discard pixels not on person

        pix_to_face = (pix_to_face + (F + 1)) % (F + 1)

        tmp_pix_to_face[batch_idx] = pix_to_face.to(
            utils.CPU_DEVICE, torch.int64)
        tmp_bary_coord[batch_idx] = bary_coord.to(
            utils.CPU_DEVICE, torch.float64)

        ref_img = einops.rearrange(
            cur_img.reshape(B, C, H, W), "b c h w -> b h w c")
        # [B, H, W, C]

        utils.mem_clear()

        tex_vert_color_pca_calculator.scatter_feed(
            idx=tex_f_to_vvv[pix_to_face],  # [B, H, W, 3]
            w=bary_coord,  # [B, H, W, 3]
            x=ref_img[:, :, :, None, :],  # [B, H, W, 1, C]
        )

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

        utils.dummy_eye(
            shape=(TV + 1, C, C),
            dtype=tex_vert_color_pca.dtype,
            device=tex_vert_color_pca.device,
        )
    ).inverse()
    # [F, C, C]

    tex_vert_inlier_color_pca_calculator = pca_utils.PCACalculator(
        n=TV + 1,
        dim=C,
        dtype=torch.float64,
        device=device,
    )

    for B, batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            shape, batch_size=1, shuffle=False)):
        utils.mem_clear()

        pix_to_face = tmp_pix_to_face[batch_idx].to(device)
        # [H, W]

        bary_coord = tmp_bary_coord[batch_idx].to(device)
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

    tex_bary_coord = tex_frags.bary_coord.reshape(tex_h, tex_w, 3)

    tex_vert_inlier_color_mean[TV] = 1.0

    tex = (tex_vert_inlier_color_mean[tex_f_to_vvv[
        (tex_pix_to_face + (F + 1)) % (F + 1)
    ]] * tex_bary_coord[..., None]).sum(dim=-2)
    # [tex_h, tex_w, C]

    utils.print_cur_pos()
    print(f"{tex.shape=})")

    return tex


class WrapMode(enum.Enum):
    REPEAT = enum.auto()
    CLAMP = enum.auto()
    MIRROR = enum.auto()


class SamplingMode(enum.Enum):
    NEAREST = enum.auto()
    LINEAR = enum.auto()
    CUBIC = enum.auto()


def _wrap_idx(
    idx: torch.Tensor,  # [...]
    size: int,
    wrap_mode: WrapMode,
):
    assert 0 < size

    match wrap_mode:
        case WrapMode.REPEAT:
            return idx % size

        case WrapMode.CLAMP:
            return idx.clamp(0, size - 1)

        case WrapMode.MIRROR:
            idx = idx % (2 * size - 2)
            return (size - 1) - torch.abs(idx - (size - 1))

        case _:
            utils.MismatchException()


@beartype
def calc_tex_coord(
    pix_to_face: torch.Tensor,  # [...]
    bary_coord: torch.Tensor,  # [..., 3]
    tex_f_to_vvv: torch.Tensor,  # [F, 3]
    tex_vert_pos: torch.Tensor,  # [TV, 2]
) -> torch.Tensor:  # [..., 2]
    tex_vvv = tex_f_to_vvv[pix_to_face]
    # [..., 3]

    tex_vvv_pos = tex_vert_pos[tex_vvv]
    # [..., 3, 2]

    tex_coord = (bary_coord[..., :, None] * tex_vvv_pos).sum(dim=-2)
    # [..., 2]

    return tex_coord


@beartype
def sample_texture(
    *,
    texture: torch.Tensor,  # [H, W, C]
    tex_coord: torch.Tensor,  # [..., 2]
    wrap_mode: WrapMode,
    sampling_mode: SamplingMode,

    int_type: torch.dtype = torch.int32,
) -> torch.Tensor:  # [..., C]
    int_info = torch.iinfo(int_type)

    H, W, C = -1, -2, -3

    H, W, C = utils.check_shapes(
        texture, (H, W, C),
        tex_coord, (..., 2),
    )

    assert 0 < H
    assert H <= int_info.max // 2

    assert 0 < W
    assert W <= int_info.max // 2

    # assert

    img_idx = tex_coord_to_img_idx(tex_coord, H, W)
    # [..., 2]

    if sampling_mode == SamplingMode.NEAREST:
        img_idx = img_idx.round()

        img_idx_x = _wrap_idx(img_idx[..., 0], H, wrap_mode)
        img_idx_y = _wrap_idx(img_idx[..., 1], W, wrap_mode)

        return texture[img_idx_x, img_idx_y, :]

    if sampling_mode == SamplingMode.LINEAR:
        img_idx_x = img_idx[..., 0]
        img_idx_xa = img_idx_x.detach().floor().to(int_type)
        img_idx_xb = img_idx_x.detach().ceil().to(int_type)
        img_idx_xd = img_idx_x - img_idx_xa
        # [...]

        img_idx_y = img_idx[..., 1]
        img_idx_ya = img_idx_y.detach().floor().to(int_type)
        img_idx_yb = img_idx_y.detach().ceil().to(int_type)
        img_idx_yd = img_idx_y - img_idx_ya
        # [...]

        img_idx_xa = _wrap_idx(img_idx_xa, H, wrap_mode)
        img_idx_xb = _wrap_idx(img_idx_xb, H, wrap_mode)
        img_idx_ya = _wrap_idx(img_idx_ya, W, wrap_mode)
        img_idx_yb = _wrap_idx(img_idx_yb, W, wrap_mode)
        # [...]

        img_idx_xd = img_idx_xd[..., None]
        img_idx_yd = img_idx_yd[..., None]
        img_idx_xdyd = img_idx_xd * img_idx_yd
        # [..., 1]

        val_aa = texture[img_idx_xa, img_idx_ya, :]
        val_ab = texture[img_idx_xa, img_idx_yb, :]
        val_ba = texture[img_idx_xb, img_idx_ya, :]
        val_bb = texture[img_idx_xb, img_idx_yb, :]
        # [..., C]

        return \
            val_aa * (1 - img_idx_xd - img_idx_yd + img_idx_xdyd) + \
            val_ab * (img_idx_yd - img_idx_xdyd) + \
            val_ba * (img_idx_xd - img_idx_xdyd) + \
            val_bb * (img_idx_xdyd)

    raise utils.MismatchException()
