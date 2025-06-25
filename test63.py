import copy
import itertools
import math
import pathlib
import time
import typing

import einops
import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, config, gom_avatar_utils,
               mesh_utils, people_snapshot_utils, rendering_utils, sdf_utils,
               smplx_utils, training_utils, transform_utils, utils,
               vision_utils)

DTYPE = torch.float64
DEVICE = utils.CUDA_DEVICE

SDF_MODULE_DIR = config.DIR / "sdf_module_2025_0522_1"


ALPHA_SIGNED_DIST = (lambda epoch: 1.0)
ALPHA_EIKONAL = (lambda epoch: 1.0)

BATCH_SIZE = 64
BATCHES_CNT = 32


LR = (lambda epoch: 1e-3 * (0.1 ** (epoch / 64)))

SUBJECT_NAME = "female-4-casual"


FEMALE_3_CASUAL_BARE_TEX_PATH = config.DIR / \
    "tex_avatar_2025_0518_4" / "tex_1747581876.png"

FEMALE_3_CASUAL_OBJECT_TEX_PATH = config.DIR / \
    "tex_avatar_f3c_2025_0609_1" / "tex_1749543458.png"

FEMALE_3_CASUAL_HAIR_MODEL_DATA_PATH = config.DIR / \
    "mesh_seg_2025_0609_1" / "obj_model_data_HAIR_1749640608.pkl"

FEMALE_3_CASUAL_UPPER_GARMENT_MODEL_DATA_PATH = config.DIR / \
    "mesh_seg_2025_0609_1" / "obj_model_data_UPPER_GARMENT_1749640613.pkl"

FEMALE_3_CASUAL_LOWER_GARMENT_MODEL_DATA_PATH = config.DIR / \
    "mesh_seg_2025_0609_1" / "obj_model_data_LOWER_GARMENT_1749640619.pkl"


FEMALE_4_CASUAL_BARE_TEX_PATH = config.DIR / \
    "tex_avatar_2025_0527_1" / "tex_1748286865.png"

FEMALE_4_CASUAL_OBJECT_TEX_PATH = config.DIR / \
    "tex_avatar_2025_0529_1" / "tex_1748504545.png"

FEMALE_4_CASUAL_HAIR_MODEL_DATA_PATH = config.DIR / \
    "mesh_seg_2025_0529_2" / "obj_model_data_HAIR_1748791443.pkl"

FEMALE_4_CASUAL_UPPER_GARMENT_MODEL_DATA_PATH = config.DIR / \
    "mesh_seg_2025_0529_2" / "obj_model_data_UPPER_GARMENT_1748791447.pkl"

FEMALE_4_CASUAL_LOWER_GARMENT_MODEL_DATA_PATH = config.DIR / \
    "mesh_seg_2025_0529_2" / "obj_model_data_LOWER_GARMENT_1748791451.pkl"


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data, DTYPE, DEVICE)

    return subject_data


def load_trainer():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )

    # ---

    dataset = sdf_utils.Dataset(
        mesh_graph=model_data.mesh_graph,
        vert_pos=model_data.vert_pos,

        std=50e-3,

        shape=torch.Size((BATCH_SIZE * BATCHES_CNT,)),
    )

    # ---

    module = sdf_utils.Module(
        range_min=(-2.0, -2.0, -2.0),
        range_max=(+2.0, +2.0, +2.0),
        dtype=DTYPE,
        device=DEVICE,
    ).train()

    # ---

    trainer_core = sdf_utils.TrainerCore(
        config=sdf_utils.TrainerCoreConfig(
            proj_dir=SDF_MODULE_DIR,
            device=DEVICE,

            batch_size=BATCH_SIZE,

            lr=LR,
            betas=(0.9, 0.99),

            alpha_signed_dist=ALPHA_SIGNED_DIST,
            alpha_eikonal=ALPHA_EIKONAL,
        ),
        module=module,
        dataset=dataset,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=SDF_MODULE_DIR,
        trainer_core=trainer_core,
    )

    return trainer


@beartype
def F(
    avatar_blender: list[avatar_utils.AvatarBlender],
    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,
    blending_param: list[typing.Any],
    texture: list[torch.Tensor],
) -> tuple[
    torch.Tensor,  # zbuf[H, W]
    torch.Tensor,  # tex[H, W, C]
]:
    N = utils.all_same(
        len(avatar_blender),
        len(blending_param),
        len(texture),
    )

    H, W = camera_config.img_h, camera_config.img_w

    acc_wbuf: torch.Tensor = None
    acc_tex: torch.Tensor = None

    for i in range(N):
        cur_avatar_blender = avatar_blender[i]
        cur_blending_param = blending_param[i]
        cur_texture = texture[i]

        avatar_model: avatar_utils.AvatarModel = \
            cur_avatar_blender(cur_blending_param)

        cur_rasterize_mesh_result = rendering_utils.rasterize_mesh(
            vert_pos=avatar_model.vert_pos,
            faces=avatar_model.mesh_graph.f_to_vvv,

            camera_config=camera_config,
            camera_transform=camera_transform,

            faces_per_pixel=1,

            cull_backface=True,
        )

        cur_pix_to_face = cur_rasterize_mesh_result.pix_to_face.view(H, W)
        cur_bary_coord = cur_rasterize_mesh_result.bary_coord.view(H, W, 3)
        cur_zbuf = cur_rasterize_mesh_result.zbuf.view(H, W)

        print(f"{cur_pix_to_face.min()=}")
        print(f"{cur_pix_to_face.max()=}")

        print(f"{cur_zbuf.min()=}")
        print(f"{cur_zbuf.max()=}")

        cur_tex_coord = rendering_utils.calc_tex_coord(
            pix_to_face=cur_pix_to_face,
            bary_coord=cur_bary_coord,
            tex_f_to_vvv=avatar_model.tex_mesh_graph.f_to_vvv,
            tex_vert_pos=avatar_model.tex_vert_pos,
        )

        cur_tex = rendering_utils.sample_texture(
            texture=cur_texture,
            tex_coord=cur_tex_coord,
            wrap_mode=rendering_utils.WrapMode.MIRROR,
            sampling_mode=rendering_utils.SamplingMode.LINEAR,
        )

        cur_tex = torch.where(
            (cur_pix_to_face == -1)[..., None],
            1.0,
            cur_tex,
        )

        cur_wbuf = 1 / cur_zbuf

        if acc_wbuf is None:
            acc_wbuf = cur_wbuf
            acc_tex = cur_tex
        else:
            p = acc_wbuf < cur_wbuf
            # [H, W]

            acc_wbuf = torch.where(
                p,
                cur_wbuf,
                acc_wbuf,
            )

            acc_tex = torch.where(
                p[..., None],
                cur_tex,
                acc_tex,
            )

    return 1 / acc_wbuf, acc_tex


def load_origin_model_data(path):
    return smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )


def load_model_data(path):
    return smplx_utils.ModelData.from_state_dict(
        state_dict=utils.read_pickle(path),
        dtype=DTYPE,
        device=DEVICE,
    )


def load_tex(path):
    return einops.rearrange(
        (vision_utils.read_image(path, "RGB").image / 255.0),
        "c h w -> h w c",
    ).to(DEVICE, DTYPE)


def main1():
    female_model_data = load_origin_model_data(config.SMPL_FEMALE_MODEL_PATH)

    blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
        female_model_data)

    model_data = [
        # bare model data
        female_model_data,

        blending_coeff_field.query_model_data(load_model_data(
            FEMALE_3_CASUAL_HAIR_MODEL_DATA_PATH)),
        blending_coeff_field.query_model_data(load_model_data(
            FEMALE_3_CASUAL_UPPER_GARMENT_MODEL_DATA_PATH)),
        blending_coeff_field.query_model_data(load_model_data(
            FEMALE_3_CASUAL_LOWER_GARMENT_MODEL_DATA_PATH)),
    ]

    for i in range(len(model_data)):
        print(f"{model_data[i].pose_vert_dir.min()=}")
        print(f"{model_data[i].pose_vert_dir.max()=}")

        model_data[i].body_shape_vert_dir.fill_(0)
        model_data[i].expr_shape_vert_dir.fill_(0)
        model_data[i].pose_vert_dir.fill_(0)

        print(f"{model_data[i].lbs_weight.min()=}")
        print(f"{model_data[i].lbs_weight.max()=}")

        k = model_data[i].lbs_weight.sum(-1)

        print(f"{k.min()=}")
        print(f"{k.max()=}")

    mesh_utils.show_mesh_data(
        [
            mesh_utils.MeshData(
                mesh_graph=m.mesh_graph,
                vert_pos=m.vert_pos,
            )
            for m in model_data
        ]
    )

    female_3_casual_object_tex = load_tex(FEMALE_3_CASUAL_OBJECT_TEX_PATH)
    female_4_casual_object_tex = load_tex(FEMALE_4_CASUAL_OBJECT_TEX_PATH)

    tex = [
        load_tex(FEMALE_3_CASUAL_BARE_TEX_PATH),

        female_3_casual_object_tex,
        female_3_casual_object_tex,
        female_3_casual_object_tex,
    ]

    N = utils.all_same(len(model_data), len(tex))

    model_blender = [
        smplx_utils.ModelBlender(
            model_builder=smplx_utils.StaticModelBuilder(
                model_data=cur_model_data),
        )
        for cur_model_data in model_data
    ]

    subject_data = read_subject()

    T, C, H, W = subject_data.video.shape

    subject_data.blending_param.body_shape = None
    subject_data.blending_param.expr_shape = None

    video_writer = vision_utils.VideoWriter(
        path=config.DIR / f"rgb_{utils.timestamp_sec()}.avi",
        height=H,
        width=W,
        color_type="RGB",
        fps=subject_data.fps,
    )

    for t in tqdm.tqdm(range(T)):
        cur_blending_param = subject_data.blending_param[t]

        cur_zbuf, cur_img = F(
            avatar_blender=model_blender,
            camera_config=subject_data.camera_config,
            camera_transform=subject_data.camera_transform,
            blending_param=[cur_blending_param] * N,
            texture=tex,
        )
        # [H, W, C]

        cur_img = einops.rearrange(cur_img, "h w c -> c h w")
        # [C, H, W]

        print(f"{cur_img.min()=}")
        print(f"{cur_img.max()=}")

        cur_frame = utils.rct(cur_img * 255, dtype=torch.uint8)

        vision_utils.show_image("cur_frame", cur_frame)

        video_writer.write(cur_frame)

    video_writer.close()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(False, False):
        main1()

    print("ok")
