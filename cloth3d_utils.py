import copy
import dataclasses
import enum
import math
import os
import struct

import einops
import numpy as np
import scipy
import scipy.io
import scipy.io.matlab
import torch
from beartype import beartype

from . import (camera_utils, config, mesh_utils, smplx_utils, transform_utils,
               utils, vision_utils)

cloth3d_camera_config = camera_utils.CameraConfig(
    proj_type=camera_utils.ProjType.PERS,
    foc_u=27 * 0.5 / 66,
    foc_d=27 * 0.5 / 66,
    foc_l=36 * 0.5 / 66,
    foc_r=36 * 0.5 / 66,
    depth_near=utils.DEPTH_NEAR,
    depth_far=utils.DEPTH_FAR,
    img_h=480,
    img_w=640,
)


class FabricType(enum.Enum):
    COTTON = enum.auto()
    SILK = enum.auto()
    WOOL = enum.auto()
    LEATHER = enum.auto()
    DENIM = enum.auto()
    POLYESTER = enum.auto()
    NYLON = enum.auto()
    ACRYLIC = enum.auto()
    RAYON = enum.auto()


@dataclasses.dataclass
class GarmentData:
    name: str
    fabric_type: FabricType

    mesh: mesh_utils.MeshGraph
    tex_mesh: mesh_utils.MeshGraph  # texture mesh

    vert_pos: torch.Tensor  # [V, 3]
    tex_vert_pos: torch.Tensor  # [TV, 2]

    posed_vert_pos: torch.Tensor  # [T, V, 3]


@dataclasses.dataclass
class SubjectData:
    gender: smplx_utils.Gender

    camera_config: camera_utils.CameraConfig
    camera_transform: transform_utils.ObjectTransform

    blending_param: smplx_utils.BlendingParam

    model_data: smplx_utils.ModelData

    garment_info: dict[str]
    garment: dict[str]

    video: torch.Tensor   # [T, C, H, W]
    mask: torch.Tensor  # [T, 1, H, W]

    fps: float


@beartype
def _todict(matobj: scipy.io.matlab._mio5_params.mat_struct) -> dict:
    d = dict()

    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]

        if isinstance(elem, scipy.io.matlab._mio5_params.mat_struct):
            d[strg] = _todict(elem)
            continue

        if isinstance(elem, np.ndarray) and any(
            isinstance(item, scipy.io.matlab._mio5_params.mat_struct)
            for item in elem
        ):
            d[strg] = [
                _todict(item) if isinstance(
                    item, scipy.io.matlab._mio5_params.mat_struct) else item
                for item in elem
            ]

            continue

        print(f"{type(elem)=}")

        d[strg] = elem

    return d


@beartype
def _check_keys(d: dict) -> dict[str, dict]:
    return {
        key: _todict(value)
        if isinstance(value, scipy.io.matlab._mio5_params.mat_struct) else value
        for key, value in d.items()
    }


@beartype
def _load_info_dict(path: os.PathLike):
    path = utils.to_pathlib_path(path)

    data = scipy.io.loadmat(
        path / "info.mat", struct_as_record=False, squeeze_me=True)

    del data["__globals__"]
    del data["__header__"]
    del data["__version__"]

    return _check_keys(data)


@beartype
def make_z_rot_matrix(
    z_rot: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:  # [4, 4]
    s = math.sin(z_rot)
    c = math.cos(z_rot)

    return torch.tensor([
        [c, -s, 0, 0],
        [s, +c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=dtype, device=device)


@beartype
def make_camera_transform(
    camera_pos: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> transform_utils.ObjectTransform:
    camera_pos = camera_pos.flatten()

    return transform_utils.ObjectTransform.from_matching(
        "BRU",
        pos=torch.tensor([
            float(camera_pos[0]),
            float(camera_pos[1]),
            float(camera_pos[2]),
        ], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
    )


@beartype
def read_video(
    subject_dir: os.PathLike,
    device: torch.device,
) -> vision_utils.VideoReader:
    subject_dir = utils.to_pathlib_path(subject_dir)

    with vision_utils.VideoReader(
        subject_dir / f"{subject_dir.name}.mkv",
        "RGB",
    ) as video_reader:
        pass

    vision_utils.read_video(
        subject_dir / f"{subject_dir.name}.mkv",
        "RGB",
        device=device,
    )


@beartype
def read_mask(subject_dir: os.PathLike) -> vision_utils.VideoReader:
    subject_dir = utils.to_pathlib_path(subject_dir)

    return vision_utils.VideoReader(
        subject_dir / f"{subject_dir.name}_segm.mkv",
        "GRAY",
    )


@beartype
def read_pc16(
    path: os.PathLike,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:  # [T, V, 3]
    bytes = 2

    with open(path, "rb") as f:
        f.seek(16)

        V = struct.unpack("<i", f.read(4))[0]

        f.seek(28)

        T = struct.unpack("<i", f.read(4))[0]

        ret = torch.from_numpy(np.frombuffer(
            f.read(T * V * 3 * bytes),
            dtype=np.float16,
        )).to(device, dtype, copy=True)

    return ret.view(T, V, 3)


@beartype
def read_canonical_garment(
    *,
    subject_dir: os.PathLike,
    garment_name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    mesh_utils.MeshGraph,  # mesh
    mesh_utils.MeshGraph,  # tex_mesh
    torch.Tensor,  # [V, 3]
    torch.Tensor,  # [TV, 2]
]:
    obj_data = mesh_utils.read_obj(
        open(utils.to_pathlib_path(subject_dir) / f"{garment_name}.obj", "r"))

    mesh_graph = mesh_utils.MeshGraph.from_faces(
        len(obj_data.vert_pos),
        torch.tensor(
            obj_data.vert_pos_faces,
            dtype=torch.long, device=device
        ),
    )

    tex_mesh = mesh_utils.MeshGraph.from_faces(
        len(obj_data.tex_vert_pos),
        torch.tensor(
            obj_data.tex_vert_pos_faces,
            dtype=torch.long, device=device
        ),
    )

    vert_pos = torch.tensor(
        obj_data.vert_pos, dtype=dtype, device=device)

    tex_vert_pos = torch.tensor(
        obj_data.tex_vert_pos, dtype=dtype, device=device)

    return mesh_graph, tex_mesh, vert_pos, tex_vert_pos


@beartype
def read_posed_garment_vert_pos(
    *,
    subject_dir: os.PathLike,
    garment_name: str,
    transl: torch.Tensor,  # [T, 3]
    z_rot: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:  # [T, V, 3]
    rel_vert_pos = read_pc16(
        utils.to_pathlib_path(subject_dir) / f"{garment_name}.pc16",
        dtype=dtype,
        device=device,
    )
    # [T, V, 3]

    z_rot_mat = make_z_rot_matrix(z_rot, dtype, device)[:3, :3]
    # [3, 3]

    return (
        z_rot_mat @ (rel_vert_pos + transl[:, None, :])[:, :, :, None]
    )[:, :, :, 0]


@beartype
def read_garment(
    *,
    subject_dir: os.PathLike,
    garment_name: str,

    transl: torch.Tensor,  # [T, 3]
    z_rot: float,

    fabric_type: FabricType,

    dtype: torch.dtype,
    device: torch.device,
) -> GarmentData:
    subject_dir = utils.to_pathlib_path(subject_dir)

    mesh, tex_mesh, vert_pos, tex_vert_pos = read_canonical_garment(
        subject_dir=subject_dir,
        garment_name=garment_name,
        dtype=dtype,
        device=device,
    )

    posed_garment_vert_pos = read_posed_garment_vert_pos(
        subject_dir=subject_dir,
        garment_name=garment_name,
        transl=transl,
        z_rot=z_rot,
        dtype=dtype,
        device=device,
    )

    return GarmentData(
        name=garment_name,
        fabric_type=fabric_type,

        mesh=mesh,
        tex_mesh=tex_mesh,

        vert_pos=vert_pos,
        tex_vert_pos=tex_vert_pos,

        posed_vert_pos=posed_garment_vert_pos,
    )


@beartype
def read_subject(
    subject_dir: os.PathLike,
    dtype: torch.dtype,
    device: torch.device,
):
    subject_dir = utils.to_pathlib_path(subject_dir)

    info_dict = _load_info_dict(subject_dir)

    camera_config = copy.copy(cloth3d_camera_config)

    camera_transform = make_camera_transform(
        info_dict["camLoc"],
        dtype=dtype,
        device=device,
    )

    gender = smplx_utils.Gender.FEMALE if info_dict["gender"] == 0 \
        else smplx_utils.Gender.MALE

    match gender:
        case smplx_utils.Gender.FEMALE:
            model_data = smplx_utils.ModelData.from_origin_file(
                model_data_path=config.SMPL_FEMALE_MODEL_PATH,
                model_config=smplx_utils.smpl_model_config,
                dtype=dtype,
                device=device,
            )
        case smplx_utils.Gender.MALE:
            model_data = smplx_utils.ModelData.from_origin_file(
                model_data_path=config.SMPL_MALE_MODEL_PATH,
                model_config=smplx_utils.smpl_model_config,
                dtype=dtype,
                device=device,
            )

        case _:
            raise utils.MismatchException()

    body_shape = torch.from_numpy(info_dict["shape"])
    # [10]

    transl = torch.from_numpy(info_dict["trans"])
    # [3, T]

    poses = torch.from_numpy(info_dict["poses"])
    # [72, T]

    T = utils.check_shapes(
        transl, (3, -1),
        body_shape, (model_data.body_shapes_cnt,),
        poses, (model_data.joints_cnt * 3, -1),
    )

    z_rot = info_dict["zrot"]

    z_rot_mat = make_z_rot_matrix(z_rot, dtype, device)

    transl = einops.rearrange(transl, "d t -> t d")
    # [T, 3]

    blending_param = smplx_utils.BlendingParam(
        body_shape=body_shape,

        global_transl=(
            z_rot_mat[:3, :3] @ transl.to(device, dtype)[:, :, None]
        )[:, :, 0],

        global_rot=einops.rearrange(poses[:3, :], "d t -> t d"),

        body_pose=einops.rearrange(
            poses[3:, :].view(model_data.joints_cnt - 1, 3, T),
            "b d t -> t b d",
        ),

        dtype=dtype,
        device=device,
    )

    garment_info = info_dict["outfit"]

    garment = {
        gname: read_garment(
            subject_dir=subject_dir,
            garment_name=gname,

            transl=transl,
            z_rot=z_rot,

            fabric_type=FabricType[ginfo["fabric"].upper()],

            dtype=dtype,
            device=device,
        )

        for gname, ginfo in garment_info.items()
    }

    video, fps = vision_utils.read_video(
        subject_dir / f"{subject_dir.name}.mkv",
        "RGB",
        device=device,
    )

    mask = vision_utils.read_video(
        subject_dir / f"{subject_dir.name}_segm.mkv",
        "GRAY",
        device=device,
    )[0]

    return SubjectData(
        gender=gender,

        camera_config=camera_config,
        camera_transform=camera_transform,

        blending_param=blending_param,

        model_data=model_data,

        garment_info=garment_info,
        garment=garment,

        video=video,
        mask=mask,

        fps=fps,
    )
