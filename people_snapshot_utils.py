import dataclasses
import os
import pathlib

import h5py
import numpy as np
import torch
import tqdm
from beartype import beartype

from . import camera_utils, smplx_utils, transform_utils, utils


@beartype
@dataclasses.dataclass
class SubjectData:
    video: torch.Tensor  # [T, C, H, W]
    mask: torch.Tensor  # [T, H, W]

    camera_transform: transform_utils.ObjectTransform
    camera_config: camera_utils.CameraConfig

    model_data: smplx_utils.SMPLXModelData

    blending_param: smplx_utils.SMPLXBlendingParam


@beartype
def _ReadMask(
    subject_dir: pathlib.Path,
    fps: int,
    device: torch.device,
):
    mask_video_path = subject_dir / "masks.mp4"

    if mask_video_path.exists():
        return utils.ImageNormalize(
            utils.ReadVideo(mask_video_path)[0].to(dtype=utils.FLOAT, device=device).mean(1))

    f = h5py.File(subject_dir / "masks.hdf5")

    masks = f["masks"]

    T, H, W = masks.shape

    ret = torch.empty((T, 1, H, W), dtype=torch.uint8)

    for i in tqdm.tqdm(range(T)):
        ret[i, 0] = torch.from_numpy(masks[i].astype(np.uint8) * 255)

    utils.WriteVideo(mask_video_path, ret.expand((T, 3, H, W)), fps)

    return utils.ImageNormalize(ret.to(dtype=utils.FLOAT, device=device)) \
        .squeeze(1)


@beartype
def _ReadCamera(
    subject_dir: pathlib.Path,
    img_h: int,
    img_w: int,
) -> tuple[
    transform_utils.ObjectTransform,  # camera <-> world
    camera_utils.CameraConfig,
]:
    camera = utils.ReadPickle(subject_dir / "camera.pkl")

    fx = float(camera['camera_f'][0])
    fy = float(camera['camera_f'][1])
    cx = float(camera['camera_c'][0])
    cy = float(camera['camera_c'][1])
    # in pixels

    """

    fx  0 cx
     0 fy cy
     0  0  1

    """

    camera_transform = transform_utils.ObjectTransform.FromMatching("RDF")

    camera_config = camera_utils.CameraConfig.FromSlopeUDLR(
        slope_u=cy / fy,
        slope_d=(img_h - cy) / fy,
        slope_l=cx / fx,
        slope_r=(img_w - cx) / fx,
        depth_near=utils.DEPTH_NEAR,
        depth_far=utils.DEPTH_FAR,
        img_h=img_h,
        img_w=img_w,
    )

    return camera_transform, camera_config


@beartype
def _ReadSMPLBlendingParam(
    subject_dir: pathlib.Path,
    smplx_model_data: smplx_utils.SMPLXModelData,
    device: torch.device,
) -> smplx_utils.SMPLXBlendingParam:
    body_shapes_cnt = smplx_model_data.body_shape_dirs.shape[-1]
    poses_cnt = smplx_model_data.body_joints_cnt
    # global_rot + smplx_utils.BODY_POSES_CNT

    d = h5py.File(subject_dir / "reconstructed_poses.hdf5")

    body_shapes = torch.tensor(d["betas"], dtype=utils.FLOAT)
    # [?]

    global_transl = torch.tensor(d["trans"], dtype=utils.FLOAT)
    # [T, 3]

    poses = torch.tensor(d["pose"], dtype=utils.FLOAT)
    # [T, ? * 3]

    # ---

    T, cur_body_shapes_cnt, cur_flatten_poses_cnt = -1, -2, -3

    T, cur_body_shapes_cnt, cur_flatten_poses_cnt = utils.CheckShapes(
        body_shapes, (cur_body_shapes_cnt, ),
        global_transl, (T, 3),
        poses, (T, cur_flatten_poses_cnt),
    )

    # ---

    if cur_body_shapes_cnt < body_shapes_cnt:
        body_shapes = torch.nn.functional.pad(
            body_shapes,
            (0, body_shapes_cnt - cur_body_shapes_cnt),
            "constant",
            0.0
        )
    elif body_shapes_cnt < cur_body_shapes_cnt:
        body_shapes = body_shapes[:, :-body_shapes_cnt]

    # [body_shapes_cnt]

    # ---

    if cur_flatten_poses_cnt < poses_cnt * 3:
        poses = torch.nn.functional.pad(
            poses,
            (0, poses_cnt * 3 - cur_flatten_poses_cnt),
            "constant",
            0.0
        )
    elif poses_cnt * 3 < cur_flatten_poses_cnt:
        poses = poses[:, :-poses_cnt * 3]

    poses = poses.reshape((T, poses_cnt, 3))

    global_rot = poses[:, 0, :]
    # [T, 3]

    body_poses = poses[:, 1:, :]
    # [T, smplx_model_data.body_joints_cnt, 3]

    # ---

    return smplx_utils.SMPLXBlendingParam(
        body_shapes=body_shapes.to(dtype=utils.FLOAT, device=device),
        global_transl=global_transl.to(dtype=utils.FLOAT, device=device),
        global_rot=global_rot.to(dtype=utils.FLOAT, device=device),
        body_poses=body_poses.to(dtype=utils.FLOAT, device=device),
    )


@beartype
def ReadSubject(
    subject_dir: os.PathLike,
    model_data_dict: dict[str, smplx_utils.SMPLXModelData],
    device: torch.device,
):
    subject_dir = pathlib.Path(subject_dir)

    assert subject_dir.is_dir()

    assert "female" in model_data_dict
    assert "male" in model_data_dict

    subject_name = subject_dir.name

    subject_video_path = subject_dir / f"{subject_name}.mp4"

    if "female" in subject_name:
        model_data = model_data_dict["female"]
    else:
        model_data = model_data_dict["male"]

    video, audio, d = utils.ReadVideo(subject_video_path)
    # [T, C, H, W]

    print(f"{d=}")

    fps = int(d["video_fps"])

    img_h, img_w = video.shape[2:]

    camera_transform, camera_config = \
        _ReadCamera(subject_dir, img_h, img_w)

    blending_param = _ReadSMPLBlendingParam(subject_dir, model_data, device)

    mask = _ReadMask(subject_dir, fps, device)

    ret = SubjectData(
        video=video,
        camera_transform=camera_transform,
        camera_config=camera_config,
        model_data=model_data,
        blending_param=blending_param,
        mask=mask,
    )

    return ret
