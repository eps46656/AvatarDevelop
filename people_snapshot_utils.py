import dataclasses
import typing

import h5py
import torch
import tqdm
from beartype import beartype

from . import (camera_utils, config, smplx_utils, transform_utils, utils,
               vision_utils)


@beartype
@dataclasses.dataclass
class SubjectData:
    fps: float

    video: torch.Tensor  # [T, C, H, W]
    mask: torch.Tensor  # [T, 1, H, W]

    camera_config: camera_utils.CameraConfig
    camera_transform: transform_utils.ObjectTransform

    model_data: smplx_utils.ModelData

    blending_param: smplx_utils.BlendingParam

    tex: torch.Tensor  # [C, H, W]


@beartype
def _read_mask(
    subject_dir: utils.PathLike,
    fps: float,
    device: torch.device,
) -> torch.Tensor:  # [T, 1, H, W]
    subject_dir = utils.to_pathlib_path(subject_dir)

    assert subject_dir.is_dir()

    mask_video_path = subject_dir / "masks.avi"

    def read_and_return():
        return vision_utils.read_video_mask(
            mask_video_path, dtype=torch.float32, device=device)[0]

    if mask_video_path.exists():
        return read_and_return()

    f = h5py.File(subject_dir / "masks.hdf5")

    masks = f["masks"]

    T, H, W = masks.shape

    with vision_utils.VideoWriter(
        mask_video_path,
        height=H,
        width=W,
        color_type="RGB",
        fps=fps,
    ) as video_writer:
        for i in tqdm.tqdm(range(T)):
            video_writer.write(torch.from_numpy(
                masks[i] * 255).to(torch.uint8).expand(3, H, W))

    return read_and_return()


@beartype
def _read_camera(
    subject_dir: utils.PathLike,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> tuple[
    camera_utils.CameraConfig,
    transform_utils.ObjectTransform,  # camera <-> world
]:
    subject_dir = utils.to_pathlib_path(subject_dir)

    assert subject_dir.is_dir()

    camera = utils.read_pickle(subject_dir / "camera.pkl")

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

    camera_transform = transform_utils.ObjectTransform.from_matching(
        "RDF", device=device)

    camera_config = camera_utils.CameraConfig.from_slope_udlr(
        slope_u=cy / fy,
        slope_d=(img_h - cy) / fy,
        slope_l=cx / fx,
        slope_r=(img_w - cx) / fx,
        depth_near=utils.DEPTH_NEAR,
        depth_far=utils.DEPTH_FAR,
        img_h=img_h,
        img_w=img_w,
    )

    return camera_config, camera_transform


@beartype
def _read_blending_param(
    subject_dir: utils.PathLike,
    model_data: smplx_utils.ModelData,
    frames_cnt: int,
    dtype: torch.dtype,
    device: torch.device,
) -> smplx_utils.BlendingParam:
    subject_dir = utils.to_pathlib_path(subject_dir)

    assert subject_dir.is_dir()

    body_shapes_cnt = model_data.body_shape_vert_dir.shape[-1]
    poses_cnt = model_data.body_joints_cnt
    # global_rot + smplx_utils.BODY_POSES_CNT

    d = h5py.File(subject_dir / "reconstructed_poses.hdf5")

    body_shapes = torch.from_numpy(d["betas"][...])
    # [?]

    global_transl = torch.from_numpy(d["trans"][...])
    # [T, 3]

    poses = torch.from_numpy(d["pose"][...])
    # [T, ? * 3]

    # ---

    T, cur_body_shapes_cnt, cur_flatten_poses_cnt = -1, -2, -3

    T, cur_body_shapes_cnt, cur_flatten_poses_cnt = utils.check_shapes(
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
        body_shapes = body_shapes[:, :body_shapes_cnt]

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
        poses = poses[:, :poses_cnt * 3]

    poses = poses.reshape(T, poses_cnt, 3)

    global_rot = poses[:, 0, :]
    # [T, 3]

    body_poses = poses[:, 1:, :]
    # [T, smplx_model_data.body_joints_cnt, 3]

    # ---

    dd = (device, dtype)

    return smplx_utils.BlendingParam(
        body_shape=body_shapes[:frames_cnt].to(*dd),
        global_transl=global_transl[:frames_cnt].to(*dd),
        global_rot=global_rot[:frames_cnt].to(*dd),
        body_pose=body_poses[:frames_cnt].to(*dd),
    )


@beartype
def _read_tex(
    subject_dir: utils.PathLike,
    device: torch.device,
):
    subject_dir = utils.to_pathlib_path(subject_dir)

    assert subject_dir.is_dir()

    return vision_utils.read_image(
        subject_dir / f"tex-{subject_dir.name}.jpg", "RGB"
    ).image.to(device)


@beartype
def read_subject(
    subject_dir: utils.PathLike,
    model_data: typing.Optional[smplx_utils.ModelData],
    dtype: torch.dtype,
    device: torch.device,
):
    subject_dir = utils.to_pathlib_path(subject_dir)

    assert subject_dir.is_dir()

    subject_name = subject_dir.name

    if model_data is None:
        assert "male" in subject_name or "female" in subject_name

        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH
            if "female" in subject_name else
            config.SMPL_MALE_MODEL_PATH,

            model_config=smplx_utils.smpl_model_config,

            dtype=dtype,
            device=device,
        )

    video, fps = vision_utils.read_video(
        subject_dir / f"{subject_name}.mp4",
        "RGB",
        device=device,
    )
    # [T, C, H, W]

    mask = _read_mask(subject_dir, fps, device)
    # [T, 1, H, W]

    img_h, img_w = video.shape[2:]

    camera_config, camera_transform = \
        _read_camera(subject_dir, img_h, img_w, device)

    blending_param = _read_blending_param(
        subject_dir, model_data, video.shape[0], dtype, device)

    tex = _read_tex(subject_dir, device)

    ret = SubjectData(
        fps=fps,
        video=video,
        mask=mask,
        camera_config=camera_config,
        camera_transform=camera_transform,
        model_data=model_data,
        blending_param=blending_param,
        tex=tex,
    )

    return ret
