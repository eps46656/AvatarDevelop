import json
import pathlib
import pickle

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch

import config
import smplx.smplx
import utils
import camera_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


DEVICE = torch.device("cuda")


SMPLX_FT = +utils.Z_AXIS
SMPLX_BK = -utils.Z_AXIS

SMPLX_LT = +utils.X_AXIS
SMPLX_RT = -utils.X_AXIS

SMPLX_UP = +utils.Y_AXIS
SMPLX_DW = -utils.Y_AXIS


class AlbedoMeshShader(pytorch3d.renderer.mesh.shader.ShaderBase):
    def forward(self, fragments: pytorch3d.renderer.mesh.rasterizer.Fragments, meshes: pytorch3d.structures.Meshes, **kwargs):
        texels = meshes.sample_textures(fragments)

        print(f"{type(texels)=}")
        print(f"{texels.shape=}")

        return texels


def main1():
    model_mat: torch.Tensor = \
        torch.linalg.lstsq(
            # model coordinate
            torch.tensor([
                utils.ORIGIN.tolist() + [1],  # position trnasform
                SMPLX_FT.tolist() + [0],  # direction transform
                SMPLX_RT.tolist() + [0],  # direction transform
                SMPLX_UP.tolist() + [0],  # direction transform
            ], dtype=utils.FLOAT),

            # world coordinate
            torch.tensor([
                utils.ORIGIN.tolist() + [1],  # position trnasform
                utils.X_AXIS.tolist() + [0],  # direction transform
                utils.Z_AXIS.tolist() + [0],  # direction transform
                utils.Y_AXIS.tolist() + [0],  # direction transform
            ], dtype=utils.FLOAT),
        )[0].transpose(0, 1)

    # model_mat = utils.GetRotMat(Y_AXIS, 0*utils.DEG) @ model_mat

    model_mat = model_mat.to(dtype=utils.FLOAT, device=DEVICE)

    with open(DIR / "smplx_param.json") as f:
        pose_params = json.load(f)

    with utils.Timer():
        smplx_builder = smplx.smplx.SMPLX(
            model_path=DIR / "smplx/models/smplx",
            num_betas=10,
            use_pca=False,
            dtype=utils.FLOAT
        )

    with utils.Timer():
        smplx_model = smplx_builder.forward(
            betas=torch.tensor(
                pose_params["shape"], dtype=utils.FLOAT).reshape((1, -1)),
            body_pose=torch.tensor(
                pose_params["body_pose"], dtype=utils.FLOAT).reshape((1, -1, 3)),
            left_hand_pose=torch.tensor(
                pose_params["lhand_pose"], dtype=utils.FLOAT).reshape((1, -1)),
            right_hand_pose=torch.tensor(
                pose_params["rhand_pose"], dtype=utils.FLOAT).reshape((1, -1)),
            jaw_pose=torch.tensor(
                pose_params["jaw_pose"], dtype=utils.FLOAT).reshape((1, 3)),
            expression=torch.tensor(
                pose_params["expr"], dtype=utils.FLOAT).reshape((1, -1)),
            return_full_pose=True,
            return_verts=True,
            return_shaped=True,
        )

    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with open(model_path, "rb") as model_file:
        model_data = pickle.load(model_file, encoding="latin1")

    vertices = torch.tensor(
        smplx_model.vertices, dtype=utils.FLOAT, device=DEVICE).reshape((-1, 3))
    # [V, 3]

    vertices = vertices @ model_mat[:3, :3].transpose(0, 1) + \
        model_mat[:3, 3:].transpose(0, 1)
    # [V, 3]

    faces = torch.tensor(
        model_data["f"], dtype=utils.INT, device=DEVICE).reshape((-1, 3))
    # [F, 3]

    face_textures = torch.tensor(
        model_data["ft"], dtype=utils.INT, device=DEVICE).reshape((-1, 3))
    # [F, 3]

    vertex_textures = torch.tensor(
        model_data["vt"], dtype=utils.FLOAT, device=DEVICE).reshape((-1, 2))
    # [V, 2]

    albedo_map = utils.read_image(DIR / "black_male.png", "hwc"
                                  ).to(dtype=utils.FLOAT, device=DEVICE)
    print(f"{albedo_map.shape=}")

    albedo_texture = pytorch3d.renderer.TexturesUV(
        maps=albedo_map.unsqueeze(0),
        verts_uvs=vertex_textures.unsqueeze(0),
        faces_uvs=face_textures.unsqueeze(0),
        padding_mode="reflection",
        sampling_mode="bilinear",
    )

    mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=albedo_texture,
    )

    img_shape = (720, 1280)

    radius = 10
    theta = 60 * utils.DEG
    phi = (180 + 45) * utils.DEG

    view_mat = camera_utils.MakeViewMat(
        origin=utils.Sph2XYZ(radius, theta, phi,
                             utils.Z_AXIS, utils.X_AXIS, utils.Y_AXIS),
        aim=utils.ORIGIN,
        quasi_u_dir=utils.Y_AXIS,
        view_axes="luf",
        dtype=utils.FLOAT,
        device=DEVICE,
    ).to(dtype=utils.FLOAT, device=DEVICE)

    print(f"{view_mat=}")

    cameras = pytorch3d.renderer.PerspectiveCameras(
        R=view_mat[:3, :3].transpose(0, 1).unsqueeze(0),
        T=view_mat[:3, 3].unsqueeze(0),
        focal_length=[camera_utils.make_focal_length_by_fov_diag(
            img_shape, 45 * utils.DEG)],
        principal_point=[(img_shape[1] / 2, img_shape[0] / 2)],
        in_ndc=False,
        device=DEVICE,
    )

    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=img_shape,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=AlbedoMeshShader(),
    )

    img = renderer(mesh, image_size=img_shape)

    print(f"{img.shape=}")

    utils.write_image(DIR / "output.png", img.reshape(img_shape + (3,)), "hwc")


if __name__ == "__main__":
    main1()
