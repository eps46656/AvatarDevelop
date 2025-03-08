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


INT = torch.int32
FLOAT = torch.float32
DEVICE = torch.device("cuda")


ORIGIN = torch.tensor([[0], [0], [0]], dtype=FLOAT)
X_AXIS = torch.tensor([[1], [0], [0]], dtype=FLOAT)
Y_AXIS = torch.tensor([[0], [1], [0]], dtype=FLOAT)
Z_AXIS = torch.tensor([[0], [0], [1]], dtype=FLOAT)

SMPLX_FT = +Z_AXIS
SMPLX_BK = -Z_AXIS

SMPLX_LT = +X_AXIS
SMPLX_RT = -X_AXIS

SMPLX_UP = +Y_AXIS
SMPLX_DW = -Y_AXIS


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
                ORIGIN.flatten().tolist() + [1],  # position trnasform
                SMPLX_FT.flatten().tolist() + [0],  # direction transform
                SMPLX_RT.flatten().tolist() + [0],  # direction transform
                SMPLX_UP.flatten().tolist() + [0],  # direction transform
            ], dtype=FLOAT),

            # world coordinate
            torch.tensor([
                ORIGIN.flatten().tolist() + [1],  # position trnasform
                X_AXIS.flatten().tolist() + [0],  # direction transform
                Z_AXIS.flatten().tolist() + [0],  # direction transform
                Y_AXIS.flatten().tolist() + [0],  # direction transform
            ], dtype=FLOAT),
        )[0].transpose(0, 1)

    # model_mat = utils.GetRotMat(Y_AXIS, 0*utils.DEG) @ model_mat

    model_mat = model_mat.to(dtype=FLOAT, device=DEVICE)

    with open(DIR / "smplx_param.json") as f:
        pose_params = json.load(f)

    with utils.Timer():
        smplx_builder = smplx.smplx.SMPLX(
            model_path=DIR / "smplx/models/smplx",
            num_betas=10,
            use_pca=False,
            dtype=FLOAT
        )

    with utils.Timer():
        smplx_model = smplx_builder.forward(
            betas=torch.tensor(
                pose_params["shape"], dtype=FLOAT).reshape((1, -1)),
            body_pose=torch.tensor(
                pose_params["body_pose"], dtype=FLOAT).reshape((1, -1, 3)),
            left_hand_pose=torch.tensor(
                pose_params["lhand_pose"], dtype=FLOAT).reshape((1, -1)),
            right_hand_pose=torch.tensor(
                pose_params["rhand_pose"], dtype=FLOAT).reshape((1, -1)),
            jaw_pose=torch.tensor(
                pose_params["jaw_pose"], dtype=FLOAT).reshape((1, 3)),
            expression=torch.tensor(
                pose_params["expr"], dtype=FLOAT).reshape((1, -1)),
            return_full_pose=True,
            return_verts=True,
            return_shaped=True,
        )

    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with open(model_path, "rb") as model_file:
        model_data = pickle.load(model_file, encoding="latin1")

    vertices = torch.tensor(
        smplx_model.vertices, dtype=FLOAT, device=DEVICE).reshape((-1, 3))
    # [V, 3]

    vertices = vertices @ model_mat[:3, :3].transpose(0, 1) + \
        model_mat[:3, 3:].transpose(0, 1)
    # [V, 3]

    faces = torch.tensor(
        model_data["f"], dtype=INT, device=DEVICE).reshape((-1, 3))
    # [F, 3]

    face_textures = torch.tensor(
        model_data["ft"], dtype=INT, device=DEVICE).reshape((-1, 3))
    # [F, 3]

    vertex_textures = torch.tensor(
        model_data["vt"], dtype=FLOAT, device=DEVICE).reshape((-1, 2))
    # [V, 2]

    albedo_map = utils.ReadImage(DIR / "black_male.png", "hwc"
                                 ).to(dtype=FLOAT, device=DEVICE)
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

    print(f"{ORIGIN=}")
    print(f"{X_AXIS=}")
    print(f"{Y_AXIS=}")
    print(f"{Z_AXIS=}")

    view_mat = camera_utils.MakeViewMat(
        origin=utils.Sph2XYZ(radius, theta, phi, Z_AXIS, X_AXIS, Y_AXIS),
        aim=ORIGIN,
        quasi_u_dir=Y_AXIS,
        view_axes="luf",
    ).to(dtype=FLOAT, device=DEVICE)

    print(f"{view_mat=}")

    cameras = pytorch3d.renderer.PerspectiveCameras(
        R=view_mat[:3, :3].transpose(0, 1).unsqueeze(0),
        T=view_mat[:3, 3].unsqueeze(0),
        focal_length=[camera_utils.GetFocalLengthByDiagFoV(
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

    utils.WriteImage(DIR / "output.png", img.reshape(img_shape + (3,)), "hwc")


if __name__ == "__main__":
    main1()
