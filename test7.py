import json
import pathlib
import pickle

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch

import dataclasses
import config
import smplx.smplx
import utils
import camera_utils
import blending_utils
from kin_tree import KinTree

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


@dataclasses.dataclass
class SMPLXModel:
    joints: torch.Tensor
    vertices: torch.Tensor


def SMPLXForward(
    kin_tree: KinTree,
    vertices: torch.Tensor,  # [..., V, 3]
    shape_dirs: None | torch.Tensor,  # [..., V, 3, B]
    joint_regressor: torch.Tensor,  # [..., J, V],
    lbs_weights: torch.Tensor,  # [..., V, J]
    shapes: None | torch.Tensor,  # [..., B]
    root_transl: None | torch.Tensor,  # [..., 3],
    poses: torch.Tensor,  # [..., J, 3],
    device: torch.device
):
    assert 2 <= len(vertices)
    assert 2 <= len(joint_regressor)

    J = kin_tree.joints_cnt
    V = vertices.shape[0]

    assert vertices.shape[-2:] == (V, 3)

    if shape_dirs is not None:
        assert 3 <= len(shape_dirs)
        assert shape_dirs.shape[-3] == V
        assert shape_dirs.shape[-2] == 3

    assert joint_regressor.shape[-2:] == (J, V)
    assert lbs_weights.shape[-2:] == (V, J)

    if shapes is not None:
        assert 1 <= len(shapes.shape)

    if root_transl is not None:
        assert 1 <= len(root_transl.shape)
        assert root_transl.shape[-1] == 3

    assert 2 <= len(poses.shape)
    assert poses.shape[-2] == (J, 3)

    vs = vertices

    if shape_dirs is not None and shapes is not None:
        B = shape_dirs.shape[-1]

        assert shape_dirs.shape[-3:] == (V, 3, B)
        assert shapes.shape[-1] == B

        vs = vs + torch.einsum("vxb,...b->...vx", shape_dirs, shapes)

    binding_pose_rs = torch.eye(3, dtype=FLOAT, device=device).unsqueeze(0) \
        .expand((J, 3, 3))

    joints = torch.einsum(
        "jv,...vx->...jx", joint_regressor, vs)
    # [..., J, 3]

    binding_pose_ts = torch.empty_like(joints)
    # [..., J, 3]

    binding_pose_ts[..., kin_tree.root, :] = \
        0 if root_transl is None else root_transl

    for u in kin_tree.joints_tp:
        p = kin_tree.parents[u]

        if p == -1:
            continue

        binding_pose_ts[..., u, :] = joints[..., u, :] - joints[..., p, :]

    vs = blending_utils.LBS(
        kin_tree,
        vs,
        lbs_weights,
        binding_pose_rs,
        binding_pose_ts,
        utils.GetRotMat(poses),
        binding_pose_ts,
    )

    return SMPLXModel(
        joints=joints,
        vertices=vs,
    )


class SMPLXBuilder:
    def __init__(self,
                 model_path,
                 device: torch.device,
                 ):
        self.device = device

        with open(model_path, "rb") as f:
            self.model_data = pickle.load(f, encoding="latin1")

        kintree_table = self.model_data["kintree_table"]

        kin_tree_links = [(int(kintree_table[0, j]), int(kintree_table[1, j]))
                          for j in range(J)]

        self.kin_tree = KinTree.FromLinks(kin_tree_links, 2**32-1)
        # joints_cnt = J

        # [V, 3]

        self.lbs_weights = torch.from_numpy(self.model_data["weights"]) \
            .to(dtype=FLOAT, device=self.device)
        # [V, J]

        self.joint_regressor = torch.from_numpy(
            self.model_data["J_regressor"]) \
            .to(dtype=FLOAT, device=self.device)
        # [J, V]

        self.shape_dirs = torch.from_numpy(self.model_data["shapedirs"]) \
            .to(dtype=FLOAT, device=self.device)
        # [V, 3, B]

        self.vertices = torch.from_numpy(self.model_data["v_template"]) \
            .to(dtype=FLOAT, device=self.device)
        # [V, 3]

        self.vertex_textures = torch.from_numpy(self.model_data["vt"]) \
            .to(dtype=FLOAT, device=self.device)

        self.faces = torch.from_numpy(self.model_data["f"]) \
            .to(dtype=INT, device=self.device)

        self.face_textures = torch.from_numpy(self.model_data["ft"]) \
            .to(dtype=INT, device=self.device)

        self.face_textures = torch.from_numpy(self.model_data["ft"]) \
            .to(dtype=INT, device=self.device)

        B = self.shape_dirs.shape[2]
        J = self.kin_tree.joints_cnt
        V = self.vertices.shape[0]

        assert self.kin_tree.joints_cnt == J
        assert self.vertices.shape == (V, 3)
        assert self.lbs_weights.shape == (V, J)
        assert self.joint_regressor.shape == (J, V)
        assert self.shape_dirs.shape == (V, 3, B)

    def GetShapesCnt(self):
        return self.shape_dirs.shape[2]

    def GetJointsCnt(self):
        return self.kin_tree.joints_cnt

    def GetVerticesCnt(self):
        return self.vertices.shape[0]

    def GetFacesCnt(self):
        return 0
        # return self.faces

    def forward(self,
                shape: torch.Tensor,  # [..., B]
                pose: torch.Tensor,  # [..., J, 3]
                ):
        B = self.shape_dirs.shape[2]
        J = self.kin_tree.joints_cnt

        assert shape.shape[-1] == B
        assert pose.shape[-2:] == (J, 3)

        vs = self.vertices + \
            torch.einsum("vxb,...b->...vx", self.shape_dirs, shape)
        # [..., V, 3]

        binding_pose_rs = torch.eye(3, dtype=FLOAT, device=self.device
                                    ).unsqueeze(0).expand((J, 3, 3))

        binding_pose_ts = torch.einsum(
            "jv,...vx->...jx", self.joint_regressor, vs)
        # [..., J, 3]

        return binding_pose_ts, blending_utils.LBS(
            self.kin_tree,
            vs,
            self.lbs_weights,
            binding_pose_rs,
            binding_pose_ts,
            utils.GetRotMat(pose),
            binding_pose_ts,
        )

    def GetVertices(self):
        return self.vertices

    def GetVertexTextures(self) -> torch.Tensor:  # [V, 2]
        return self.vertex_textures

    def GetFaces(self) -> torch.Tensor:  # [F, 3]
        return self.faces

    def GetFaceTextures(self) -> torch.Tensor:  # [F, 3]
        return self.face_textures


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
        smplx_builder = SMPLXBuilder(
            model_path=DIR / "smplx/models/smplx",
            device=DEVICE
        )

    with utils.Timer():
        smplx_model = smplx_builder.forward(

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
