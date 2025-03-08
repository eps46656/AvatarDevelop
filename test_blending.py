
import enum

import typing
import pathlib
import torch
import pickle

import utils
import blending_utils

from kin_tree import KinTree

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")
INT = torch.int32
FLOAT = torch.float32


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


class SMPL_Skeleton:
    joint_names = [
        "pelvis",  # 0
        "left_hip",  # 1
        "right_hip",  # 2
        "spine1",  # 3
        "left_knee",  # 4
        "right_knee",  # 5
        "spine2",  # 6
        "left_ankle",  # 7
        "right_ankle",  # 8
        "spine3",  # 9
        "left_foot",  # 10
        "right_foot",  # 11
        "neck",  # 12
        "left_collar",  # 13
        "right_collar",  # 14
        "head",  # 15
        "left_shoulder",  # 16
        "right_shoulder",  # 17
        "left_elbow",  # 18
        "right_elbow",  # 19
        "left_wrist",  # 20
        "right_wrist",  # 21
        "left_hand",  # 22
        "right_hand",  # 23
    ]

    joint_idxes = {
        joint_name: joint_idx
        for joint_idx, joint_name in enumerate(joint_names)
    }

    parents = {
        joint_idxes["pelvis"]: -1,
        joint_idxes["left_hip"]: joint_idxes["pelvis"],
        joint_idxes["right_hip"]: joint_idxes["pelvis"],
    }

    @staticmethod
    def JointIdxToJointName(joint_idx: int):
        return SMPL_Skeleton.joint_names[joint_idx]

    @staticmethod
    def JointNameToJointIdx(joint_name: str):
        return SMPL_Skeleton.joint_idxes[joint_name]

    @staticmethod
    def GetJointParent(joint_idx: int):
        pass


class SMPLX_JOINT_ENUM(enum.IntEnum):
    PELVIS = 0,
    LEFT_HIP = 1,
    RIGHT_HIP = 2,
    SPINE1 = 3,
    LEFT_KNEE = 4,
    RIGHT_KNEE = 5,
    SPINE2 = 6,
    LEFT_ANKLE = 7,
    RIGHT_ANKLE = 8,
    SPINE3 = 9,
    LEFT_FOOT = 10,
    RIGHT_FOOT = 11,
    NECK = 12,
    LEFT_COLLAR = 13,
    RIGHT_COLLAR = 14,
    HEAD = 15,
    LEFT_SHOULDER = 16,
    RIGHT_SHOULDER = 17,
    LEFT_ELBOW = 18,
    RIGHT_ELBOW = 19,
    LEFT_WRIST = 20,
    RIGHT_WRIST = 21,
    JAW = 22,
    LEFT_EYE_SMPLHF = 23,
    RIGHT_EYE_SMPLHF = 24,
    LEFT_INDEX1 = 25,
    LEFT_INDEX2 = 26,
    LEFT_INDEX3 = 27,
    LEFT_MIDDLE1 = 28,
    LEFT_MIDDLE2 = 29,
    LEFT_MIDDLE3 = 30,
    LEFT_PINKY1 = 31,
    LEFT_PINKY2 = 32,
    LEFT_PINKY3 = 33,
    LEFT_RING1 = 34,
    LEFT_RING2 = 35,
    LEFT_RING3 = 36,
    LEFT_THUMB1 = 37,
    LEFT_THUMB2 = 38,
    LEFT_THUMB3 = 39,
    RIGHT_INDEX1 = 40,
    RIGHT_INDEX2 = 41,
    RIGHT_INDEX3 = 42,
    RIGHT_MIDDLE1 = 43,
    RIGHT_MIDDLE2 = 44,
    RIGHT_MIDDLE3 = 45,
    RIGHT_PINKY1 = 46,
    RIGHT_PINKY2 = 47,
    RIGHT_PINKY3 = 48,
    RIGHT_RING1 = 49,
    RIGHT_RING2 = 50,
    RIGHT_RING3 = 51,
    RIGHT_THUMB1 = 52,
    RIGHT_THUMB2 = 53,
    RIGHT_THUMB3 = 54,
    NOSE = 55,
    RIGHT_EYE = 56,
    LEFT_EYE = 57,
    RIGHT_EAR = 58,
    LEFT_EAR = 59,
    LEFT_BIG_TOE = 60,
    LEFT_SMALL_TOE = 61,
    LEFT_HEEL = 62,
    RIGHT_BIG_TOE = 63,
    RIGHT_SMALL_TOE = 64,
    RIGHT_HEEL = 65,
    LEFT_THUMB = 66,
    LEFT_INDEX = 67,
    LEFT_MIDDLE = 68,
    LEFT_RING = 69,
    LEFT_PINKY = 70,
    RIGHT_THUMB = 71,
    RIGHT_INDEX = 72,
    RIGHT_MIDDLE = 73,
    RIGHT_RING = 74,
    RIGHT_PINKY = 75,
    RIGHT_EYE_BROW1 = 76,
    RIGHT_EYE_BROW2 = 77,
    RIGHT_EYE_BROW3 = 78,
    RIGHT_EYE_BROW4 = 79,
    RIGHT_EYE_BROW5 = 80,
    LEFT_EYE_BROW5 = 81,
    LEFT_EYE_BROW4 = 82,
    LEFT_EYE_BROW3 = 83,
    LEFT_EYE_BROW2 = 84,
    LEFT_EYE_BROW1 = 85,
    NOSE1 = 86,
    NOSE2 = 87,
    NOSE3 = 88,
    NOSE4 = 89,
    RIGHT_NOSE_2 = 90,
    RIGHT_NOSE_1 = 91,
    NOSE_MIDDLE = 92,
    LEFT_NOSE_1 = 93,
    LEFT_NOSE_2 = 94,
    RIGHT_EYE1 = 95,
    RIGHT_EYE2 = 96,
    RIGHT_EYE3 = 97,
    RIGHT_EYE4 = 98,
    RIGHT_EYE5 = 99,
    RIGHT_EYE6 = 100,
    LEFT_EYE4 = 101,
    LEFT_EYE3 = 102,
    LEFT_EYE2 = 103,
    LEFT_EYE1 = 104,
    LEFT_EYE6 = 105,
    LEFT_EYE5 = 106,
    RIGHT_MOUTH_1 = 107,
    RIGHT_MOUTH_2 = 108,
    RIGHT_MOUTH_3 = 109,
    MOUTH_TOP = 110,
    LEFT_MOUTH_3 = 111,
    LEFT_MOUTH_2 = 112,
    LEFT_MOUTH_1 = 113,
    LEFT_MOUTH_5 = 114,  # 59 in OpenPose output
    LEFT_MOUTH_4 = 115,  # 58 in OpenPose output
    MOUTH_BOTTOM = 116,
    RIGHT_MOUTH_4 = 117,
    RIGHT_MOUTH_5 = 118,
    RIGHT_LIP_1 = 119,
    RIGHT_LIP_2 = 120,
    LIP_TOP = 121,
    LEFT_LIP_2 = 122,
    LEFT_LIP_1 = 123,
    LEFT_LIP_3 = 124,
    LIP_BOTTOM = 125,
    RIGHT_LIP_3 = 126,
    # Face contour
    RIGHT_CONTOUR_1 = 125,
    RIGHT_CONTOUR_2 = 126,
    RIGHT_CONTOUR_3 = 127,
    RIGHT_CONTOUR_4 = 128,
    RIGHT_CONTOUR_5 = 129,
    RIGHT_CONTOUR_6 = 130,
    RIGHT_CONTOUR_7 = 131,
    RIGHT_CONTOUR_8 = 132,
    CONTOUR_MIDDLE = 133,
    LEFT_CONTOUR_8 = 134,
    LEFT_CONTOUR_7 = 135,
    LEFT_CONTOUR_6 = 136,
    LEFT_CONTOUR_5 = 137,
    LEFT_CONTOUR_4 = 138,
    LEFT_CONTOUR_3 = 139,
    LEFT_CONTOUR_2 = 140,
    LEFT_CONTOUR_1 = 141,


class SMPLX_Loader:
    def __init__(self,
                 model_path,
                 device: torch.device,
                 ):
        self.device = device

        with open(model_path, "rb") as f:
            self.model_data = pickle.load(f, encoding="latin1")

        kintree_table = self.model_data["kintree_table"]

        kin_tree_links = [(int(kintree_table[0, j]), int(kintree_table[1, j]))
                          for j in range(kintree_table.shape[1])]

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

    def GetVertices(self,
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

    def GetVertexTextures(self) -> torch.Tensor:  # [V, 2]
        return self.vertex_textures

    def GetFaces(self) -> torch.Tensor:  # [F, 3]
        return self.faces

    def GetFaceTextures(self) -> torch.Tensor:  # [F, 3]
        return self.face_textures


def main1():
    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with open(model_path, "rb") as f:
        model_data = pickle.load(f, encoding="latin1")

    kintree_table = model_data["kintree_table"]

    print(kintree_table.shape)

    J = kintree_table.shape[1]

    '''
    for j in range(J):
        a = int(kintree_table[0, j])
        b = int(kintree_table[1, j])

        print(f"{a}, {b}")

        if a < 0 or J <= a:
            a_name = "none"
        else:
            a_name = SMPL_Skeleton.JointIdxToJointName(a)

        if b < 0 or J <= b:
            b_name = "none"
        else:
            b_name = SMPL_Skeleton.JointIdxToJointName(b)

        print(f"{a_name}, {b_name}")
    '''

    kin_tree_links = [(int(kintree_table[0, j]), int(kintree_table[1, j]))
                      for j in range(J)]

    kin_tree = KinTree.FromLinks(kin_tree_links, 2**32-1)
    # joints_cnt = J

    vertices = torch.from_numpy(model_data["v_template"]).to(
        dtype=FLOAT, device=DEVICE)
    # [V, 3]

    lbs_weights = torch.from_numpy(model_data["weights"]).to(
        dtype=FLOAT, device=DEVICE)
    # [V, J]

    joint_regressor = torch.from_numpy(model_data["J_regressor"]).to(
        dtype=FLOAT, device=DEVICE)
    # [J, V]

    shape_dirs = torch.from_numpy(model_data["shapedirs"]).to(
        dtype=FLOAT, device=DEVICE)
    # [V, 3, B]

    ###

    J = kin_tree.joints_cnt
    V = vertices.shape[0]

    ###

    binding_pose_rs = torch.eye(3, dtype=FLOAT, device=DEVICE) \
        .reshape(1, 3, 3).expand((J, 3, 3))

    binding_pose_ts = torch.zeros((J, 3), dtype=FLOAT, device=DEVICE)

    pose_rs = utils.GetRotMat(utils.RandUnit((J, 3)),
                              torch.rand((J,)))
    # [J, 3, 3]

    pose_ts = torch.rand((J, 3))

    print(f"{pose_rs.shape=}")

    v = blending_utils.LBS(
        kin_tree,
        vertices,
        lbs_weights,
        binding_pose_rs,
        binding_pose_ts,
        pose_rs,
        pose_ts
    )

    print(f"{v.shape}")


def main3():
    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    smplx_loader = SMPLX_Loader(model_path, DEVICE)

    B = smplx_loader.GetShapesCnt()
    J = smplx_loader.GetJointsCnt()

    poses = utils.RandUnit((J, 3)) * torch.rand((J, 1))
    # [J, 3]

    joints_ts, vertices = smplx_loader.GetVertices(
        torch.zeros((B,), dtype=FLOAT, device=DEVICE),
        poses
    )

    print(f"{vertices.shape}")


if __name__ == "__main__":
    main3()
