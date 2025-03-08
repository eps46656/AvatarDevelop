
import smplx.smplx

import pickle

import pathlib
import torch

import utils
import numpy as np

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with open(model_path, "rb") as model_file:
        model_data = pickle.load(model_file, encoding="latin1")

    print(f"{type(model_data)}")

    print(f"{model_data.keys()=}")

    print(f"{model_data["f"]=}")
    print(f"{type(model_data["f"])=}")

    print(f"{model_data["ft"]=}")
    print(f"{type(model_data["ft"])=}")

    print(f"{model_data["vt"]=}")
    print(f"{type(model_data["vt"])=}")


def main2():
    model_path = DIR / "smplx/models/smplx"

    with utils.Timer() as t:
        d = smplx.smplx.SMPLX(
            model_path=model_path,
            dtype=float)

    with utils.Timer() as t:
        smplx_output = d.forward(
            return_full_pose=True,
            return_verts=True,
            return_shaped=True,
        )

    # smplx_output

    # print(smplx_output)

    for key, val in smplx_output.items():
        print(f"{key=}\t\t{type(val)=}")

    print(f"{smplx_output.full_pose.shape=}")
    print(f"{smplx_output.vertices.shape=}")
    print(f"{smplx_output.joints.shape=}")


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


def main3():
    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with open(model_path, "rb") as f:
        model_data = pickle.load(f, encoding="latin1")

    for key, value in model_data.items():
        print(f"{key=}")
        print(f"{type(value)=}")

        if isinstance(value, list):
            print(f"{len(value)}")

        if isinstance(value, np.ndarray):
            print(f"{value.shape=}")

        if isinstance(value, torch.Tensor):
            print(f"{value.shape=}")

        print("")

    kintree_table = model_data["kintree_table"]

    print(kintree_table.shape)

    J = kintree_table.shape[1]

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


if __name__ == "__main__":
    main3()
