
import dataclasses
import math
import os
import pathlib
import typing

import matplotlib.pyplot as plt
import pyvista as pv
import torch
import torchrbf
import tqdm
from beartype import beartype

from . import (config, gart_utils, gaussian_utils, mesh_utils,
               people_snapshot_utils, rbf_utils, smplx_utils,
               utils, video_seg_utils, vision_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-3-casual"


def main1():
    f = open(
        DIR / "SuGaR/output/refined_mesh/south-building/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj"
    )

    vert_pos_faces, \
        tex_vert_pos, \
        vert_nor_faces, \
        vert_pos, \
        tex_vert_pos, \
        vert_nor = mesh_utils.read_obj(f)

    mesh_data = mesh_utils.MeshData(
        mesh_utils.MeshGraph.from_faces(
            len(vert_pos),
            torch.tensor(vert_pos_faces, dtype=torch.int32, device=DEVICE),
            DEVICE,
        ),

        torch.tensor(vert_pos, dtype=torch.float32, device=DEVICE),
    )

    mesh_data.show()


if __name__ == "__main__":
    main1()
