
import os

import pymeshlab
import pymeshlab.pmeshlab
import torch

from . import config, mesh_utils, smplx_utils, utils

DEVICE = utils.CUDA_DEVICE


def load_gom_avatar(iter_path: os.PathLike):
    model = torch.load(iter_path)["network"]

    faces = model["faces"]  # [F, 3]
    vertices = model["vertices"].T  # [V, 3]
    lbs_weight = model["lbs_weights"].T  # [V, 25]


def main1():
    path = config.DIR / "GoMAvatar/log/snapshot_f3c/checkpoints/iter_200000.pt"

    model = torch.load(path, map_location=utils.CPU_DEVICE)["network"]

    faces = model["faces"]  # [F, 3]
    vertices = model["vertices"].T  # [V, 3]
    lbs_weight = model["lbs_weights"].T  # [V, 25]

    F, V = -1, -2

    F, V = utils.check_shapes(
        faces, (F, 3),
        vertices, (V, 3),
        lbs_weight, (V, 25),
    )

    mesh_data = mesh_utils.MeshData(
        mesh_utils.MeshGraph.from_faces(
            V,
            faces.to(DEVICE, torch.float64),
            DEVICE,
        ),

        vertices.to(DEVICE, torch.float64),
    )

    new_mesh_data = mesh_data.remesh(10 * 1e-3, 5)

    mesh_data.show()
    new_mesh_data.show()


if __name__ == "__main__":
    main1()
