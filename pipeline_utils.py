import copy
import dataclasses
import enum
import itertools
import math
import pathlib
import time
import typing

import torch
from beartype import beartype

from . import (config, gom_avatar_utils, mesh_layer_utils, mesh_utils,
               people_snapshot_utils, sdf_utils, smplx_utils, training_utils,
               utils)


class LoadEnum(enum.IntEnum):
    SKIP_LOAD = -2
    LOAD_LATEST = -1


@dataclasses.dataclass
class SMPLXModelInfo:
    model_data_path: pathlib.Path
    model_config: smplx_utils.ModelConfig


smpl_male_model_info = SMPLXModelInfo(
    model_data_path=config.SMPL_MALE_MODEL_PATH,
    model_config=smplx_utils.smpl_model_config,
)

smpl_female_model_info = SMPLXModelInfo(
    model_data_path=config.SMPL_FEMALE_MODEL_PATH,
    model_config=smplx_utils.smpl_model_config,
)

smplx_male_model_info = SMPLXModelInfo(
    model_data_path=config.SMPLX_MALE_MODEL_PATH,
    model_config=smplx_utils.smplx_model_config,
)

smplx_female_model_info = SMPLXModelInfo(
    model_data_path=config.SMPLX_FEMALE_MODEL_PATH,
    model_config=smplx_utils.smplx_model_config,
)


@beartype
def load_smplx_model_data(
    model_info: SMPLXModelInfo,
    dtype: torch.dtype,
    device: torch.device,
) -> smplx_utils.ModelData:
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=model_info.model_data_path,
        model_config=model_info.model_config,
        dtype=dtype,
        device=device,
    )

    return model_data


@beartype
def load_sdf_module_trainer(
    sdf_module_dir: pathlib.Path,
    mesh_data: typing.Optional[mesh_utils.MeshData],
    dtype: torch.dtype,
    device: torch.device,
) -> training_utils.Trainer:
    batch_size = 64
    batches_cnt = 64

    alpha_signed_dist = (lambda epoch:  1e6)
    alpha_eikonal = (lambda epoch: 1.0)

    # ---

    dataset = None if mesh_data is None else sdf_utils.Dataset(
        mesh_data=mesh_data.to(device),

        std=50e-3,

        shape=torch.Size((batch_size * batches_cnt,)),
    )

    # ---

    module = sdf_utils.Module(
        range_min=(-2.0, -2.0, -2.0),
        range_max=(+2.0, +2.0, +2.0),
        dtype=dtype,
        device=device,
    ).train()

    # ---

    trainer_core = sdf_utils.TrainerCore(
        config=sdf_utils.TrainerCoreConfig(
            proj_dir=sdf_module_dir,
            device=device,

            batch_size=batch_size,

            lr=(lambda epoch: 1e-3 * (0.1 ** (epoch / 256))),
            betas=(0.9, 0.99),

            alpha_signed_dist=alpha_signed_dist,
            alpha_eikonal=alpha_eikonal,
        ),
        module=module,
        dataset=dataset,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=sdf_module_dir,
        trainer_core=trainer_core,
    )

    return trainer


@beartype
def load_sdf_module(
    sdf_module_dir: pathlib.Path,

    ckpt_id: int | LoadEnum,
    # SKIP_LOAD
    # LOAD_LATEST
    # otherwise: load specific checkpoint

    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    training_utils.Trainer,
    sdf_utils.Module,
]:
    sdf_trainer = load_sdf_module_trainer(
        sdf_module_dir=sdf_module_dir,
        mesh_data=None,
        dtype=dtype,
        device=device,
    )

    match ckpt_id:
        case LoadEnum.SKIP_LOAD:
            pass
        case LoadEnum.LOAD_LATEST:
            sdf_trainer.load_latest()
        case _:
            sdf_trainer.load(int(ckpt_id))

    sdf_module: sdf_utils.Module = sdf_trainer.trainer_core.module

    return sdf_trainer, sdf_module


@beartype
def load_mesh_layer_trainer(
    mesh_layer_dir: pathlib.Path,

    sdf_module: typing.Callable[[torch.Tensor], torch.Tensor],

    mesh_graph: mesh_utils.MeshGraph,
    vert_pos: torch.Tensor,  # [..., D]

    signed_dist_lb: torch.Tensor,  # [...]
    signed_dist_rb: torch.Tensor,  # [...]

    dtype: torch.dtype,
    device: torch.device,
) -> training_utils.Trainer:
    mesh_layer_trainer = training_utils.Trainer(
        proj_dir=mesh_layer_dir,
        trainer_core=mesh_layer_utils.TrainerCore(
            mesh_layer_utils.TrainerCoreConfig(
                proj_dir=mesh_layer_dir,
                device=device,

                batches_cnt=16,

                lr=(lambda epoch: 1e-3 * (0.95 ** epoch)),
                betas=(0.5, 0.5),

                signed_dist_lb=signed_dist_lb,
                signed_dist_rb=signed_dist_rb,

                vert_grad_norm_threshold=1e-3,

                alpha_pe=(lambda epoch: 1.0),
                alpha_lap_diff=(lambda epoch: 1000.0),
                alpha_edge_diff=(lambda epoch: 10.0),
            ),
            sdf_module=sdf_module,
            mesh_graph=mesh_graph,
            vert_pos=vert_pos,
        )
    )

    return mesh_layer_trainer
