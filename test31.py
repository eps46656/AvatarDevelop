import copy
import itertools
import math
import pathlib
import time
import typing

import einops
import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, config, dataset_utils,
               dw_interp_utils, gaussian_utils, gom_utils,
               people_snapshot_utils, rendering_utils, smplx_utils,
               texture_utils, training_utils, transform_utils, utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

PROJ_DIR = DIR / "train_2025_0404_2"

ALPHA_RGB = 1.0
ALPHA_LAP_SMOOTHING = 10.0
ALPHA_NORMAL_SIM = 1.0
ALPHA_COLOR_DIFF = 1.0

BATCH_SIZE = 4


@beartype
def MyLossFunc(
    avatar_model: avatar_utils.AvatarModel,

    rendered_img: torch.Tensor,  # [..., C, H, W]

    rgb_loss: float | torch.Tensor,
    lap_smoothing_loss: float | torch.Tensor,
    nor_sim_loss: float | torch.Tensor,
    color_diff_loss: float | torch.Tensor,
):
    weighted_rgb_loss = ALPHA_RGB * rgb_loss
    weighted_lap_smoothing_loss = ALPHA_LAP_SMOOTHING * lap_smoothing_loss
    weighted_nor_sim_loss = ALPHA_NORMAL_SIM * nor_sim_loss
    weighted_color_diff_loss = ALPHA_COLOR_DIFF * color_diff_loss

    return weighted_rgb_loss + weighted_lap_smoothing_loss + weighted_nor_sim_loss + weighted_color_diff_loss


@beartype
class MyTrainingCore(training_utils.TrainingCore):
    def __init__(
        self,
        module: gom_utils.Module,
        dataset: gom_utils.Dataset,
        loss_func: typing.Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
    ):
        self.__module = module
        self.__dataset = dataset
        self.__loss_func = loss_func
        self.__optimizer = optimizer
        self.__scheduler = scheduler

    # --

    @property
    def module(self) -> gom_utils.Module:
        return self.__module

    @module.setter
    def module(self, module: gom_utils.Module) -> None:
        self.__module = module

    # --

    @property
    def dataset(self) -> gom_utils.Dataset:
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: gom_utils.Dataset) -> None:
        self.__dataset = dataset

    # --

    @property
    def loss_func(self) -> typing.Callable:
        return self.__loss_func

    @loss_func.setter
    def loss_func(self, loss_func: typing.Callable) -> None:
        self.__loss_func = loss_func

    # --

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.__optimizer = optimizer

    # --

    @property
    def scheduler(self) -> object:
        return self.__scheduler

    @scheduler.setter
    def scheduler(self, scheduler: object) -> None:
        self.__scheduler = scheduler

    # --

    @utils.mem_clear
    def train(self) \
            -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for batch_idxes, sample in tqdm.tqdm(
                dataset_utils.load(self.dataset, batch_size=BATCH_SIZE)):
            batch_idxes: tuple[torch.Tensor, ...]

            sample: gom_utils.Sample = self.dataset[batch_idxes]

            result: gom_utils.ModuleForwardResult = self.module(
                camera_transform=sample.camera_transform,
                camera_config=sample.camera_config,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            loss: torch.Tensor = self.loss_func(**result.__dict__)

            sum_loss += float(loss) * batch_idxes[0].numel()

            self.optimizer.zero_grad()

            loss.backward()

            """
            for name, param in self.module.named_parameters():

                print(f"{name=}")
                print(f"{param=}")
                print(f"{param.grad=}")

                assert param.isfinite().all()
                assert param.grad.isfinite().all()
            """

            self.optimizer.step()

        avg_loss = sum_loss / self.dataset.shape.numel()

        print(f"{avg_loss=}")

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return training_utils.TrainingResult(
            avg_loss=avg_loss
        )

    @utils.mem_clear
    def show_params(self):
        for name, param in self.module.named_parameters():
            print(f"{name}: {param}")

        print(self.optimizer.param_groups)

    @utils.mem_clear
    def output_rgb_video(self):
        self.dataset: gom_utils.Dataset

        img_h, img_w = self.dataset.sample.img.shape[-2:]

        rgb_frames = torch.empty_like(
            self.dataset.sample.img,
            dtype=torch.float16,
            device=DEVICE,
        )
        # [T, C, H, W]

        mesh_frames = torch.empty_like(
            self.dataset.sample.img,
            dtype=torch.float16,
            device=DEVICE,
        )
        # [T, C, H, W]

        T, C, H, W = self.dataset.sample.img.shape

        zeros = torch.zeros((1, img_h, img_w), dtype=utils.FLOAT)
        ones = torch.ones((1, img_h, img_w), dtype=utils.FLOAT)

        batch_shape = self.dataset.shape

        self.module: gom_utils.Module

        with torch.no_grad():
            if True:
                avatar_model: smplx_utils.Model = \
                    self.module.avatar_blender.get_avatar_model()

                avatar_model.mesh_data.show(avatar_model.vert_pos)

            for batch_idxes, sample in tqdm.tqdm(
                    dataset_utils.load(self.dataset, batch_size=BATCH_SIZE)):
                batch_idxes: tuple[torch.Tensor, ...]

                idxes = utils.ravel_idxes(
                    batch_idxes, self.dataset.shape)
                # [K]

                idxes = idxes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    idxes.shape + (C, H, W))
                # [K, C, H, W]

                K = idxes.shape[0]

                sample: gom_utils.Sample = self.dataset[batch_idxes]

                result: gom_utils.ModuleForwardResult = self.module(
                    camera_config=sample.camera_config,
                    camera_transform=sample.camera_transform,
                    img=sample.img,
                    mask=sample.mask,
                    blending_param=sample.blending_param,
                )

                rendered_img = result.rendered_img.reshape((-1, C, H, W))
                # [K, C, H, W]

                rgb_frames.scatter_(
                    0,
                    idxes.to(rgb_frames.device),
                    rendered_img.to(rgb_frames))

                """

                out_frames[idxes[k, c, h, w], c, h, w] =
                    rendered_img[k, c, h, w]

                """

                avatar_model: smplx_utils.Model = result.avatar_model

                """
                for k in range(K):
                    mesh_ras_result = rendering_utils.rasterize_mesh(
                        vert_pos=avatar_model.vert_pos[k],
                        faces=avatar_model.faces,
                        camera_config=sample.camera_config,
                        camera_transform=sample.camera_transform[k],
                        faces_per_pixel=1,
                    )

                    # d["pixel_to_faces"][img_h, img_w, 1]

                    mesh_frames[idxes[k, 0, 0, 0]] = torch.where(
                        (mesh_ras_result["pixel_to_faces"][:, :, 0] == -1)
                        .to(mesh_frames.device),

                        zeros,
                        ones,
                    )

                    del mesh_ras_result

                    lap_smoothing_loss = avatar_model.mesh_data.calc_lap_smoothing_loss(
                        avatar_model.vert_pos[k])
                    # [..., V, 3]

                    lap_loss = lap_smoothing_loss

                    print(f"{lap_loss=}")
                """

        utils.write_video(
            path=PROJ_DIR / f"rgb_{int(time.time())}.mp4",
            video=rgb_frames,
            fps=25,
        )

        """
        utils.write_video(
            path=PROJ_DIR / f"mesh_{int(time.time())}.mp4",
            video=mesh_frames,
            fps=25,
        )
        """

    @utils.mem_clear
    def bake_texture_dw(self, tex_h: int, tex_w: int):
        utils._mem_clear()

        self.dataset: gom_utils.Dataset

        img_h, img_w = self.dataset.sample.img.shape[-2:]

        rgb_frames = torch.empty_like(
            self.dataset.sample.img,
            dtype=torch.float16,
            device=DEVICE,
        )
        # [T, C, H, W]

        mesh_frames = torch.empty_like(
            self.dataset.sample.img,
            dtype=torch.float16,
            device=DEVICE,
        )
        # [T, C, H, W]

        T, C, H, W = self.dataset.sample.img.shape

        zeros = torch.zeros((1, img_h, img_w), dtype=utils.FLOAT)
        ones = torch.ones((1, img_h, img_w), dtype=utils.FLOAT)

        batch_shape = self.dataset.shape

        self.module: gom_utils.Module

        # tex_grid

        tex_values = torch.zeros(
            (tex_h, tex_w, 3),
            dtype=utils.FLOAT, device=DEVICE)

        tex_weights = torch.zeros(
            (tex_h, tex_w),
            dtype=utils.FLOAT, device=DEVICE)

        tex_idx_grid = utils.idx_grid((tex_h, tex_w),
                                      dtype=torch.long, device=DEVICE)

        def weight_func(x): return (1e-3 + x).pow(-2)

        with torch.no_grad():
            if True:
                avatar_model: avatar_utils.AvatarModel = \
                    self.module.avatar_blender.get_avatar_model()

                avatar_model.mesh_data.show(avatar_model.vert_pos)

            for batch_idxes, sample in tqdm.tqdm(
                    dataset_utils.load(self.dataset, batch_size=BATCH_SIZE)):
                utils._mem_clear()

                batch_idxes: tuple[torch.Tensor, ...]

                batch_size = batch_idxes[0].shape[0]

                idxes = utils.ravel_idxes(batch_idxes, self.dataset.shape)
                # [K]

                idxes = idxes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    idxes.shape + (C, H, W))
                # [K, C, H, W]

                K = idxes.shape[0]

                sample: gom_utils.Sample = self.dataset[batch_idxes]

                result: gom_utils.ModuleForwardResult = self.module(
                    camera_config=sample.camera_config,
                    camera_transform=sample.camera_transform,
                    img=sample.img,
                    mask=sample.mask,
                    blending_param=sample.blending_param,
                )

                rendered_img = result.rendered_img.reshape((-1, C, H, W))
                # [K, C, H, W]

                rgb_frames.scatter_(
                    0,
                    idxes.to(rgb_frames.device),
                    rendered_img.to(rgb_frames))

                """

                out_frames[idxes[k, c, h, w], c, h, w] =
                    rendered_img[k, c, h, w]

                """

                data_points: list[torch.Tensor] = list()
                data_values: list[torch.Tensor] = list()

                for k in range(batch_size):
                    cur_avatar_model = result.avatar_model[k]

                    rendered_img = result.rendered_img[k]
                    # [C, H, W]

                    mesh_ras_result = rendering_utils.rasterize_mesh(
                        vert_pos=cur_avatar_model.vert_pos,
                        faces=cur_avatar_model.mesh_data.f_to_vvv,
                        camera_config=sample.camera_config,
                        camera_transform=sample.camera_transform[k],
                        faces_per_pixel=1,
                    )

                    pixel_to_faces: torch.Tensor = \
                        mesh_ras_result["pixel_to_faces"]
                    # [img_h, img_w, faces_per_pixel]

                    bary_coords: torch.Tensor = \
                        mesh_ras_result["bary_coords"]
                    # [img_h, img_w, faces_per_pixel, 3]

                    idx = (pixel_to_faces != -1).nonzero()
                    # [z, 3]

                    if idx.numel() == 0:
                        continue

                    pixel_h_idx = idx[:, 0]
                    pixel_w_idx = idx[:, 1]
                    p_idx = idx[:, 2]

                    f = pixel_to_faces[pixel_h_idx, pixel_w_idx, p_idx]
                    # [z]

                    tv = cur_avatar_model.tex_mesh_data.f_to_vvv[f, :]
                    # [z, 3]

                    tva, tvb, tvc = tv[:, 0], tv[:, 1], tv[:, 2]
                    # [z]

                    tvpa = cur_avatar_model.tex_vert_pos[tva]
                    tvpb = cur_avatar_model.tex_vert_pos[tvb]
                    tvpc = cur_avatar_model.tex_vert_pos[tvc]
                    # [z, 2]

                    b = bary_coords[pixel_h_idx, pixel_w_idx, p_idx]
                    # [z, 3]

                    ba, bb, bc = b[:, 0, None], b[:, 1, None], b[:, 2, None]
                    # [z, 1]

                    tvp = tvpa * ba + tvpb * bb + tvpc * bc
                    # [z, 2]

                    data_points = tvp
                    # [z, 2]

                    data_values = rendered_img[:, pixel_h_idx, pixel_w_idx]
                    # [3, z]

                    data_values = data_values.transpose(-1, -2)
                    # [z, 3]

                    cur_tex_values, cur_tex_weights = dw_interp_utils.gather(
                        data_points,  # [z, 2]
                        data_values,  # [z, 3]
                        tex_idx_grid,  # [tex_h, tex_w, 2]
                        weight_func,
                    )

                    # cur_tex_values[tex_h, tex_w, 3]
                    # cur_tex_weights[tex_h, tex_w]

                    tex_values += cur_tex_values
                    tex_weights += cur_tex_weights

        tex_values /= tex_weights.unsqueeze(-1)
        # [H, W, C] / ([H, W] -> [H, W, 1])

        utils.write_image(
            path=PROJ_DIR / f"mesh_{int(time.time())}.png",
            img=einops.rearrange(tex_values, "h w c -> c h w"),
        )

        """
        utils.write_video(
            path=PROJ_DIR / f"mesh_{int(time.time())}.mp4",
            video=mesh_frames,
            fps=25,
        )
        """

    @utils.mem_clear
    def bake_texture_oven(self, tex_h: int, tex_w):
        assert 0 < tex_h
        assert 0 < tex_w

        self.module: gom_utils.Module

        avatar_model: avatar_utils.AvatarModel = \
            self.module.avatar_blender.get_avatar_model()

        tex_oven = texture_utils.make_texture_oven(
            avatar_model.tex_vert_pos, tex_h, tex_w)

        tex_vert_pos = tex_oven.tex_vert_pos

        tex_faces = avatar_model.tex_mesh_data.f_to_vvv

        global_gp_result = self.module.get_world_gp(
            tex_vert_pos[tex_faces[:, 0], :],
            tex_vert_pos[tex_faces[:, 1], :],
            tex_vert_pos[tex_faces[:, 2], :],
        )

        # global_gp_result.gp_means[F, 3]
        # global_gp_result.gp_rot_qs[F, 3]
        # global_gp_result.gp_scales[F, 3]
        # global_gp_result.gp_colors[F, C]
        # global_gp_result.gp_opacities[F, 1]

        color_channels_cnt = global_gp_result.gp_colors.shape[1]

        rendered_result = gaussian_utils.render_gaussian(
            camera_config=tex_oven.camera_config,
            camera_transform=tex_oven.camera_transform,

            sh_degree=0,

            bg_color=torch.ones((color_channels_cnt,),
                                dtype=global_gp_result.gp_colors.dtype),

            gp_means=global_gp_result.gp_means,
            gp_rots=global_gp_result.gp_rot_qs,
            gp_scales=global_gp_result.gp_scales,

            gp_shs=None,
            gp_colors=global_gp_result.gp_colors,

            gp_opacities=global_gp_result.gp_opacities,

            device=utils.CUDA_DEVICE,
        )  # [...]

        tex = rendered_result.colors
        # [C, H, W]

        utils.write_image(
            PROJ_DIR / f"tex_{int(time.time())}.png",
            tex,
        )

    @utils.mem_clear
    @torch.no_grad()
    def bake_texture_oven_dw(self, tex_h: int, tex_w):
        # call bake_texture_oven_dw tex_h=1000 tex_w=1000

        assert 0 < tex_h
        assert 0 < tex_w

        self.module: gom_utils.Module

        avatar_model: avatar_utils.AvatarModel = \
            self.module.avatar_blender.get_avatar_model()

        tex_vert_pos = avatar_model.tex_vert_pos
        # [TV, 2]

        TV = utils.check_shapes(tex_vert_pos, (-1, 2))

        img_tex_vert_pos = torch.empty(
            (TV, 3), dtype=tex_vert_pos.dtype, device=tex_vert_pos.device)

        img_tex_vert_pos[:, :2] = texture_utils.tex_coord_to_img_coord(
            tex_vert_pos, tex_h, tex_w)
        img_tex_vert_pos[:, 2] = 0

        tex_faces = avatar_model.tex_mesh_data.f_to_vvv

        global_gp_result = self.module.get_world_gp(
            img_tex_vert_pos[tex_faces[:, 0], :],
            img_tex_vert_pos[tex_faces[:, 1], :],
            img_tex_vert_pos[tex_faces[:, 2], :],
        )

        # global_gp_result.gp_means[F, 3]
        # global_gp_result.gp_rot_qs[F, 3]
        # global_gp_result.gp_scales[F, 3]
        # global_gp_result.gp_colors[F, C]
        # global_gp_result.gp_opacities[F, 1]

        color_channels_cnt = global_gp_result.gp_colors.shape[1]

        bg_color = torch.ones(
            (color_channels_cnt,),
            dtype=global_gp_result.gp_colors.dtype,
            device=global_gp_result.gp_colors.device)

        colors = bg_color + \
            (global_gp_result.gp_colors - bg_color) * \
            global_gp_result.gp_opacities
        # [F, C]

        tex_idx_grid = utils.idx_grid(
            (tex_h, tex_w), dtype=torch.long, device=DEVICE)

        def weight_func(x): return (1e-3 + x).pow(-2)

        tex_values = dw_interp_utils.interp(
            global_gp_result.gp_means[:, :2],  # [F, 2]
            colors,  # [F, C]
            tex_idx_grid,  # [tex_h, tex_w, 2]
            weight_func,
        )
        # tex_values[tex_h, tex_w, C]

        utils.write_image(
            PROJ_DIR / f"tex_{int(time.time())}.png",
            einops.rearrange(tex_values, "h w c -> c h w"),
        )

    @utils.mem_clear
    @torch.no_grad()
    def bake_texture_face(self, tex_h: int, tex_w: int):
        # call bake_texture_face tex_h=1000 tex_w=1000

        self.dataset: gom_utils.Dataset

        img_h, img_w = self.dataset.sample.img.shape[-2:]

        T, C, H, W = self.dataset.sample.img.shape

        self.module: gom_utils.Module

        avatar_model: avatar_utils.AvatarModel = \
            self.module.avatar_blender.get_avatar_model()

        F = avatar_model.faces_cnt

        face_color_sum = torch.zeros(
            (F + 1, 3), dtype=torch.float64, device=DEVICE)

        face_weight_sum = torch.zeros(
            (F + 1,), dtype=torch.int, device=DEVICE)

        face_ones = torch.ones(
            (tex_h * tex_w,), dtype=torch.int, device=DEVICE)

        for batch_idxes, sample in tqdm.tqdm(
                dataset_utils.load(self.dataset, batch_size=BATCH_SIZE)):
            utils._mem_clear()

            batch_idxes: tuple[torch.Tensor, ...]

            batch_size = batch_idxes[0].shape[0]

            idxes = utils.ravel_idxes(batch_idxes, self.dataset.shape)
            # [K]

            idxes = idxes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                idxes.shape + (C, H, W))
            # [K, C, H, W]

            K = idxes.shape[0]

            sample: gom_utils.Sample = self.dataset[batch_idxes]

            result: gom_utils.ModuleForwardResult = self.module(
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            rendered_img = result.rendered_img.reshape((-1, C, H, W))
            # [K, C, H, W]

            for k in range(batch_size):
                cur_avatar_model = result.avatar_model[k]

                rendered_img = result.rendered_img[k]
                # [C, H, W]

                mesh_ras_result = rendering_utils.rasterize_mesh(
                    vert_pos=cur_avatar_model.vert_pos,
                    faces=cur_avatar_model.mesh_data.f_to_vvv,
                    camera_config=sample.camera_config,
                    camera_transform=sample.camera_transform[k],
                    faces_per_pixel=1,
                )

                pixel_to_faces = mesh_ras_result.pixel_to_faces
                # [img_h, img_w, 1]

                rendered_img = einops.rearrange(
                    rendered_img, "c h w -> h w c").reshape((-1, 3))
                # [H * W, 3]

                pixel_to_faces = pixel_to_faces.reshape(-1)
                # [H * W]

                idx = (pixel_to_faces != -1).nonzero().reshape(-1)
                # [z]

                if idx.numel() == 0:
                    continue

                rendered_img = rendered_img[idx, :]
                # [z, 3]

                pixel_to_faces = pixel_to_faces[idx]
                # [z]

                face_color_sum.index_add_(
                    0, pixel_to_faces, rendered_img.to(face_color_sum.dtype))
                face_weight_sum.index_add_(
                    0, pixel_to_faces, face_ones[:pixel_to_faces.shape[0]])

                """

                face_color_sum[pixel_to_faces[i], :] += rendered_img[i, :]
                face_weight_sum[pixel_to_faces[i]] += face_ones[i]

                """

        face_idx_map = texture_utils.calc_face_idx(
            tex_vert_pos=avatar_model.tex_vert_pos,
            tex_faces=avatar_model.tex_mesh_data.f_to_vvv,

            tex_h=tex_h,
            tex_w=tex_w,
        )
        # [H, W, 1]

        print(f"{face_idx_map=}")

        print(f"{face_idx_map.shape=}")

        print(f"{face_idx_map.min()=}")
        print(f"{face_idx_map.max()=}")

        face_color = face_color_sum / (1e-2 + face_weight_sum).unsqueeze(-1)
        # [F + 1, 3]

        face_color[F, :] = 1

        print(f"{face_idx_map.dtype=}")

        face_idx_map = (face_idx_map + (F + 1)) % (F + 1)
        # -1 -> F           [0, F) -> [0, F)

        face_idx_map = face_idx_map.reshape(-1).unsqueeze(-1) \
            .expand((tex_h * tex_w, 3))

        print(f"{face_color_sum=}")
        print(f"{face_weight_sum=}")

        # [H * W]

        tex = torch.gather(face_color, 0, face_idx_map)
        # [H * W, 3]

        """
        tex[i, j] = face_color[face_idx_map[i, j], j]
        """

        tex = tex.reshape((tex_h, tex_w, 3))

        utils.write_image(
            path=PROJ_DIR / f"tex_{int(time.time())}.png",
            img=einops.rearrange(tex, "h w c -> c h w"),
        )


@beartype
def map_to_texture(
    gom_module: gom_utils.Module,
    mapping_blending_param: object,
    img_h: int,
    img_w: int,
):
    assert 0 < img_h
    assert 0 < img_w

    avatar_blender = gom_module.avatar_blender

    avatar_model: avatar_utils.AvatarModel = avatar_blender(
        mapping_blending_param)

    vert_pos = avatar_model.vert_pos \
        .reshape((-1, 3)).to(utils.CPU_DEVICE)
    # [V, 3]

    faces = avatar_model.faces \
        .reshape((-1, 3)).to(utils.CPU_DEVICE)
    # [F, 3]

    tex_vert_pos = avatar_model.tex_vert_pos \
        .reshape((-1, 2)).to(utils.CPU_DEVICE)
    # [TV, 2]

    tex_faces = avatar_model.texture_faces \
        .reshape((-1, 3)).to(utils.CPU_DEVICE)
    # [F, 3]

    gp_colors = torch.exp(gom_module.gp_colors) \
        .to(utils.CPU_DEVICE)
    # [F, C]

    gp_opacities = torch.exp(gom_module.gp_log_opacities) \
        .to(utils.CPU_DEVICE)
    # [F, 1]

    C, V, TV, F = -1, -2, -3, -4

    C, V, TV, F = utils.check_shapes(
        vert_pos, (V, 3),
        tex_vert_pos, (TV, 2),
        tex_faces, (F, 3),
        gp_colors, (F, C),
        gp_opacities, (F, 1),
    )

    m = texture_utils.position_to_map(
        vert_pos=vert_pos,
        faces=faces,

        tex_vert_pos=tex_vert_pos,
        tex_faces=tex_faces,

        img_h=img_h,
        img_w=img_w,
    )
    # [img_h, img_w]

    bg_color = torch.ones((C,), dtype=utils.FLOAT)

    ret = torch.zeros((3, img_h, img_w), dtype=utils.FLOAT)

    with torch.no_grad():
        for pixel_i in range(img_h):
            for pixel_j in range(img_w):
                l = m[pixel_i][pixel_j]

                if l is None:
                    continue

                if 1 < len(l):
                    print(f"multi mapping at ({pixel_i}, {pixel_j})")
                    print(l)

                face_i, _ = l[0]

                gp_opacity = gp_opacities[face_i]
                # [1]

                ret[:, pixel_i, pixel_j] = (
                    bg_color * (1 - gp_opacity) +
                    gp_colors[face_i, :].abs() * gp_opacity)

    print(f"writing")

    print(f"{ret.min()=}")
    print(f"{ret.max()=}")

    utils.write_image(
        DIR / " tex_map.png",
        ret.to(utils.CPU_DEVICE),
    )


def main1():
    torch.autograd.set_detect_anomaly(True, True)

    smpl_model_data_path_dict = {
        "male": config.SMPL_MALE_MODEL,
        "female": config.SMPL_FEMALE_MODEL,
        "neutral": config.SMPL_NEUTRAL_MODEL,
    }

    smplx_model_data_path_dict = {
        "male": config.SMPLX_MALE_MODEL,
        "female": config.SMPLX_FEMALE_MODEL,
        "neutral": config.SMPLX_NEUTRAL_MODEL,
    }

    model_data_dict = {
        key: smplx_utils.ModelData.from_file(
            model_data_path=value,
            model_config=smplx_utils.smpl_model_config,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    subject_data = people_snapshot_utils.read_subject(
        subject_dir, model_data_dict, DEVICE)

    dataset = gom_utils.Dataset(gom_utils.Sample(
        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,
        img=subject_data.video,
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,
    )).to(DEVICE)

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        model_data=subject_data.model_data,
    ).to(DEVICE)

    smplx_model_builder.unfreeze()

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    gom_avatar_module = gom_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
    ).to(DEVICE).train()

    lr = 1e-3

    param_groups = utils.get_param_groups(gom_avatar_module, lr)

    print(param_groups)

    optimizer = torch.optim.Adam(
        param_groups,
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=pow(0.1, 1/4),
        patience=5,
        threshold=0.05,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-8,
    )

    training_core = MyTrainingCore(
        module=gom_avatar_module,
        dataset=dataset,
        loss_func=MyLossFunc,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        device=DEVICE,
    )

    trainer.set_training_core(training_core)

    # trainer.load_latest()

    # trainer.training_core.bake_texture_face(1000, 1000)

    trainer.enter_cli()


if __name__ == "__main__":
    main1()

    print("ok")
