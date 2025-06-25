
import functools
import typing

import densepose
import densepose.vis
import densepose.vis.densepose_results
import densepose.vis.extractor
import detectron2
import detectron2.config
import detectron2.engine
import detectron2.projects
import numpy as np
import torch
from beartype import beartype

from . import config, utils, vision_utils


@functools.cache
def get_cfg():
    cfg = detectron2.config.get_cfg()

    densepose.add_densepose_config(cfg)

    cfg.merge_from_file(str(config.DENSEPOSE_CONFIG.resolve()))

    cfg.MODEL.WEIGHTS = str(config.DENSEPOSE_MODEL.resolve())

    cfg.MODEL.DEVICE = str(
        utils.CUDA_DEVICE if torch.cuda.is_available() else utils.CPU_DEVICE)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    return cfg


@functools.cache
def get_predictor():
    return detectron2.engine.DefaultPredictor(get_cfg())


@functools.cache
def get_extractor():
    return densepose.vis.extractor.DensePoseResultExtractor()


@functools.cache
def get_visualizer():
    return densepose.vis.densepose_results.DensePoseResultsFineSegmentationVisualizer()


@beartype
class predict:
    def __init__(
        self,
        img: typing.Iterable[torch.Tensor],  # [C, H, W]
    ):
        self.img = img

        try:
            self.length = len(img)
        except:
            self.length = None

    @beartype
    def __len__(self) -> typing.Optional[int]:
        return self.length

    @beartype
    def __iter__(self) -> typing.Iterable[torch.Tensor]:  # [C, H, W]
        for cur_img in self.img:
            cur_opencv_img = vision_utils.to_opencv_image(
                cur_img, "RGB")

            with torch.no_grad():
                outputs = get_extractor()(
                    get_predictor()(cur_opencv_img)["instances"])

            cur_opencv_seg_img = np.zeros(cur_opencv_img.shape, dtype=np.uint8)

            get_visualizer().visualize(cur_opencv_seg_img, outputs)

            yield vision_utils.from_opencv_image(
                cur_opencv_seg_img, "RGB")
