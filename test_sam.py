import torch

import sam2


import torch
import pathlib

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    config = DIR / "sam2_data/sam2.1_hiera_l.yaml"
    checkpoint = DIR / "sam2_data/sam2.1_hiera_large.pt"

    video_path = DIR / "people_snapshot_public/female-3-sport/female-3-sport.mp4"

    module = sam2.build_sam.build_sam2_video_predictor(config, checkpoint)

    inference_state = module.init_state(video_path)

    predictor = sam2.sam2_image_predictor.SAM2ImagePredictor(
        sam2.build_sam.build_sam2(model_cfg, checkpoint))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image()
        masks, _, _ = predictor.predict()


if __name__ == "__main__":
    main1()
