import torch
import tqdm
from beartype import beartype

from . import config, utils, video_seg_utils, vision_utils

DEVICE = utils.CPU_DEVICE

SEG_DIR = config.DIR / "video_seg_2025_0514_1"


def main1():
    video_path = config.DIR / "people_snapshot_public/female-1-casual/female-1-casual.mp4"

    video, fps = vision_utils.read_video(
        video_path, "RGB")

    video_seg_utils.segment(
        src=[i for i in video],  # [T, C, H, W]
        fps=fps,
        out_dir=SEG_DIR,
    )


def main2():
    seg_dir = SEG_DIR

    obj_type_list = [
        video_seg_utils.ObjectType.HAIR,
        video_seg_utils.ObjectType.UPPER_GARMENT,
        video_seg_utils.ObjectType.LOWER_GARMENT,
    ]

    person_mask_video = vision_utils.VideoReader(
        seg_dir / video_seg_utils.PERSON_MASK_FILENAME, "GRAY")

    mask_videos = {
        obj_type:  vision_utils.VideoReader(
            seg_dir /
            video_seg_utils.get_obj_mask_filename(obj_type), "GRAY")

        for obj_type in obj_type_list
    }

    video_writer = vision_utils.VideoWriter(
        SEG_DIR / "mask_SKIN.avi",
        height=person_mask_video.height,
        width=person_mask_video.width,
        color_type="GRAY",
        fps=person_mask_video.fps,
    )

    while True:
        person_mask_frame = person_mask_video.read()
        frames = [mask_video.read() for mask_video in mask_videos.values()]

        if person_mask_frame is None or any(frame is None for frame in frames):
            break

        acc_mask = torch.zeros_like(person_mask_frame)

        for frame in frames:
            cur_mask = frame.to(torch.float64) / 255

            if acc_mask is None:
                acc_mask = cur_mask
            else:
                acc_mask = torch.max(acc_mask, cur_mask)

        acc_mask = torch.min(person_mask_frame.to(torch.float64) / 255,
                             1 - acc_mask)

        video_writer.write(utils.rct(
            acc_mask * 255,
            dtype=torch.uint8))

    video_writer.close()

    video_reader = vision_utils.VideoReader(
        SEG_DIR / "mask_SKIN.avi",
        color_type="GRAY",
    )

    refined_video_writer = vision_utils.VideoWriter(
        SEG_DIR / "refined_mask_SKIN.avi",
        height=person_mask_video.height,
        width=person_mask_video.width,
        color_type="GRAY",
        fps=video_reader.fps,
    )

    for i in tqdm.tqdm(video_seg_utils.refine(video_reader, video_reader.fps)):
        refined_video_writer.write(utils.rct(i, dtype=torch.uint8))

    refined_video_writer.close()


if __name__ == "__main__":
    main2()

    print("ok")
