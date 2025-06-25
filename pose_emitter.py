import asyncio
import datetime
import glob
import json
import os
import pathlib
import pickle
import time

import aiohttp

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


async def EmitSMPLXParam(timestamp, smplx_param):
    pose_server_url = "http://jorjinapp.ddns.net:16385"

    url = f"{pose_server_url}/set_pose"

    request_data = {
        "timestamp": timestamp,
        "pose": smplx_param,
    }

    timeout = aiohttp.ClientTimeout(total=datetime.timedelta(hours=1).seconds)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False), timeout=timeout) as session:
        async with session.post(url, json=request_data, timeout=timeout) as response:
            await response.json()


async def LoopEmitSMPLXParams(smplx_params, duration):
    l = len(smplx_params)

    while True:
        """
        for i in range(l * 2 - 2):
            j = i if i < l else l * 2 - 2 - i
            print(f"{l} {i} {j}")

            await EmitSMPLXParam(int(time.time() * 1000), smplx_params[j])

            await asyncio.sleep(duration)
        """

        for i in range(l):
            if i % 10 == 0:
                print(f"{l} {i}")
            await EmitSMPLXParam(int(time.time() * 1000), smplx_params[i])
            await asyncio.sleep(duration)

        await asyncio.sleep(10)


def main1():
    smplx_params = list()

    smplx_params_dir = DIR / "0513/smplx_params"

    with open(DIR / "merged.json") as f:
        hand_pose = json.load(f)

    for frame_idx in range(len(glob.glob(f"{smplx_params_dir}/*.json"))):
        filename = smplx_params_dir / f"smplx_param_{frame_idx:0>5}.json"
        # filename = smplx_params_dir / f"smplx_param_{frame_idx}.json"

        print(f"{filename}")
        with open(filename) as f:
            d = json.load(f)

            # d["lhand_pose"] = hand_pose["lhand_pose"]
            # d["rhand_pose"] = hand_pose["rhand_pose"]

            smplx_params.append(d)

    asyncio.run(LoopEmitSMPLXParams(smplx_params, 1 / 35))


def main2():
    dir = DIR / "ROMP_result_v3"

    frame_idx = 0

    for filename in sorted(glob.glob(f"{dir}/smplx_params_/*.json")):
        with open(filename) as f:
            data = json.load(f)["pose"]

        (dir / "smplx_params").mkdir(parents=True, exist_ok=True)

        with open(f"{dir}/smplx_params/smplx_param_{frame_idx}.json", "w") as f:
            json.dump(data, f)
            frame_idx += 1


if __name__ == "__main__":
    main1()
