import pickle

import pathlib

import json
import asyncio

import datetime

import time

import aiohttp

import glob

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

    smplx_params_dir = DIR / "sample2/smplx_params"

    for frame_idx in range(len(glob.glob(f"{smplx_params_dir}/*.json"))):
        filename = smplx_params_dir / f"smplx_param_{frame_idx}.json"

        print(f"{filename}")
        with open(filename) as f:
            smplx_params.append(json.load(f))

    asyncio.run(LoopEmitSMPLXParams(smplx_params, 1 / 35))


def main2():
    dir = DIR / "sample3"

    frame_idx = 0

    for filename in sorted(glob.glob(f"{dir}/poses original/*.json")):
        with open(filename) as f:
            data = json.load(f)["pose"]

        with open(f"{dir}/smplx_params/smplx_param_{frame_idx}.json", "w") as f:
            json.dump(data, f)
            frame_idx += 1


if __name__ == "__main__":
    main1()
