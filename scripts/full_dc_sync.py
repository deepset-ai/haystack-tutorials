from typing import List, Dict

import os
import sys
import yaml
import requests
import asyncio
import json
from pathlib import Path


class dCAuth(requests.auth.AuthBase):
    def __call__(self, r):
        r.headers["authorization"] = f"Bearer {dc_token()}"
        return r


def dc_token():
    api_key = os.getenv("DEEPSET_CLOUD_API_KEY", None)
    if not api_key:
        raise Exception("DEEPSET_CLOUD_API_KEY env var is not set")
    return api_key


async def upload_to_dc(workspace: str, file: Path, meta: Dict):
    url = f"https://api.cloud.deepset.ai/api/v1/workspaces/{workspace}/files?write_mode=OVERWRITE"
    files = {"file": (file.name, file.open("rb"), "text/plain")}
    data = {"meta": json.dumps(meta, default=str)}
    print(f"Uploading {file.name}")
    res = requests.post(url, data=data, files=files, auth=dCAuth())
    try:
        res.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{file.name} upload failed.\n{e}: {res.text}")
    print(res.json())


async def upload_file(file: Path, workspace: str):
    metadata_file = Path("text", f"{file.stem}.yml")
    meta = {}
    with metadata_file.open(encoding="utf-8") as f:
        meta = yaml.unsafe_load(f)

    await upload_to_dc(workspace, file, meta)


async def upload_files(files: List[Path], workspace: str):
    async with asyncio.TaskGroup() as tg:
        for f in files:
            tg.create_task(upload_file(f, workspace))


def delete_all_cloud_files(workspace: str):
    url = f"https://api.cloud.deepset.ai/api/v1/workspaces/{workspace}/files"
    res = requests.delete(url, auth=dCAuth())
    try:
        res.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e}: {res.text}")


if __name__ == "__main__":
    workspace = os.getenv("DEEPSET_CLOUD_WORKSPACE", None)
    if not workspace:
        print("DEEPSET_CLOUD_WORKSPACE env var not set")
        sys.exit(1)

    if "DEEPSET_CLOUD_API_KEY" not in os.environ:
        print("DEEPSET_CLOUD_API_KEY env var not set")
        sys.exit(1)

    delete_all_cloud_files(workspace)
    print(f"Deleted all files from workspace {workspace}")

    tutorials = Path(".", "text").glob("*.txt")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(upload_files(tutorials, workspace))
