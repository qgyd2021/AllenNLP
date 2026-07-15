#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"


_download_locks: Dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_download_lock(model_id: str) -> threading.Lock:
    with _registry_lock:
        lock = _download_locks.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _download_locks[model_id] = lock
        return lock


def list_model_files(model_id: str, revision: str = "master") -> List[dict]:
    url = "{}/api/v1/models/{}/repo/files?Revision={}&Recursive=true".format(
        MODELSCOPE_ENDPOINT,
        model_id,
        quote_plus(revision),
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    if data.get("Code") != 200 or not data.get("Success"):
        raise RuntimeError("Failed to list model files: {}".format(data.get("Message", data)))

    files = data["Data"]["Files"]
    return [
        file_info
        for file_info in files
        if file_info.get("Type") != "tree"
        and file_info["Name"] not in (".gitignore", ".gitattributes")
    ]


def get_file_download_url(model_id: str, file_path: str, revision: str = "master") -> str:
    return "{}/api/v1/models/{}/repo?Revision={}&FilePath={}".format(
        MODELSCOPE_ENDPOINT,
        model_id,
        quote_plus(revision),
        quote_plus(file_path),
    )


def download_file(url: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(save_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=save_path.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def snapshot_download(
    model_id: str,
    local_dir: Path,
    revision: str = "master",
) -> Path:
    files = list_model_files(model_id, revision=revision)
    local_dir.mkdir(parents=True, exist_ok=True)

    for file_info in files:
        file_path = file_info["Path"]
        url = get_file_download_url(model_id, file_path, revision=revision)
        download_file(url, local_dir / file_path)

    return local_dir


def ensure_model_downloaded(
    model_id: str,
    model_root: Path,
    revision: str = "master",
) -> Tuple[Path, bool]:
    local_dir = model_root / model_id
    if local_dir.exists():
        return local_dir, False

    lock = _get_download_lock(model_id)
    with lock:
        logger.info("下载模型: %s -> %s", model_id, local_dir)
        snapshot_download(
            model_id=model_id,
            local_dir=local_dir,
            revision=revision,
        )
        logger.info("模型下载完成: %s", local_dir)
        return local_dir, True
