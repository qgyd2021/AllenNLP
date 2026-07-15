#!/usr/bin/python3
# -*- coding: utf-8 -*-
import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Tuple, Union

logger = logging.getLogger(__name__)

LoaderResult = Union[Any, Tuple[Any, str]]
ModelLoader = Callable[[str], LoaderResult]


def _get_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


@dataclass
class _ModelRecord:
    key: str
    model: Any
    device: str
    last_used: float
    in_use: int = 0


class TTLModelManager(object):
    def __init__(
        self,
        loader: ModelLoader,
        gpu_idle_seconds: int = 300,
        cpu_idle_seconds: int = 900,
        sweep_interval: int = 30,
        prefer_gpu: bool = True,
        cuda_device: str = "cuda:0",
        relocate_on_acquire: bool = True,
    ):
        self.loader = loader
        self.gpu_idle_seconds = max(0, int(gpu_idle_seconds))
        self.cpu_idle_seconds = max(0, int(cpu_idle_seconds))
        self.sweep_interval = max(1, int(sweep_interval))
        self.prefer_gpu = bool(prefer_gpu)
        self.cuda_device = cuda_device
        self.relocate_on_acquire = bool(relocate_on_acquire)

        self._records: Dict[str, _ModelRecord] = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="ttl-model-manager",
            daemon=True,
        )
        self._sweeper.start()

    def _target_device(self) -> str:
        torch = _get_torch()
        if self.prefer_gpu and torch is not None and torch.cuda.is_available():
            return self.cuda_device
        return "cpu"

    def _move_model(self, model: Any, device: str) -> bool:
        if hasattr(model, "to"):
            try:
                model.to(device)
                return True
            except Exception:
                logger.exception("failed to move model to %s", device)
        return False

    def _empty_cuda_cache(self) -> None:
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_loaded(self, loaded: LoaderResult) -> Tuple[Any, str]:
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, device = loaded
            return model, str(device)
        model = loaded
        target_device = self._target_device()
        if self.relocate_on_acquire:
            self._move_model(model, target_device)
        return model, target_device

    def _load_record(self, key: str, now: float) -> _ModelRecord:
        model, target_device = self._resolve_loaded(self.loader(key))
        return _ModelRecord(
            key=key,
            model=model,
            device=target_device,
            last_used=now,
            in_use=0,
        )

    def acquire(self, key: str) -> Tuple[Any, str]:
        with self._lock:
            now = time.time()
            record = self._records.get(key)
            if record is None:
                record = self._load_record(key, now=now)
                self._records[key] = record
            else:
                if self.relocate_on_acquire:
                    target_device = self._target_device()
                    if record.device != target_device:
                        if self._move_model(record.model, target_device):
                            record.device = target_device
                record.last_used = now

            record.in_use += 1
            return record.model, record.device

    def release(self, key: str) -> None:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return
            record.in_use = max(0, record.in_use - 1)
            record.last_used = time.time()

    @contextmanager
    def use(self, key: str) -> Iterator[Tuple[Any, str]]:
        model, device = self.acquire(key)
        try:
            yield model, device
        finally:
            self.release(key)

    def sweep_once(self) -> None:
        with self._lock:
            now = time.time()
            keys_to_delete = []

            for key, record in list(self._records.items()):
                if record.in_use > 0:
                    continue

                idle_seconds = now - record.last_used
                if record.device != "cpu" and idle_seconds >= self.gpu_idle_seconds:
                    if self._move_model(record.model, "cpu"):
                        record.device = "cpu"
                        record.last_used = now
                        self._empty_cuda_cache()
                        logger.info("model moved to cpu. key=%s", key)
                    else:
                        keys_to_delete.append(key)
                        logger.info("model unload scheduled. key=%s", key)
                    continue

                if record.device == "cpu" and idle_seconds >= self.cpu_idle_seconds:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                self._records.pop(key, None)
                logger.info("model unloaded. key=%s", key)

        if keys_to_delete:
            gc.collect()
            self._empty_cuda_cache()

    def _sweep_loop(self) -> None:
        while not self._closed.wait(self.sweep_interval):
            self.sweep_once()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            self._records.clear()
        gc.collect()
        self._empty_cuda_cache()

    def state(self) -> Dict[str, dict]:
        with self._lock:
            return {
                key: {
                    "device": record.device,
                    "last_used": record.last_used,
                    "in_use": record.in_use,
                }
                for key, record in self._records.items()
            }
