"""Shared reproducibility, I/O, and configuration utilities."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import numpy as np
import yaml

try:
    import orjson

    def loads_json(line: str | bytes) -> Any:
        return orjson.loads(line)

    def dumps_json(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")

except ImportError:  # pragma: no cover

    def loads_json(line: str | bytes) -> Any:
        return json.loads(line)

    def dumps_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash_int(text: str) -> int:
    """Deterministic cross-platform 64-bit integer hash (independent of PYTHONHASHSEED)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Fast streaming JSONL reader."""
    with open(path, "rb") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            yield loads_json(raw_line)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    """Fast streaming JSONL writer returning number of records written."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(dumps_json(rec))
            f.write("\n")
            count += 1
    return count
