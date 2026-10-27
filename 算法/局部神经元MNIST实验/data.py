"""Download and validate the canonical MNIST IDX files using only the stdlib.

Official-test files are fetched only when ``load_mnist(..., split="test")`` is
called explicitly. Training code should use ``train`` plus a frozen dev split
from the official training set.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import struct
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4


DEFAULT_CACHE_DIR = Path("C:/pioneer/mnist-local-neuron-data")
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
)

_FILES = {
    "train": {
        "images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        "labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        "count": 60000,
    },
    "test": {
        "images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        "labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
        "count": 10000,
    },
}


@dataclass(frozen=True)
class MNISTData:
    split: Literal["train", "test"]
    images: bytes
    labels: bytes
    count: int
    rows: int
    columns: int
    compressed_sha256: dict[str, str]
    raw_sha256: dict[str, str]
    cache_files: dict[str, str]

    def hashes(self) -> dict[str, object]:
        return {
            "split": self.split,
            "count": self.count,
            "image_shape": [self.count, self.rows, self.columns],
            "compressed_sha256": self.compressed_sha256,
            "raw_sha256": self.raw_sha256,
            "cache_files": self.cache_files,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(filename: str, expected_md5: str, cache_dir: Path, timeout: float) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if destination.is_file() and _md5_file(destination) == expected_md5:
        return destination

    failures: list[str] = []
    for mirror in MIRRORS:
        temp_path = cache_dir / f".{filename}.{uuid4().hex}.part"
        url = f"{mirror}/{filename}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mnist-local-neuron/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()

            actual_md5 = _md5_file(temp_path)
            if actual_md5 != expected_md5:
                raise ValueError(f"MD5 mismatch: expected {expected_md5}, received {actual_md5}")
            temp_path.replace(destination)
            return destination
        except (OSError, urllib.error.URLError, ValueError) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    details = "\n".join(failures)
    raise RuntimeError(f"Could not fetch verified MNIST file {filename}. Attempts:\n{details}")


def _read_idx_images(gz_path: Path, expected_count: int) -> tuple[bytes, int, int]:
    with gzip.open(gz_path, "rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Truncated MNIST image IDX header: {gz_path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic {magic} in {gz_path}")
        if count != expected_count or rows != 28 or columns != 28:
            raise ValueError(
                f"Unexpected image IDX shape {(count, rows, columns)} in {gz_path}; "
                f"expected {(expected_count, 28, 28)}"
            )
        pixels = handle.read()
    expected_bytes = count * rows * columns
    if len(pixels) != expected_bytes:
        raise ValueError(f"Truncated image IDX payload in {gz_path}: {len(pixels)} != {expected_bytes}")
    return pixels, rows, columns


def _read_idx_labels(gz_path: Path, expected_count: int) -> bytes:
    with gzip.open(gz_path, "rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated MNIST label IDX header: {gz_path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049 or count != expected_count:
            raise ValueError(f"Unexpected label IDX header {(magic, count)} in {gz_path}")
        labels = handle.read()
    if len(labels) != expected_count:
        raise ValueError(f"Truncated label IDX payload in {gz_path}: {len(labels)} != {expected_count}")
    if any(label > 9 for label in labels):
        raise ValueError(f"Invalid digit label in {gz_path}")
    return labels


def load_mnist(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    split: Literal["train", "test"] = "train",
    timeout: float = 60.0,
) -> MNISTData:
    """Fetch, verify, and decode one MNIST split.

    ``split="test"`` is deliberately explicit; normal training should use the
    official training set and fixed train/dev indices only.
    """
    if split not in _FILES:
        raise ValueError(f"split must be one of {tuple(_FILES)}, received {split!r}")
    cache_dir = Path(cache_dir)
    spec = _FILES[split]
    count = int(spec["count"])
    image_name, image_md5 = spec["images"]  # type: ignore[misc]
    label_name, label_md5 = spec["labels"]  # type: ignore[misc]

    image_path = _download_verified(image_name, image_md5, cache_dir, timeout)
    label_path = _download_verified(label_name, label_md5, cache_dir, timeout)
    images, rows, columns = _read_idx_images(image_path, count)
    labels = _read_idx_labels(label_path, count)
    return MNISTData(
        split=split,
        images=images,
        labels=labels,
        count=count,
        rows=rows,
        columns=columns,
        compressed_sha256={"images": sha256_file(image_path), "labels": sha256_file(label_path)},
        raw_sha256={"images": sha256_bytes(images), "labels": sha256_bytes(labels)},
        cache_files={"images": str(image_path.resolve()), "labels": str(label_path.resolve())},
    )


def make_fixed_split_indices(
    total_count: int = 60000,
    train_count: int = 10000,
    dev_count: int = 5000,
    seed: int = 20261026,
) -> tuple[list[int], list[int]]:
    """Create deterministic, disjoint train/dev subsets from official train."""
    if train_count <= 0 or dev_count <= 0:
        raise ValueError("train_count and dev_count must be positive")
    if train_count + dev_count > total_count:
        raise ValueError("train_count + dev_count cannot exceed total_count")
    indices = list(range(total_count))
    random.Random(seed).shuffle(indices)
    train_indices = indices[:train_count]
    dev_indices = indices[train_count : train_count + dev_count]
    return train_indices, dev_indices


def split_payload(train_indices: list[int], dev_indices: list[int], seed: int) -> dict[str, object]:
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("train indices contain duplicates")
    if len(set(dev_indices)) != len(dev_indices):
        raise ValueError("dev indices contain duplicates")
    if set(train_indices).intersection(dev_indices):
        raise ValueError("train and dev indices overlap")
    return {
        "source_split": "official_train",
        "seed": seed,
        "train_count": len(train_indices),
        "dev_count": len(dev_indices),
        "train_indices": train_indices,
        "dev_indices": dev_indices,
    }


def write_split_indices(path: str | Path, payload: dict[str, object]) -> str:
    """Write a frozen split file and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return sha256_bytes(encoded)
