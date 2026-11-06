"""Content-addressed stage caches. Only an atomically published manifest is valid."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import errno
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np

from .contracts import SCHEMA_VERSION, STAGES


class CacheError(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: Path, format_name: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if (not isinstance(value, dict) or value.get("format") != format_name
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != SCHEMA_VERSION):
        raise ValueError(f"unsupported {format_name} schema: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    if "format" not in value or "schema_version" not in value:
        raise ValueError("JSON artifacts require format and schema_version")
    data = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stage_key(stage: str, version: str, parameters: dict, parents: tuple[str, ...] = ()) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage}")
    return {"stage": stage, "algorithm_version": version,
            "parameters": parameters, "parents": list(parents)}


class Cache:
    def __init__(self, path: Path, expected_key: dict | None = None):
        self.path = Path(path)
        try:
            self.manifest = read_json(self.path / "manifest.json", "ez2rc.stage-cache")
            checksum = self.manifest.get("manifest_sha256")
            if checksum != digest({k: v for k, v in self.manifest.items() if k != "manifest_sha256"}):
                raise CacheError("cache manifest checksum mismatch")
            key = self.manifest["key"]
            if (self.manifest["status"] != "complete" or key["stage"] not in STAGES
                    or digest(key) != self.manifest["cache_id"]
                    or expected_key is not None and key != expected_key):
                raise CacheError("cache identity, version or status mismatch")
            chunks = self.manifest["chunks"]
            if not isinstance(chunks, list) or not chunks:
                raise CacheError("cache has no chunks")
            for i, chunk in enumerate(chunks):
                if chunk["file"] != f"chunk-{i:06d}.npz":
                    raise CacheError("invalid chunk path/order")
                source = self.path / chunk["file"]
                if source.is_symlink() or file_hash(source) != chunk["sha256"]:
                    raise CacheError("cache chunk checksum mismatch")
                self._read_chunk(chunk)
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            raise CacheError(f"invalid cache {self.path}: {exc}") from exc

    @property
    def id(self) -> str:
        return self.manifest["cache_id"]

    def _read_chunk(self, chunk: dict) -> dict[str, np.ndarray]:
        with np.load(self.path / chunk["file"], allow_pickle=False) as archive:
            if set(archive.files) != set(chunk["arrays"]):
                raise CacheError("cache array names mismatch")
            arrays = {name: archive[name] for name in archive.files}
        for name, array in arrays.items():
            spec = chunk["arrays"][name]
            if (list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]
                    or array.dtype.hasobject):
                raise CacheError("cache array shape/type mismatch")
        return arrays

    def chunks(self):
        for chunk in self.manifest["chunks"]:
            if file_hash(self.path / chunk["file"]) != chunk["sha256"]:
                raise CacheError("cache changed after validation")
            yield self._read_chunk(chunk)


class CacheWriter(AbstractContextManager):
    def __init__(self, root: Path, key: dict):
        self.key = key
        self.id = digest(key)
        self.final = Path(root) / key["stage"] / self.id
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.temp = Path(tempfile.mkdtemp(prefix=f".{self.id}.", dir=self.final.parent))
        self.chunks = []
        self.published = False

    def append(self, **arrays: np.ndarray):
        if self.published or not arrays or any(a.dtype.hasobject for a in arrays.values()):
            raise ValueError("invalid cache append")
        name = f"chunk-{len(self.chunks):06d}.npz"
        path = self.temp / name
        with path.open("wb") as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        self.chunks.append({"file": name, "sha256": file_hash(path),
                            "arrays": {n: {"shape": list(a.shape), "dtype": a.dtype.str}
                                       for n, a in arrays.items()}})

    def commit(self, metadata: dict, *, validate=None) -> Cache:
        manifest = {
            "format": "ez2rc.stage-cache", "schema_version": SCHEMA_VERSION,
            "status": "complete", "cache_id": self.id, "key": self.key,
            "metadata": metadata, "chunks": self.chunks,
        }
        manifest["manifest_sha256"] = digest(manifest)
        write_json(self.temp / "manifest.json", manifest)
        Cache(self.temp, self.key)
        if validate is not None:
            validate(self.temp)
        if self.final.exists():
            # A valid concurrent publisher wins. Corrupt caches stay reviewable.
            try:
                existing = Cache(self.final, self.key)
            except CacheError:
                quarantine = self.final.with_name(f".{self.id}.corrupt-{self.temp.name.rsplit('.', 1)[-1]}")
                self.final.rename(quarantine)
            else:
                self.published = True
                return existing
        try:
            self.temp.rename(self.final)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            return Cache(self.final, self.key)
        self.published = True
        return Cache(self.final, self.key)

    def __exit__(self, *args):
        if self.temp.exists():
            shutil.rmtree(self.temp)


def safe_component(value: str) -> str:
    value = re.sub(r"[^\w. -]", "_", value, flags=re.UNICODE).strip(" .")
    return value or "untitled"
