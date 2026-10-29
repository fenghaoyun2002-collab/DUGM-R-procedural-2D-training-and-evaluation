"""Configuration paths and local artifact manifests."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(name_or_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / path
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return data, path


def config_file(value: str | Path, *, parent: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"Cannot write {type(value).__name__} as JSON.")


def record_artifact(output: Path, name: str, artifact: Path, *, metadata: dict[str, Any] | None = None) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"artifacts": {}}
    manifest.setdefault("artifacts", {})[name] = {
        "path": str(artifact.resolve()), "sha256": sha256(artifact), **(metadata or {}),
    }
    write_json(manifest_path, manifest)
