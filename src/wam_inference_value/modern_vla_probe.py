from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .evaluation import results_dir, write_json


PACKAGE_CANDIDATES = [
    "torch",
    "transformers",
    "huggingface_hub",
    "lerobot",
    "openvla",
    "octo",
    "openpi",
    "gr00t",
]

LOCAL_NAME_TOKENS = (
    "openvla",
    "vla",
    "octo",
    "rt-1",
    "rt1",
    "rt-2",
    "rt2",
    "openpi",
    "gr00t",
    "checkpoint",
    "ckpt",
)

HF_MODEL_CANDIDATES = [
    "openvla/openvla-7b",
    "openvla/openvla-7b-finetuned-libero-spatial",
    "openvla/openvla-7b-finetuned-libero-object",
    "physical-intelligence/fast",
    "nvidia/GR00T-N1.5-3B",
]


def _package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {
        "name": name,
        "importable": spec is not None,
        "origin": str(spec.origin) if spec is not None and spec.origin else None,
    }


def _safe_local_matches(root_paths: list[Path], *, max_matches: int = 80, max_dirs: int = 5_000) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    skipped_names = {".git", ".venv", ".venvs", "__pycache__", "site-packages", "AUDIT_best-of-n-wam"}
    visited_dirs = 0
    for root in root_paths:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            visited_dirs += 1
            if visited_dirs > max_dirs or len(matches) >= max_matches:
                return matches
            dirnames[:] = [name for name in dirnames if name not in skipped_names]
            for name in [*dirnames, *filenames]:
                lower = name.lower()
                if not any(token in lower for token in LOCAL_NAME_TOKENS):
                    continue
                path = Path(dirpath) / name
                matches.append({"path": str(path), "kind": "directory" if path.is_dir() else "file"})
                if len(matches) >= max_matches:
                    return matches
    return matches


def _secret_presence(paths: list[Path]) -> dict[str, Any]:
    env_names = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME"]
    env_present = {name: bool(os.environ.get(name)) for name in env_names}
    secret_like_file_count = 0
    for base in paths:
        if not base.exists() or not base.is_dir():
            continue
        for path in base.rglob("*"):
            if secret_like_file_count >= 40:
                break
            lower = path.name.lower()
            if path.is_file() and any(token in lower for token in ("hf", "huggingface", "token", "key")):
                secret_like_file_count += 1
    return {
        "env_present": env_present,
        "secret_like_file_count_capped": secret_like_file_count,
        "secret_like_file_paths_redacted": True,
        "tokens_redacted": True,
    }


def _hf_model_probe(repo_id: str, *, timeout_s: float) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repo_id}"
    started = time.time()
    request = Request(url, headers={"User-Agent": "wam-inference-value-vla-probe"})
    try:
        with urlopen(request, timeout=float(timeout_s)) as response:
            raw = response.read(512_000)
        payload = json.loads(raw.decode("utf-8"))
        siblings = payload.get("siblings") if isinstance(payload.get("siblings"), list) else []
        return {
            "repo_id": repo_id,
            "reachable": True,
            "private": bool(payload.get("private", False)),
            "sha_present": bool(payload.get("sha")),
            "pipeline_tag": payload.get("pipeline_tag"),
            "library_name": payload.get("library_name"),
            "sibling_count": len(siblings),
            "elapsed_seconds": float(time.time() - started),
        }
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "reachable": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
            "elapsed_seconds": float(time.time() - started),
        }


def run_modern_vla_availability_probe(
    root: Path,
    *,
    output_results_dir: Path | None = None,
    probe_hf: bool = True,
    timeout_s: float = 10.0,
    scan_user_roots: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    output_results_dir = (output_results_dir or results_dir()).resolve()
    user_root = Path.home()
    external_root = user_root / "external_benchmarks"
    mirror_root = user_root / "MIRROR"
    local_roots = [root, external_root, mirror_root] if scan_user_roots else [root]
    packages = [_package_status(name) for name in PACKAGE_CANDIDATES]
    local_matches = _safe_local_matches(local_roots)
    hf_models = [_hf_model_probe(repo_id, timeout_s=timeout_s) for repo_id in HF_MODEL_CANDIDATES] if probe_hf else []
    secret_status = _secret_presence([mirror_root, user_root / ".cache" / "huggingface"])
    runtime_probe_path = output_results_dir / "external_benchmark_runtime_probe.json"
    runtime_probe: dict[str, Any] = {}
    if runtime_probe_path.exists():
        try:
            loaded = json.loads(runtime_probe_path.read_text(encoding="utf-8"))
            runtime_probe = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            runtime_probe = {}
    pretrained_probe_path = output_results_dir / "modern_vla_pretrained_load_probe.json"
    pretrained_probe: dict[str, Any] = {}
    if pretrained_probe_path.exists():
        try:
            loaded = json.loads(pretrained_probe_path.read_text(encoding="utf-8"))
            pretrained_probe = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            pretrained_probe = {}

    importable = {row["name"] for row in packages if row["importable"]}
    vla_package_importable = bool(importable.intersection({"openvla", "octo", "openpi", "gr00t", "lerobot"}))
    local_vla_like = [
        row
        for row in local_matches
        if any(token in Path(row["path"]).name.lower() for token in ("openvla", "octo", "openpi", "gr00t"))
    ]
    reachable_hf = [row for row in hf_models if row.get("reachable")]
    vla_joint_runtime_available = runtime_probe.get("vla_libero_joint_runtime_available") is True
    pretrained_vla_loaded = pretrained_probe.get("verified") is True and pretrained_probe.get("pretrained_vla_loaded") is True
    ready_for_policy_eval = bool((vla_package_importable and local_vla_like) or vla_joint_runtime_available)

    payload = {
        "experiment": "modern_vla_availability_probe",
        "verified": True,
        "python": sys.version.split()[0],
        "probe_hf": bool(probe_hf),
        "scan_user_roots": bool(scan_user_roots),
        "packages": packages,
        "vla_package_importable": vla_package_importable,
        "local_match_count": len(local_matches),
        "local_vla_like_count": len(local_vla_like),
        "local_matches": local_matches,
        "hf_models": hf_models,
        "hf_reachable_count": len(reachable_hf),
        "secret_status": secret_status,
        "joint_runtime_probe_present": bool(runtime_probe),
        "vla_libero_joint_runtime_available": vla_joint_runtime_available,
        "vla_runtime_success": runtime_probe.get("vla_runtime_success"),
        "pretrained_load_probe_present": bool(pretrained_probe),
        "pretrained_vla_loaded": pretrained_vla_loaded,
        "pretrained_vla_parameter_count": pretrained_probe.get("parameter_count"),
        "pretrained_vla_model_id": pretrained_probe.get("model_id"),
        "ready_for_policy_eval": ready_for_policy_eval,
        "missing_for_ideal_claim": [
            item
            for item, ok in [
                ("runnable modern VLA policy package", vla_package_importable or vla_joint_runtime_available),
                ("local VLA policy repository or joint runtime", bool(local_vla_like) or vla_joint_runtime_available),
                ("pretrained VLA weights loaded", pretrained_vla_loaded),
                ("LIBERO-compatible sparse-success VLA evaluation artifact", False),
            ]
            if not ok
        ],
        "note": "Availability probe only. Runtime/package availability, public model metadata, or secret presence is not policy validation and does not support a modern VLA LIBERO result without a heldout evaluation artifact.",
    }
    write_json(output_results_dir / "modern_vla_availability_probe.json", payload)
    return payload


def modern_vla_availability_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Modern VLA Availability Probe",
        "",
        f"- verified audit: `{payload.get('verified')}`",
        f"- VLA package importable: `{payload.get('vla_package_importable')}`",
        f"- local VLA-like matches: `{payload.get('local_vla_like_count')}`",
        f"- Hugging Face models reachable: `{payload.get('hf_reachable_count')}`",
        f"- joint LIBERO+VLA runtime available: `{payload.get('vla_libero_joint_runtime_available')}`",
        f"- pretrained VLA loaded: `{payload.get('pretrained_vla_loaded')}`",
        f"- pretrained VLA parameters: `{payload.get('pretrained_vla_parameter_count')}`",
        f"- ready for policy eval: `{payload.get('ready_for_policy_eval')}`",
        "",
        "## Missing For Ideal Claim",
        "",
    ]
    for item in payload.get("missing_for_ideal_claim") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Importable Packages", ""])
    for row in payload.get("packages") or []:
        lines.append(f"- `{row.get('name')}`: importable=`{row.get('importable')}`")
    lines.extend(["", "## Hugging Face Metadata Probe", ""])
    for row in payload.get("hf_models") or []:
        if row.get("reachable"):
            lines.append(
                f"- `{row.get('repo_id')}`: reachable=`True`, private=`{row.get('private')}`, files=`{row.get('sibling_count')}`"
            )
        else:
            lines.append(f"- `{row.get('repo_id')}`: reachable=`False`, error=`{row.get('error_type')}`")
    lines.extend(
        [
            "",
            "This is an availability/blocker artifact only. It does not download checkpoints, expose secrets, or validate a modern VLA policy.",
        ]
    )
    return "\n".join(lines) + "\n"
