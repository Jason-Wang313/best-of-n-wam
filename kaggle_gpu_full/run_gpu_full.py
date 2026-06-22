from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_input_dir() -> Path:
    input_root = Path("/kaggle/input")
    expected = input_root / "frozen-visual-inference-probe-input"
    if expected.exists():
        return expected
    print("input root exists:", input_root.exists(), flush=True)
    if input_root.exists():
        for path in sorted(input_root.glob("*")):
            print("input child:", path, flush=True)
        for metadata_path in sorted(input_root.rglob("metadata.csv")):
            candidate = metadata_path.parent
            if (candidate / "frames.npz").exists() or (candidate / "frames.npy").exists():
                print("discovered input dataset:", candidate, flush=True)
                return candidate
    raise FileNotFoundError(f"could not find metadata.csv plus frames under {input_root}")


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    scratch = Path("/tmp") if Path("/tmp").exists() else working
    repo = scratch / "best-of-n-wam"
    input_dir = find_input_dir()
    output_dir = working / "frozen_visual_inference_probe_gpu"

    print("python:", sys.version, flush=True)
    run(["nvidia-smi"])
    print("using input_dir:", input_dir, flush=True)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "--no-cache-dir",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "transformers>=4.40",
            "safetensors",
            "pillow",
            "matplotlib",
            "pandas",
            "numpy",
        ]
    )
    run(
        [
            sys.executable,
            "-c",
            (
                "import torch, torchvision; "
                "print('torch:', torch.__version__, 'cuda:', torch.version.cuda, flush=True); "
                "print('torchvision:', torchvision.__version__, flush=True); "
                "print('cuda_available:', torch.cuda.is_available(), flush=True); "
                "print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None, flush=True); "
                "x=torch.ones((1,), device='cuda'); "
                "print('cuda_sanity:', float((x+1).detach().cpu().item()), flush=True); "
                "torch.cuda.synchronize()"
            ),
        ]
    )
    if repo.exists():
        shutil.rmtree(repo)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "codex/v6-evidence-hardening",
            "https://github.com/Jason-Wang313/best-of-n-wam.git",
            str(repo),
        ]
    )
    run(
        [
            sys.executable,
            "experiments/frozen_visual_inference_probe.py",
            "--precomputed-input-dir",
            str(input_dir),
            "--device",
            "auto",
            "--require-cuda",
            "--no-cpu-fallback",
            "--model-name",
            "openai/clip-vit-base-patch32",
            "--batch-size",
            "16",
        ],
        cwd=repo,
    )
    src = repo / "results" / "frozen_visual_inference_probe"
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    assert summary["available"] is True
    assert summary["verified"] is True
    assert summary["device"] == "cuda"
    assert summary["gpu_verified"] is True
    assert summary["runtime"]["selected_device"] == "cuda"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(src, output_dir)
    print((output_dir / "summary.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
