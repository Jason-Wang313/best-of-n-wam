from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    scratch = Path("/tmp") if Path("/tmp").exists() else working
    repo = scratch / "best-of-n-wam"

    print("python:", sys.version, flush=True)
    run(["nvidia-smi"])
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
    run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "transformers>=4.40", "safetensors", "pillow", "matplotlib", "pandas", "numpy"])
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
            "--gpu-preflight",
            "--device",
            "auto",
            "--model-name",
            "openai/clip-vit-base-patch32",
        ],
        cwd=repo,
    )
    src = repo / "results" / "frozen_visual_inference_probe" / "gpu_preflight.json"
    shutil.copy2(src, working / "gpu_preflight.json")
    print((working / "gpu_preflight.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
