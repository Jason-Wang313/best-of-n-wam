from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wam_inference_value.evaluation import results_dir, write_json  # noqa: E402
from wam_inference_value.ideal_frontier_readiness import MODERN_VLA_MIN_PARAMETERS  # noqa: E402


CHILD_CODE = r"""
import json
import os
import tempfile
import time
import traceback

started = time.time()
try:
    import torch
    import lerobot.configs.policies as policies
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    original_load_state_dict = torch.nn.Module.load_state_dict

    def assign_load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return original_load_state_dict(self, state_dict, strict=strict, assign=True)
        except TypeError:
            return original_load_state_dict(self, state_dict, strict=strict)

    torch.nn.Module.load_state_dict = assign_load_state_dict

    class ReopenableNamedTemporaryFile:
        def __init__(self, mode="w+", *args, **kwargs):
            fd, self.name = tempfile.mkstemp()
            os.close(fd)
            self._file = open(self.name, mode, encoding=kwargs.get("encoding") or None)

        def __enter__(self):
            return self._file

        def __exit__(self, exc_type, exc, tb):
            try:
                self._file.close()
            finally:
                try:
                    os.unlink(self.name)
                except OSError:
                    pass
            return False

    # LeRobot 0.3.3 reopens a NamedTemporaryFile while it is still open, which
    # fails on Windows. This preserves the same config contents but makes the
    # temporary file reopenable for the downstream draccus parser.
    policies.tempfile.NamedTemporaryFile = ReopenableNamedTemporaryFile

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model_id = os.environ.get("WAM_VLA_MODEL_ID", "HuggingFaceVLA/smolvla_libero")
    policy = SmolVLAPolicy.from_pretrained(model_id)
    processor_stats_loaded = {"preprocessor": False, "postprocessor": False}
    try:
        pre_path = hf_hub_download(
            repo_id=model_id,
            filename="policy_preprocessor_step_5_normalizer_processor.safetensors",
        )
        pre = load_file(pre_path, device="cpu")
        processor_stats_loaded["preprocessor"] = bool(pre)
    except Exception:
        pass
    for filename in (
        "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        try:
            post_path = hf_hub_download(repo_id=model_id, filename=filename)
            post = load_file(post_path, device="cpu")
            processor_stats_loaded["postprocessor"] = bool(post)
            processor_stats_loaded["postprocessor_file"] = filename
            break
        except Exception:
            continue
    first_param = next(policy.parameters())
    parameter_count = sum(p.numel() for p in policy.parameters())
    print(json.dumps({
        "ok": True,
        "loaded": True,
        "model_id": model_id,
        "policy_class": type(policy).__name__,
        "parameter_count": int(parameter_count),
        "processor_stats_loaded": processor_stats_loaded,
        "device": str(first_param.device),
        "elapsed_seconds": float(time.time() - started),
    }))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "loaded": False,
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
        "trace_tail": traceback.format_exc().splitlines()[-16:],
        "elapsed_seconds": float(time.time() - started),
    }))
"""


def _default_robocasa_python() -> Path:
    return ROOT.parent / "external_benchmarks" / ".venvs" / "robocasa" / "Scripts" / "python.exe"


def _default_libero_source() -> Path:
    return ROOT.parent / "external_benchmarks" / "LIBERO"


def _default_libero_config() -> Path:
    return ROOT.parent / "external_benchmarks" / ".libero"


def _run_child(
    python_path: Path,
    libero_source: Path,
    libero_config: Path,
    model_id: str,
    timeout_s: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(libero_source) if not env.get("PYTHONPATH") else f"{libero_source}{os.pathsep}{env['PYTHONPATH']}"
    env["LIBERO_CONFIG_PATH"] = str(libero_config)
    env["WAM_VLA_MODEL_ID"] = str(model_id)
    proc = subprocess.run(
        [str(python_path), "-c", CHILD_CODE],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=int(timeout_s),
        check=False,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        child_payload = json.loads(stdout_lines[-1]) if stdout_lines else {}
    except json.JSONDecodeError:
        child_payload = {"ok": False, "loaded": False, "error_type": "NonJsonOutput", "error": proc.stdout[-2000:]}
    return {
        "returncode": proc.returncode,
        "stdout_tail": stdout_lines[-20:],
        "stderr_tail": proc.stderr.splitlines()[-30:],
        "child": child_payload if isinstance(child_payload, dict) else {},
    }


def build_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Modern VLA Pretrained Load Probe",
        "",
        f"- attempted: `{payload.get('attempted')}`",
        f"- verified load: `{payload.get('verified')}`",
        f"- model id: `{payload.get('model_id')}`",
        f"- policy class: `{payload.get('policy_class')}`",
        f"- parameter count: `{payload.get('parameter_count')}`",
        f"- processor stats loaded: `{payload.get('processor_stats_loaded')}`",
        f"- device: `{payload.get('device')}`",
        f"- runtime Python: `{payload.get('runtime_python')}`",
        f"- LIBERO source on PYTHONPATH: `{payload.get('libero_source')}`",
        f"- heldout LIBERO policy evaluation present: `{payload.get('heldout_libero_policy_eval')}`",
        "",
        "This is a pretrained-load prerequisite probe only. It does not validate sparse-success policy execution in LIBERO.",
        "",
        "## Child Process",
        "",
        f"- returncode: `{payload.get('child_returncode')}`",
        f"- error type: `{payload.get('error_type')}`",
        f"- error: `{payload.get('error')}`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = results_dir()
    python_path = Path(args.python).resolve()
    libero_source = Path(args.libero_source).resolve()
    libero_config = Path(args.libero_config).resolve()
    child = _run_child(python_path, libero_source, libero_config, args.model_id, args.timeout_s)
    child_payload = child.get("child") or {}
    parameter_count = int(child_payload.get("parameter_count") or 0)
    verified = (
        child.get("returncode") == 0
        and child_payload.get("ok") is True
        and child_payload.get("loaded") is True
        and parameter_count >= MODERN_VLA_MIN_PARAMETERS
    )
    payload = {
        "experiment": "modern_vla_pretrained_load_probe",
        "attempted": True,
        "verified": bool(verified),
        "scope": "pretrained load only; not a heldout LIBERO policy evaluation",
        "model_id": args.model_id,
        "runtime_python": str(python_path),
        "libero_source": str(libero_source),
        "libero_config": str(libero_config),
        "windows_tempfile_workaround_applied": True,
        "pretrained_vla_loaded": child_payload.get("loaded") is True,
        "pretrained_vla": child_payload.get("loaded") is True,
        "parameter_count": parameter_count,
        "modern_vla_min_parameters": MODERN_VLA_MIN_PARAMETERS,
        "policy_class": child_payload.get("policy_class"),
        "processor_stats_loaded": child_payload.get("processor_stats_loaded"),
        "device": child_payload.get("device"),
        "elapsed_seconds": child_payload.get("elapsed_seconds"),
        "heldout_libero_policy_eval": False,
        "child_returncode": child.get("returncode"),
        "error_type": child_payload.get("error_type"),
        "error": child_payload.get("error"),
        "trace_tail": child_payload.get("trace_tail"),
        "stdout_tail": child.get("stdout_tail"),
        "stderr_tail": child.get("stderr_tail"),
        "note": "This supports pretrained-model availability only. It is not evidence that SmolVLA solves LIBERO tasks in this project.",
    }
    write_json(output_dir / "modern_vla_pretrained_load_probe.json", payload)
    report_path = ROOT / "reports" / "modern_vla_pretrained_load_probe_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(payload), encoding="utf-8")
    payload["artifacts"] = {"report": str(report_path)}
    write_json(output_dir / "modern_vla_pretrained_load_probe.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(_default_robocasa_python()))
    parser.add_argument("--libero-source", default=str(_default_libero_source()))
    parser.add_argument("--libero-config", default=str(_default_libero_config()))
    parser.add_argument("--model-id", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(
        "modern VLA pretrained load probe: "
        f"verified={payload.get('verified')}, "
        f"loaded={payload.get('pretrained_vla_loaded')}, "
        f"params={payload.get('parameter_count')}, "
        f"heldout_eval={payload.get('heldout_libero_policy_eval')}"
    )
    if args.fail_on_error and not payload.get("verified"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
