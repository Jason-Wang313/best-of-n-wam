from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from .evaluation import results_dir, write_json


PACKAGE_CANDIDATES = [
    "rclpy",
    "rospy",
    "roslibpy",
    "moveit_commander",
    "serial",
    "pyrealsense2",
    "cv2",
    "rtde_control",
    "rtde_receive",
    "xarm",
    "pymodbus",
]

COMMAND_CANDIDATES = [
    "ros2",
    "roscore",
    "rostopic",
    "roslaunch",
    "colcon",
    "franka_control",
    "xacro",
]

ROBOT_ENV_TOKENS = (
    "ROS",
    "ROBOT",
    "REAL_ROBOT",
    "HIL",
    "FRANKA",
    "KINOVA",
    "KUKA",
    "XARM",
    "UR_",
    "ABB",
    "DOBOT",
    "REALSENSE",
)

DEVICE_NAME_TOKENS = (
    "robot",
    "franka",
    "panda",
    "kinova",
    "kuka",
    "xarm",
    "universal robot",
    "ur ",
    "realsense",
    "lidar",
    "arduino",
    "teensy",
    "stm",
    "usb serial",
    "serial",
)

TRIAL_NAME_TOKENS = ("real_robot", "hardware", "hil")
TRIAL_EVIDENCE_TOKENS = ("trial", "episode", "success", "metric", "rollout", "log")
NON_TRIAL_TOKENS = ("probe", "blocker", "availability", "readiness", "audit", "status")
TRIAL_METRIC_KEYS = ("success", "utility", "reward", "trial", "episode", "rollout", "timestamp")


def _package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {
        "name": name,
        "importable": spec is not None,
        "origin": str(spec.origin) if spec is not None and spec.origin else None,
    }


def _command_status(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {"name": name, "available": path is not None, "path_present": bool(path)}


def _redacted_robot_env() -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name in sorted(os.environ):
        upper = name.upper()
        if any(token in upper for token in ROBOT_ENV_TOKENS):
            out[name] = True
    return out


def _serial_ports() -> list[dict[str, Any]]:
    try:
        from serial.tools import list_ports  # type: ignore
    except Exception:
        return []
    ports = []
    for port in list_ports.comports():
        ports.append(
            {
                "device": str(getattr(port, "device", "")),
                "description": str(getattr(port, "description", "")),
                "manufacturer": str(getattr(port, "manufacturer", "") or ""),
            }
        )
    return ports[:50]


def _windows_pnp_candidates(timeout_s: float = 8.0) -> list[dict[str, Any]]:
    if not sys.platform.startswith("win"):
        return []
    command = (
        "Get-CimInstance Win32_PnPEntity | "
        "Select-Object Name,PNPClass,Status | "
        "ConvertTo-Json -Depth 2"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "")
        lower = name.lower()
        if not any(token in lower for token in DEVICE_NAME_TOKENS):
            continue
        candidates.append(
            {
                "name": name[:160],
                "pnp_class": str(row.get("PNPClass") or "")[:80],
                "status": str(row.get("Status") or "")[:80],
            }
        )
        if len(candidates) >= 50:
            break
    return candidates


def physical_trial_metric_artifacts(root_results: Path) -> list[str]:
    out: list[str] = []
    for path in sorted(root_results.glob("*")):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if not any(token in lower for token in TRIAL_NAME_TOKENS):
            continue
        if any(token in lower for token in NON_TRIAL_TOKENS):
            continue
        if not any(token in lower for token in TRIAL_EVIDENCE_TOKENS):
            continue
        if not _file_has_trial_metric_content(path):
            continue
        out.append(str(path))
    return out


def _file_has_trial_metric_content(path: Path) -> bool:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(payload).lower()
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
    except Exception:
        return False
    return any(key in text for key in TRIAL_METRIC_KEYS)


def run_real_robot_hil_probe(
    root: Path,
    *,
    output_results_dir: Path | None = None,
    inspect_hardware: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    output_results_dir = (output_results_dir or results_dir()).resolve()
    started = time.time()
    packages = [_package_status(name) for name in PACKAGE_CANDIDATES]
    commands = [_command_status(name) for name in COMMAND_CANDIDATES]
    env_present = _redacted_robot_env()
    serial_ports = _serial_ports() if inspect_hardware else []
    pnp_candidates = _windows_pnp_candidates() if inspect_hardware else []
    trial_artifacts = physical_trial_metric_artifacts(output_results_dir)
    possible_hardware_count = len(serial_ports) + len(pnp_candidates)
    payload = {
        "experiment": "real_robot_hil_probe",
        "verified": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "inspect_hardware": bool(inspect_hardware),
        "packages": packages,
        "commands": commands,
        "robot_env_names_present": env_present,
        "env_values_redacted": True,
        "serial_ports": serial_ports,
        "windows_pnp_candidates": pnp_candidates,
        "possible_hardware_device_count": possible_hardware_count,
        "trial_metric_artifact_count": len(trial_artifacts),
        "trial_metric_artifacts": trial_artifacts,
        "real_robot_or_hil_claim_ready": bool(trial_artifacts),
        "elapsed_seconds": float(time.time() - started),
        "note": "Availability probe only. Device/package presence is not real-robot or HIL validation without physical trial metrics.",
    }
    write_json(output_results_dir / "real_robot_hil_probe.json", payload)
    return payload


def real_robot_hil_probe_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Real Robot / HIL Availability Probe",
        "",
        f"- verified audit: `{payload.get('verified')}`",
        f"- possible hardware device count: `{payload.get('possible_hardware_device_count')}`",
        f"- physical trial metric artifacts: `{payload.get('trial_metric_artifact_count')}`",
        f"- claim ready: `{payload.get('real_robot_or_hil_claim_ready')}`",
        "",
        "## Importable Packages",
        "",
    ]
    for row in payload.get("packages") or []:
        lines.append(f"- `{row.get('name')}`: importable=`{row.get('importable')}`")
    lines.extend(["", "## Commands", ""])
    for row in payload.get("commands") or []:
        lines.append(f"- `{row.get('name')}`: available=`{row.get('available')}`")
    lines.extend(
        [
            "",
            "Environment variable values are redacted. Hardware-like device names, if any, are only availability hints and are not validation trials.",
            "This probe does not support a real-robot/HIL claim unless paired with physical trial logs and success or utility metrics.",
            "",
        ]
    )
    return "\n".join(lines)
