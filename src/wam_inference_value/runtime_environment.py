from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORE_REQUIREMENT_FILES = ("requirements.txt", "pyproject.toml")
OPTIONAL_REQUIREMENT_FILES = ("requirements-benchmark.txt",)
COMMAND_PROBES = ("python", "python3", "py", "bash", "git")
MODULE_PROBES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
    "gymnasium": "gymnasium",
    "gymnasium_robotics": "gymnasium_robotics",
    "mani_skill": "mani_skill",
    "metaworld": "metaworld",
    "robosuite": "robosuite",
    "imageio": "imageio",
    "sklearn": "sklearn",
    "scipy": "scipy",
    "torch": "torch",
}
REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(?:>=\s*([A-Za-z0-9_.!+~-]+))?")


@dataclass(frozen=True)
class RuntimeEnvironmentCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[RuntimeEnvironmentCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(RuntimeEnvironmentCheck(name=name, ok=bool(ok), detail=detail))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_requirement_line(line: str) -> dict[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = REQUIREMENT_RE.match(stripped)
    if not match:
        return None
    name, minimum = match.groups()
    return {"raw": stripped, "name": name, "minimum": minimum or ""}


def read_requirements(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return [parsed for line in path.read_text(encoding="utf-8").splitlines() if (parsed := parse_requirement_line(line))]


def version_tuple(raw: str) -> tuple[int, ...]:
    values = [int(part) for part in re.findall(r"\d+", raw)]
    return tuple(values[:4]) if values else (0,)


def version_satisfies_minimum(installed: str | None, minimum: str) -> bool:
    if not minimum:
        return installed is not None
    if installed is None:
        return False
    left = version_tuple(installed)
    right = version_tuple(minimum)
    size = max(len(left), len(right))
    left = left + (0,) * (size - len(left))
    right = right + (0,) * (size - len(right))
    return left >= right


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_record(requirement: dict[str, str], *, optional: bool) -> dict[str, Any]:
    name = requirement["name"]
    version = package_version(name)
    minimum = requirement.get("minimum") or ""
    return {
        "name": name,
        "raw": requirement.get("raw") or name,
        "minimum": minimum,
        "optional": optional,
        "installed": version is not None,
        "version": version,
        "satisfies_minimum": version_satisfies_minimum(version, minimum),
    }


def requirement_file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    record: dict[str, Any] = {"path": relative_path, "exists": path.exists()}
    if path.exists():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
        record["requirements"] = read_requirements(path) if path.suffix == ".txt" else []
    else:
        record["bytes"] = 0
        record["sha256"] = None
        record["requirements"] = []
    return record


def pyproject_dependencies(root: Path) -> list[dict[str, str]]:
    path = root / "pyproject.toml"
    if not path.exists():
        return []
    try:
        import tomllib

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    requirements: list[dict[str, str]] = []
    project = payload.get("project") or {}
    for raw in project.get("dependencies") or []:
        parsed = parse_requirement_line(str(raw))
        if parsed:
            requirements.append(parsed)
    optional = project.get("optional-dependencies") or {}
    for raw in optional.get("test") or []:
        parsed = parse_requirement_line(str(raw))
        if parsed:
            requirements.append(parsed)
    return requirements


def dedupe_requirements(requirements: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for requirement in requirements:
        key = requirement["name"].lower()
        current = by_name.get(key)
        if current is None or version_tuple(requirement.get("minimum") or "0") > version_tuple(current.get("minimum") or "0"):
            by_name[key] = requirement
    return sorted(by_name.values(), key=lambda item: item["name"].lower())


def build_runtime_environment(root: Path) -> dict[str, Any]:
    root = root.resolve()
    core_files = [requirement_file_record(root, path) for path in CORE_REQUIREMENT_FILES]
    optional_files = [requirement_file_record(root, path) for path in OPTIONAL_REQUIREMENT_FILES]
    core_requirements = []
    for record in core_files:
        core_requirements.extend(record.get("requirements") or [])
    core_requirements.extend(pyproject_dependencies(root))
    optional_requirements = []
    for record in optional_files:
        optional_requirements.extend(record.get("requirements") or [])

    core_packages = [package_record(requirement, optional=False) for requirement in dedupe_requirements(core_requirements)]
    optional_packages = [package_record(requirement, optional=True) for requirement in dedupe_requirements(optional_requirements)]
    module_probes = {
        name: {"module": module, "available": importlib.util.find_spec(module) is not None}
        for name, module in sorted(MODULE_PROBES.items())
    }
    command_probes = {command: shutil.which(command) for command in COMMAND_PROBES}
    return {
        "experiment": "runtime_environment",
        "schema_version": 1,
        "root": str(root),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "requirement_files": core_files + optional_files,
        "core_packages": core_packages,
        "optional_packages": optional_packages,
        "module_probes": module_probes,
        "command_probes": command_probes,
        "n_requirement_files": len(core_files + optional_files),
        "n_core_requirements": len(core_packages),
        "n_optional_requirements": len(optional_packages),
        "n_optional_available": sum(1 for package in optional_packages if package.get("installed")),
        "n_module_probes": len(module_probes),
        "n_command_probes": len(command_probes),
    }


def audit_runtime_environment_payload(payload: dict[str, Any], *, root: Path) -> dict[str, Any]:
    root = root.resolve()
    checks: list[RuntimeEnvironmentCheck] = []
    current = build_runtime_environment(root)
    python_info = payload.get("python") or {}
    platform_info = payload.get("platform") or {}
    core_packages = payload.get("core_packages") or []
    optional_packages = payload.get("optional_packages") or []
    requirement_files = payload.get("requirement_files") or []
    module_probes = payload.get("module_probes") or {}
    command_probes = payload.get("command_probes") or {}
    core_missing = [package.get("name") for package in core_packages if not package.get("installed")]
    core_version_issues = [
        package.get("name")
        for package in core_packages
        if package.get("installed") and not package.get("satisfies_minimum")
    ]

    current_requirement_by_path = {record["path"]: record for record in current["requirement_files"]}
    requirement_hash_mismatches = []
    missing_requirement_files = []
    for record in requirement_files:
        path = record.get("path")
        current_record = current_requirement_by_path.get(path)
        if not current_record or not current_record.get("exists"):
            missing_requirement_files.append(path)
            continue
        if record.get("sha256") != current_record.get("sha256"):
            requirement_hash_mismatches.append(path)

    current_core_by_name = {package["name"].lower(): package for package in current["core_packages"]}
    package_version_mismatches = []
    for package in core_packages:
        current_package = current_core_by_name.get(str(package.get("name", "")).lower())
        if current_package is None:
            package_version_mismatches.append(package.get("name"))
        elif package.get("version") != current_package.get("version"):
            package_version_mismatches.append(package.get("name"))

    add(checks, "runtime_manifest_present_fields", bool(python_info) and bool(platform_info), f"python={bool(python_info)}, platform={bool(platform_info)}")
    add(checks, "runtime_python_version_supported", tuple((python_info.get("version_info") or [0, 0])[:2]) >= (3, 10), f"version_info={python_info.get('version_info')}")
    add(checks, "runtime_python_matches_current", python_info == current["python"], f"payload={python_info}, current={current['python']}")
    add(checks, "runtime_platform_recorded", all(platform_info.get(key) is not None for key in ("system", "release", "machine")), f"platform={platform_info}")
    add(checks, "runtime_requirement_files_present", not missing_requirement_files, f"missing={missing_requirement_files}")
    add(checks, "runtime_requirement_hashes_match", not requirement_hash_mismatches, f"mismatches={requirement_hash_mismatches}")
    add(checks, "runtime_core_requirements_recorded", len(core_packages) >= 4, f"core={len(core_packages)}")
    add(checks, "runtime_core_packages_installed", not core_missing, f"missing={core_missing}")
    add(checks, "runtime_core_versions_satisfy_minimums", not core_version_issues, f"issues={core_version_issues}")
    add(checks, "runtime_core_package_versions_match_current", not package_version_mismatches, f"mismatches={package_version_mismatches}")
    add(checks, "runtime_optional_requirements_status_recorded", len(optional_packages) >= 4, f"optional={len(optional_packages)}")
    add(checks, "runtime_module_probes_recorded", len(module_probes) >= 10, f"modules={len(module_probes)}")
    add(checks, "runtime_command_probes_recorded", len(command_probes) >= 5, f"commands={len(command_probes)}")
    add(checks, "runtime_current_python_executable_exists", Path(str(python_info.get("executable") or "")).exists(), f"executable={python_info.get('executable')}")
    add(checks, "runtime_required_commands_available", bool(command_probes.get("bash")) and bool(command_probes.get("git")), f"commands={command_probes}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "runtime_environment",
        "verified": len(issues) == 0,
        "root": str(root),
        "n_requirement_files": len(requirement_files),
        "n_core_requirements": len(core_packages),
        "n_optional_requirements": len(optional_packages),
        "n_optional_available": sum(1 for package in optional_packages if package.get("installed")),
        "n_core_missing": len(core_missing),
        "n_core_version_issues": len(core_version_issues),
        "n_module_probes": len(module_probes),
        "n_command_probes": len(command_probes),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "core_missing": core_missing,
        "core_version_issues": core_version_issues,
        "requirement_hash_mismatches": requirement_hash_mismatches,
        "missing_requirement_files": missing_requirement_files,
        "package_version_mismatches": package_version_mismatches,
    }


def build_and_audit_runtime_environment(root: Path) -> dict[str, Any]:
    manifest = build_runtime_environment(root)
    audit = audit_runtime_environment_payload(manifest, root=root)
    return {**manifest, **audit}


def runtime_environment_markdown(payload: dict[str, Any]) -> str:
    python_info = payload.get("python") or {}
    lines = [
        "# Runtime Environment Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Python: {python_info.get('version')} at `{python_info.get('executable')}`",
        f"- Platform: {(payload.get('platform') or {}).get('system')} {(payload.get('platform') or {}).get('release')} {(payload.get('platform') or {}).get('machine')}",
        f"- Requirement files: {payload.get('n_requirement_files')}",
        f"- Core requirements: {payload.get('n_core_requirements')}",
        f"- Optional requirements available: {payload.get('n_optional_available')} / {payload.get('n_optional_requirements')}",
        f"- Module probes: {payload.get('n_module_probes')}",
        f"- Command probes: {payload.get('n_command_probes')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Python, platform, requirement-file hashes, core package versions, optional package availability, module probes, and command probes are recorded and verified for the current runtime.")
    lines.append("")
    return "\n".join(lines)
