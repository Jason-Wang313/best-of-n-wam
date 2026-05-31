# LIBERO Blocker Report

Date: 2026-05-31.

## Status

Resolved for rollout-pool validation. LIBERO now has a verified optional artifact in `results/benchmark_libero_wam.json`.

The current artifact covers three `libero_spatial` tasks with a ridge state/action-sequence WAM-lite:

- train rollout samples: `192`
- validation rollout samples: `96`
- heldout eval rollout samples: `240`
- eval rollout pools: `15`
- exact-law utility MAE: `0.0001402348651464976`
- validation utility correlation: `0.3526483541014925`
- learned energy-regularized scorer minus random N8 CI lower: `0.2652641899613382`

This supports LIBERO rollout-pool dense-utility validation, not solved-task LIBERO policy performance.

## Historical Blocker

The first attempt used `python -m pip install libero -q` on the local Windows Python 3.10 environment. That failed before `libero`, `robosuite`, or `robomimic` became importable:

```text
Failed building wheel for hf-egl-probe / egl_probe
CMake Error: Unknown argument -j
subprocess.CalledProcessError: Command 'cmake ..; make -j' returned non-zero exit status 1.
```

After the failed install, import checks reported:

```text
libero False
robosuite False
robomimic False
egl_probe False
```

## Working Path

The successful path avoids the PyPI install and uses the official LIBERO source checkout with a separate runtime:

- source checkout: `C:\Users\wangz\external_benchmarks\LIBERO`
- LIBERO commit: `8f1084e`
- Python: `3.10.11` in `C:\Users\wangz\external_benchmarks\.venvs\libero310`
- key runtime packages: `robosuite==1.4.0`, `bddl==1.0.1`, `torch==2.4.1`, `mujoco==3.9.0`
- local Windows compatibility fix: copy the installed `mujoco.dll` into `robosuite\utils\mujoco.dll` and disable RoboSuite GPU rendering macros

The run command used:

```bash
LIBERO_PYTHON=/path/to/libero310/python \
LIBERO_SOURCE_PATH=/path/to/LIBERO \
LIBERO_CONFIG_PATH=/path/to/.libero \
bash scripts/run_benchmark_full.sh
```

## Remaining Limitation

LIBERO solved-task policy performance is still not claimed. The current result is an exact-law and learned-scorer rollout-pool artifact with dense progress utility.
