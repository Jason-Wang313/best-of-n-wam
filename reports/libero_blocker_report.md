# LIBERO Blocker Report

Date: 2026-05-30.

## Attempt

`python -m pip install libero -q` was attempted on the local Windows Python 3.10 environment.

## Outcome

The install failed before `libero`, `robosuite`, or `robomimic` became importable.

Observed blocker:

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

## Interpretation

No LIBERO benchmark validation claim is supported in this repo state. The README and paper outline must continue to list LIBERO as future work unless a successful install, task reset, rollout collection, WAM-lite training, and exact-law validation artifact are added.

## Next Commands To Try

LIBERO documentation recommends a separate conda environment with Python 3.8.13, installing its requirements, installing the matching Torch stack, installing robosuite, and then installing LIBERO from source. The PyPI package also documents LIBERO task creation through `libero.libero.benchmark` and `OffScreenRenderEnv`.
