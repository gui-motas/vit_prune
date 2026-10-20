# Local validation — 2026-10-19

Validated the library-contract changes on Windows with Python 3.13.2,
Torch 2.14.1+cpu, NumPy 2.5.4, pytest 9.1.1 and Ruff 0.17.1. Dependencies were
installed in the repository's `.venv`, not in the global Python environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[pipeline,dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```

Results: **315 tests passed**, Ruff passed, and no broken package requirements.
Relative Markdown links in the repository and internal `ViTPrune/` links in all
14 vault notes resolved.

A regular wheel was built with `pip wheel . --no-deps --no-build-isolation`,
installed into a separate temporary directory and imported in an isolated Python
subprocess outside the repository. With OpenCV, YAML and Decord imports explicitly
blocked, public imports, compatibility checking, reference encoding and the `cost`
command all succeeded. The test asserted that imports came from the wheel install,
not from the editable source checkout.

Regression coverage includes invalid plans and geometry, inference state and probe
restoration, exact reference shapes, nonlinear normalisation caveats, selection
matching the encoder under the same autocast policy, dense benchmark denominators,
extraction collisions/overwrites, artifact fingerprints and evaluation exit status.

## Not established by this validation

- Compatibility with a particular real VideoMAE checkpoint or external source tree.
- Actual video decoding or external localizer behaviour; those paths are mocked in tests.
- GPU kernels, mixed-precision CUDA stability, latency, memory or temporal quality.
- Support across the entire Python/Torch version ranges declared in `pyproject.toml`.
- Reproduction of the historical CPU experiment.

CUDA was unavailable. These are software-contract checks, not new experimental
results. Follow [experiments.md](experiments.md) on the target hardware and dataset.
