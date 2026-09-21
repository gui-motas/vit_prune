"""The layering rule that keeps the library installable on its own.

vitprune.core is the method. It must depend on nothing else in the package, so that
someone who wants only the method does not also pull in a video decoder, a YAML
parser and a subprocess driver.

That rule is easy to break by accident -- a single convenient import is enough. This
test makes the breakage visible immediately rather than at packaging time.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "vitprune"

#: Which subpackage may import which. A layer may always import itself.
ALLOWED = {
    "core": set(),
    "bench": {"core"},
    "pipeline": {"core", "bench"},
}


def _imports_of(path: Path) -> set[str]:
    """Subpackages of vitprune that a module imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "vitprune" and len(parts) > 1:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "vitprune" and len(parts) > 1:
                    found.add(parts[1])
    return found


def _modules_of(layer: str) -> list[Path]:
    return sorted((SRC / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_the_layer_has_modules(layer):
    assert _modules_of(layer), f"no modules found in {layer}"


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_the_layer_only_imports_what_it_may(layer):
    permitted = ALLOWED[layer] | {layer}
    offences = []

    for module in _modules_of(layer):
        forbidden = _imports_of(module) - permitted
        if forbidden:
            offences.append(f"{module.relative_to(SRC)} imports {sorted(forbidden)}")

    assert not offences, "layering violated:\n" + "\n".join(offences)


def test_core_imports_nothing_from_the_rest():
    """Stated separately because it is the rule that matters: core has to stand
    alone for the library to be installable without the extras."""
    for module in _modules_of("core"):
        assert _imports_of(module) <= {"core"}, f"{module.name} reaches outside core"


def test_core_does_not_import_the_optional_dependencies():
    """opencv and yaml belong to the pipeline extra. Importing either from core
    would make a minimal install fail at import time."""
    optional = {"cv2", "yaml", "decord"}

    for module in _modules_of("core"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not (set(names) & optional), f"{module.name} imports {set(names) & optional}"


def test_the_cost_model_does_not_require_torch():
    """cost and hardware have to work in an environment without torch, so that an
    estimate is available before any machine is booked."""
    for name in ("cost.py", "hardware.py"):
        tree = ast.parse((SRC / "bench" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name != "torch" for a in node.names), f"{name} imports torch"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("torch"), f"{name} imports torch"
