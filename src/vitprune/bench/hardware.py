"""Accelerator characteristics used by the cost model.

These numbers are nominal: they come from vendor datasheets and describe a
theoretical ceiling, not what a card delivers in practice. They are here to produce
estimates and to order experiments, never to be reported as results.

Once wall-clock measurement is available it supersedes all of this. The gap between
what this module predicts and what the clock records is, in fact, the subject of
study -- not an error to be papered over.

Any card can be described from the command line without touching the code.
"""

from __future__ import annotations

from dataclasses import dataclass

TERA = 1e12
GIGA = 1e9


@dataclass(frozen=True)
class Accelerator:
    """A GPU, from the point of view of cost estimation.

    Attributes:
        name: a readable identifier.
        peak_flops: operations per second at peak, in reduced precision, without
            exploiting sparsity.
        bandwidth: bytes per second between memory and the compute units.
        efficiency: the fraction of peak typically reached. Half is a conservative
            and common assumption for this kind of model.
    """

    name: str
    peak_flops: float
    bandwidth: float
    efficiency: float = 0.5

    @property
    def usable_flops(self) -> float:
        return self.peak_flops * self.efficiency

    def describe(self) -> str:
        return (
            f"{self.name}: peak {self.peak_flops / TERA:.0f} TFLOP/s, "
            f"bandwidth {self.bandwidth / GIGA:.0f} GB/s, "
            f"assumed efficiency {self.efficiency:.0%}"
        )


#: Datasheet figures in reduced precision, dense (no sparsity).
CATALOG: dict[str, Accelerator] = {
    "a40": Accelerator("NVIDIA A40", peak_flops=149.7 * TERA, bandwidth=696 * GIGA),
    "l40s": Accelerator("NVIDIA L40S", peak_flops=362.0 * TERA, bandwidth=864 * GIGA),
    "a100-sxm": Accelerator("NVIDIA A100 SXM", peak_flops=312.0 * TERA, bandwidth=2039 * GIGA),
    "h100-pcie": Accelerator("NVIDIA H100 PCIe", peak_flops=756.0 * TERA, bandwidth=2000 * GIGA),
    "h100-sxm": Accelerator("NVIDIA H100 SXM", peak_flops=989.4 * TERA, bandwidth=3350 * GIGA),
}

DEFAULT_ACCELERATOR = "h100-sxm"


def accelerator(name: str = DEFAULT_ACCELERATOR) -> Accelerator:
    """Look up a card by its short name."""
    key = name.strip().lower()
    if key not in CATALOG:
        known = ", ".join(sorted(CATALOG))
        raise KeyError(f"unknown accelerator '{name}'. Known: {known}")
    return CATALOG[key]


def custom(
    name: str, peak_tflops: float, bandwidth_gbs: float, efficiency: float = 0.5
) -> Accelerator:
    """Describe a card that is not in the catalogue."""
    if peak_tflops <= 0 or bandwidth_gbs <= 0:
        raise ValueError("peak and bandwidth must be positive")
    if not 0 < efficiency <= 1:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    return Accelerator(name, peak_tflops * TERA, bandwidth_gbs * GIGA, efficiency)
