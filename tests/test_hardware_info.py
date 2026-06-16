"""HardwareInfo auto-detection tests."""
from __future__ import annotations
from records import HardwareInfo


def test_defaults_populate_cpu_and_ram():
    h = HardwareInfo()
    assert h.cpu != ""
    assert h.ram_gb > 0


def test_defaults_populate_gpu_and_backend():
    h = HardwareInfo()
    # gpu is one of: actual device name, "none", or MPS string
    assert h.gpu in ("none",) or len(h.gpu) > 0
    assert h.backend in ("cuda", "cpu", "mps")


def test_precision_default_is_fp32():
    h = HardwareInfo()
    assert h.precision == "fp32"


def test_energy_efficiency_optional():
    h = HardwareInfo()
    # When pynvml absent or no CUDA, energy_efficiency_w must be None
    assert h.energy_efficiency_w is None or isinstance(h.energy_efficiency_w, float)


def test_explicit_overrides_respected():
    h = HardwareInfo(precision="fp16", backend="cuda")
    assert h.precision == "fp16"
    assert h.backend == "cuda"
