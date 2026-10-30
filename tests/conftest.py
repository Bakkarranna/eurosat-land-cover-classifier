"""Shared pytest configuration.

Keeps every test on the CPU, and makes ``landcover`` importable without an
install step so the suite runs from a fresh clone.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_and_determinism(monkeypatch):
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    torch.manual_seed(0)
    yield
