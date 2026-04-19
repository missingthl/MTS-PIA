from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import torch.nn as nn


_MODULE_CACHE: dict[str, ModuleType] = {}


def get_time_series_library_reference_root() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "archive",
            "reference_code",
            "Time-Series-Library",
        )
    )


def ensure_time_series_library_reference_on_path() -> str:
    ref_root = get_time_series_library_reference_root()
    if not os.path.isdir(ref_root):
        raise FileNotFoundError(
            f"Time-Series-Library reference code not found at {ref_root}. "
            "Please clone https://github.com/thuml/Time-Series-Library there."
        )
    if ref_root not in sys.path:
        sys.path.insert(0, ref_root)
    _ensure_optional_dependency_stubs()
    return ref_root


def _ensure_optional_dependency_stubs() -> None:
    if "reformer_pytorch" not in sys.modules:
        try:
            import reformer_pytorch  # noqa: F401
        except ModuleNotFoundError:
            stub = ModuleType("reformer_pytorch")

            class LSHSelfAttention(nn.Module):
                """Compatibility stub for models that import Reformer support but never use it."""

                def __init__(self, *args, **kwargs) -> None:
                    super().__init__()

                def forward(self, x, **kwargs):
                    return x

            stub.LSHSelfAttention = LSHSelfAttention
            sys.modules["reformer_pytorch"] = stub


def _load_reference_module(*, module_key: str, relative_path: str) -> ModuleType:
    if module_key in _MODULE_CACHE:
        return _MODULE_CACHE[module_key]
    ref_root = ensure_time_series_library_reference_on_path()
    abs_path = os.path.join(ref_root, relative_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"reference module not found: {abs_path}")
    spec = importlib.util.spec_from_file_location(module_key, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module spec for {abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[module_key] = module
    return module


def get_patchtst_model_class():
    module = _load_reference_module(
        module_key="tsl_reference_patchtst",
        relative_path=os.path.join("models", "PatchTST.py"),
    )
    return module.Model


def get_timesnet_model_class():
    module = _load_reference_module(
        module_key="tsl_reference_timesnet",
        relative_path=os.path.join("models", "TimesNet.py"),
    )
    return module.Model


def build_patchtst_classification_config(
    *,
    seq_len: int,
    enc_in: int,
    num_class: int,
    d_model: int = 128,
    d_ff: int = 256,
    e_layers: int = 3,
    n_heads: int = 8,
    factor: int = 1,
    dropout: float = 0.1,
    activation: str = "gelu",
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="classification",
        seq_len=int(seq_len),
        pred_len=0,
        enc_in=int(enc_in),
        num_class=int(num_class),
        d_model=int(d_model),
        d_ff=int(d_ff),
        e_layers=int(e_layers),
        n_heads=int(n_heads),
        factor=int(factor),
        dropout=float(dropout),
        activation=str(activation),
    )


def build_timesnet_classification_config(
    *,
    seq_len: int,
    enc_in: int,
    num_class: int,
    d_model: int = 32,
    d_ff: int = 64,
    e_layers: int = 2,
    top_k: int = 3,
    num_kernels: int = 4,
    dropout: float = 0.1,
    embed: str = "fixed",
    freq: str = "h",
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="classification",
        seq_len=int(seq_len),
        label_len=0,
        pred_len=0,
        enc_in=int(enc_in),
        c_out=int(enc_in),
        num_class=int(num_class),
        d_model=int(d_model),
        d_ff=int(d_ff),
        e_layers=int(e_layers),
        top_k=int(top_k),
        num_kernels=int(num_kernels),
        dropout=float(dropout),
        embed=str(embed),
        freq=str(freq),
    )
