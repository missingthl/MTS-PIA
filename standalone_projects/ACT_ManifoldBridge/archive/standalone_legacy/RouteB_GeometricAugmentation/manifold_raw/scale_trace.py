from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


def trace_enabled(flag: Optional[bool] = None) -> bool:
    if flag is not None:
        return bool(flag)
    env = os.getenv("PIA_TRACE_SCALE", "")
    env = env.strip().lower()
    return env not in ("", "0", "false", "no", "off")


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        try:
            x = x.detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(x)


def print_stats(
    tag: str,
    x: Any,
    *,
    force: bool = False,
    near_zero_threshold: float = 1e-12,
) -> None:
    if not force and not trace_enabled():
        return
    arr = _to_numpy(x)
    if arr.size == 0:
        print(f"[scale][{tag}] empty array", flush=True)
        return
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr64 = arr.astype(np.float64, copy=False)
    g_mean = float(arr64.mean())
    g_std = float(arr64.std())
    g_min = float(arr64.min())
    g_max = float(arr64.max())
    print(
        f"[scale][{tag}] GLOBAL mean={g_mean:.6e} std={g_std:.6e} "
        f"min={g_min:.6e} max={g_max:.6e}",
        flush=True,
    )

    std = arr64.std(axis=1)
    ptp = np.ptp(arr64, axis=1)
    std_median = float(np.median(std))
    std_p5 = float(np.percentile(std, 5))
    std_p95 = float(np.percentile(std, 95))
    ptp_median = float(np.median(ptp))
    ptp_p5 = float(np.percentile(ptp, 5))
    ptp_p95 = float(np.percentile(ptp, 95))

    print(
        f"[scale][{tag}] per-chan std median={std_median:.6e} "
        f"p5={std_p5:.6e} p95={std_p95:.6e}",
        flush=True,
    )
    print(
        f"[scale][{tag}] per-chan ptp median={ptp_median:.6e} "
        f"p5={ptp_p5:.6e} p95={ptp_p95:.6e}",
        flush=True,
    )
    print(
        f"[scale][{tag}] std median in uV (if V): {std_median * 1e6:.6e}",
        flush=True,
    )
    print(
        f"[scale][{tag}] ptp median in uV (if V): {ptp_median * 1e6:.6e}",
        flush=True,
    )
    near_zero = int((std < near_zero_threshold).sum())
    print(
        f"[scale][{tag}] near-zero channels (std<{near_zero_threshold:.1e} V): "
        f"{near_zero} / {len(std)}",
        flush=True,
    )
