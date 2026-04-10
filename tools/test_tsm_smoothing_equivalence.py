import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.manifold_streaming_riemann import ManifoldStreamingDataset

PRODUCTION_SMOOTHING_PATH = (
    "datasets.manifold_streaming_riemann.ManifoldStreamingDataset._smooth_sequence "
    "(used by runners.manifold_deep_runner.ManifoldDeepRunner._make_tsm_collate)"
)


def _check_finite(tensor: torch.Tensor) -> bool:
    return torch.isfinite(tensor).all().item()


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a - b).abs()
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def _build_identity_padding(pad_len: int, bands: int = 5, ch: int = 62) -> torch.Tensor:
    if pad_len <= 0:
        return torch.empty((0, bands, ch, ch), dtype=torch.float32)
    eye = torch.eye(ch, dtype=torch.float32).unsqueeze(0).repeat(bands, 1, 1)
    return eye.unsqueeze(0).repeat(pad_len, 1, 1, 1)


def _smooth_sequence(
    x: torch.Tensor,
    mode: str,
    ema_alpha: float,
    kalman_qr: float,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    # Production smoothing path: classmethod used in runner collate.
    return ManifoldStreamingDataset._smooth_sequence(
        x,
        mode,
        ema_alpha,
        kalman_qr,
        padding_mask=padding_mask,
    )


def _run_case(
    *,
    x_raw: torch.Tensor,
    smooth_a: torch.Tensor,
    mode: str,
    ema_alpha: float,
    kalman_qr: float,
    pad_extra: int,
    padding_policy: str,
) -> dict:
    t_len = int(x_raw.shape[0])
    t_max = t_len + int(pad_extra)
    pad_len = t_max - t_len
    if pad_len < 1:
        raise ValueError("pad_extra must be >= 1 for padding tests")

    if padding_policy == "zero":
        pad = torch.zeros((pad_len, *x_raw.shape[1:]), dtype=torch.float32)
    elif padding_policy == "identity":
        pad = _build_identity_padding(pad_len)
    else:
        raise ValueError(f"unknown padding_policy: {padding_policy}")

    x_pad = torch.cat([x_raw, pad], dim=0).unsqueeze(0)
    padding_mask = torch.zeros((1, t_max), dtype=torch.bool)
    padding_mask[0, t_len:] = True

    x_pad64 = x_pad.to(dtype=torch.float64)
    smooth_b = _smooth_sequence(
        x_pad64,
        mode,
        ema_alpha,
        kalman_qr,
        padding_mask=padding_mask,
    ).to(dtype=torch.float32)
    smooth_b_valid = smooth_b[0, :t_len]

    finite_x = _check_finite(x_pad)
    finite_b = _check_finite(smooth_b)
    finite_a = _check_finite(smooth_a)
    diffs = _diff_stats(smooth_a, smooth_b_valid)

    return {
        "padding_policy": padding_policy,
        "t_len": t_len,
        "t_max": t_max,
        "padding_true": int(padding_mask.sum().item()),
        "valid_len": int((~padding_mask).sum().item()),
        "finite": {
            "x_pad": bool(finite_x),
            "smooth_b": bool(finite_b),
            "smooth_a": bool(finite_a),
        },
        "diff": diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TSM smoothing equivalence test.")
    parser.add_argument(
        "--manifest-path",
        required=True,
        help="TSM manifest path (cov_spd seq manifest).",
    )
    parser.add_argument(
        "--ref-mean-path",
        required=True,
        help="TSM reference mean path.",
    )
    parser.add_argument("--index", type=int, default=0, help="sample index")
    parser.add_argument(
        "--smooth-mode",
        type=str,
        default="ema",
        choices=["ema", "kalman"],
    )
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--kalman-qr", type=float, default=1e-4)
    parser.add_argument("--pad-extra", type=int, default=16)
    parser.add_argument(
        "--out-json",
        default="logs/test_tsm_smoothing_equivalence.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    ref_mean = torch.load(args.ref_mean_path, map_location="cpu")
    if isinstance(ref_mean, dict):
        if "ref_mean" in ref_mean:
            ref_mean = ref_mean["ref_mean"]
        elif "mean" in ref_mean:
            ref_mean = ref_mean["mean"]
        else:
            raise ValueError("ref_mean dict missing ref_mean/mean")
    ref_mean = torch.as_tensor(ref_mean, dtype=torch.float32)

    dataset = ManifoldStreamingDataset(
        args.manifest_path,
        reference_mean=ref_mean,
        tsm_smooth_mode="none",
    )
    x_raw, _label = dataset[int(args.index)]
    if x_raw.ndim != 4:
        raise ValueError(f"expected x_raw shape [T,5,62,62], got {x_raw.shape}")

    x_raw64 = x_raw.to(dtype=torch.float64)
    smooth_a = _smooth_sequence(
        x_raw64,
        args.smooth_mode,
        float(args.ema_alpha),
        float(args.kalman_qr),
        padding_mask=None,
    ).to(dtype=torch.float32)

    case_zero = _run_case(
        x_raw=x_raw,
        smooth_a=smooth_a,
        mode=args.smooth_mode,
        ema_alpha=float(args.ema_alpha),
        kalman_qr=float(args.kalman_qr),
        pad_extra=int(args.pad_extra),
        padding_policy="zero",
    )
    case_identity = _run_case(
        x_raw=x_raw,
        smooth_a=smooth_a,
        mode=args.smooth_mode,
        ema_alpha=float(args.ema_alpha),
        kalman_qr=float(args.kalman_qr),
        pad_extra=int(args.pad_extra),
        padding_policy="identity",
    )

    def _eval_case(case: dict) -> dict:
        finite_ok = all(case["finite"].values())
        max_diff = case["diff"]["max_abs_diff"]
        mean_diff = case["diff"]["mean_abs_diff"]
        if args.smooth_mode == "ema":
            diff_ok = max_diff < 1e-5 and mean_diff < 1e-6
        else:
            diff_ok = max_diff < 1e-4 and mean_diff < 1e-5
        return {
            "finite_ok": bool(finite_ok),
            "diff_ok": bool(diff_ok),
            "pass": bool(finite_ok and diff_ok),
        }

    eval_zero = _eval_case(case_zero)
    eval_identity = _eval_case(case_identity)

    report = {
        "manifest_path": args.manifest_path,
        "ref_mean_path": args.ref_mean_path,
        "index": int(args.index),
        "smooth_mode": args.smooth_mode,
        "ema_alpha": float(args.ema_alpha),
        "kalman_qr": float(args.kalman_qr),
        "pad_extra": int(args.pad_extra),
        "production_smoothing_path": PRODUCTION_SMOOTHING_PATH,
        "cases": {
            "zero": case_zero,
            "identity": case_identity,
        },
        "eval": {
            "zero": eval_zero,
            "identity": eval_identity,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    had_nonfinite = False
    for name, case, eval_res in [
        ("zero", case_zero, eval_zero),
        ("identity", case_identity, eval_identity),
    ]:
        status = "PASS" if eval_res["pass"] else "FAIL"
        finite = case["finite"]
        diff = case["diff"]
        if not all(finite.values()):
            bad = [k for k, v in finite.items() if not v]
            print(f"[{name}] non-finite detected in {bad}", flush=True)
            had_nonfinite = True
        print(
            f"[{name}] {status} t_len={case['t_len']} t_max={case['t_max']} "
            f"padding_true={case['padding_true']} valid_len={case['valid_len']} "
            f"finite(x_pad={finite['x_pad']} smooth_B={finite['smooth_b']} "
            f"smooth_A={finite['smooth_a']}) "
            f"max_abs_diff={diff['max_abs_diff']:.6e} "
            f"mean_abs_diff={diff['mean_abs_diff']:.6e}",
            flush=True,
        )
    print(f"[report] {out_path}", flush=True)
    if had_nonfinite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
