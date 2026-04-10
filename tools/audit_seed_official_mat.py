import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scipy.io import whosmat
except Exception:
    whosmat = None

import numpy as np


IGNORE_KEYS = {"__header__", "__version__", "__globals__"}
KEYWORDS = ["ExtractedFeatures", "Extracted", "Features", "SEED", "feature", "DE", "LDS"]


def _find_root(user_root: str | None) -> tuple[Path | None, str]:
    if user_root:
        return Path(user_root), "arg"
    env_root = os.environ.get("SEED_OFFICIAL_FEATURE_ROOT")
    if env_root:
        return Path(env_root), "env"
    candidates = []
    repo_root = ROOT

    preferred = [
        repo_root / "data/SEED/SEED_EEG/ExtractedFeatures_1s",
        repo_root / "data/SEED/SEED_EEG/ExtractedFeatures_4s",
    ]
    for p in preferred:
        if p.is_dir():
            mats = list(p.glob("*.mat"))
            if mats:
                candidates.append((p, len(mats), "preferred"))

    max_depth = 4
    for dirpath, dirnames, filenames in os.walk(repo_root):
        rel = Path(dirpath).relative_to(repo_root)
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue
        name = Path(dirpath).name
        if any(k.lower() in name.lower() for k in KEYWORDS):
            mats = [f for f in filenames if f.lower().endswith(".mat")]
            if mats:
                candidates.append((Path(dirpath), len(mats), "keyword"))

    if not candidates:
        return None, "not_found"
    candidates.sort(key=lambda x: (-x[1], 0 if "1s" in x[0].name else 1))
    return candidates[0][0], candidates[0][2]


def _parse_subject_date(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"^(\d+)_([0-9]{8})$", base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _key_base(name: str) -> str:
    return re.sub(r"\d+$", "", name)


def _scan_mat(path: Path):
    if whosmat is None:
        raise RuntimeError("scipy.io.whosmat not available; install scipy")
    rows = []
    for name, shape, dtype in whosmat(path):
        if name in IGNORE_KEYS:
            continue
        rows.append({"name": name, "shape": list(shape), "dtype": str(dtype)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Audit SEED official .mat features.")
    parser.add_argument("--root", type=str, default=None, help="feature root dir")
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--write-md", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="logs")
    args = parser.parse_args()

    root, source = _find_root(args.root)
    if root is None or not root.is_dir():
        print("[audit] official feature root not found. Provide --root or set SEED_OFFICIAL_FEATURE_ROOT.")
        sys.exit(1)

    mats = sorted(list(root.glob("*.mat")))
    if not mats:
        mats = sorted(list(root.rglob("*.mat")))
    if not mats:
        print(f"[audit] no .mat files found in {root}")
        sys.exit(1)

    max_files = min(int(args.max_files), len(mats))
    mats = mats[:max_files]

    key_counts = Counter()
    key_base_counts = Counter()
    key_shapes = defaultdict(list)
    key_base_shapes = defaultdict(list)
    file_summaries = []

    for p in mats:
        subject, date = _parse_subject_date(p.name)
        rows = _scan_mat(p)
        for row in rows:
            name = row["name"]
            shape = tuple(row["shape"])
            key_counts[name] += 1
            base = _key_base(name)
            key_base_counts[base] += 1
            key_shapes[name].append(shape)
            key_base_shapes[base].append(shape)
        file_summaries.append(
            {
                "path": str(p),
                "subject": subject,
                "date": date,
                "keys": rows,
            }
        )

    top_bases = [
        {"key_base": k, "count": int(v), "distinct_shapes": int(len(set(key_base_shapes[k])))}
        for k, v in key_base_counts.most_common(10)
    ]
    recommended = None
    for k, _ in key_base_counts.most_common():
        if "de_lds" in k.lower():
            recommended = k
            break
    if recommended is None:
        for k, _ in key_base_counts.most_common():
            if "de" in k.lower():
                recommended = k
                break
    if recommended is None and top_bases:
        recommended = top_bases[0]["key_base"]

    recommended_shape = None
    sample_unit = "unknown"
    risks = []
    if recommended:
        shapes = key_base_shapes[recommended]
        if shapes:
            recommended_shape = list(max(set(shapes), key=shapes.count))
        # detect trial-by-key suffix
        suffix_counts = []
        for summary in file_summaries:
            names = [row["name"] for row in summary["keys"]]
            matched = [n for n in names if n.startswith(recommended) and re.match(r".*\d+$", n)]
            if matched:
                suffix_counts.append(len(matched))
        if suffix_counts and int(np.median(suffix_counts)) == 15:
            sample_unit = "trial_via_key_suffix"
        else:
            # check if any shape contains 15 as trial axis
            for shape in key_base_shapes[recommended]:
                if 15 in shape:
                    sample_unit = "trial_axis_in_array"
                    break
    else:
        risks.append("no_recommended_key_base")

    if sample_unit == "unknown":
        risks.append("unable_to_infer_trial_dimension")

    report = {
        "root": str(root),
        "root_source": source,
        "mat_count": len(mats),
        "scanned_files": [str(p) for p in mats],
        "top_key_bases": top_bases,
        "recommended_key_base": recommended,
        "recommended_key_shape": recommended_shape,
        "sample_unit": sample_unit,
        "risks": risks,
        "notes": {
            "session_inference": "per-subject date-sorted filenames -> session index",
            "filename_pattern": "subject_YYYYMMDD.mat",
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baseline_preflight.json"
    json_path.write_text(json.dumps(report, indent=2))

    if int(args.write_md):
        md_path = out_dir / "baseline_preflight_report.md"
        lines = [
            "# SEED Official .mat Preflight",
            "",
            f"- Found root: `{root}` (source={source})",
            f"- mat_count (scanned): {len(mats)}",
            "",
            "## Top Keys (base names)",
        ]
        for row in top_bases:
            lines.append(
                f"- {row['key_base']}: count={row['count']} distinct_shapes={row['distinct_shapes']}"
            )
        lines += [
            "",
            "## Recommendation",
            f"- chosen feature_key_base: `{recommended}`",
            f"- chosen key shape (mode): {recommended_shape}",
            f"- sample unit verdict: {sample_unit}",
            "",
            "## Alignment Risks",
        ]
        if risks:
            for r in risks:
                lines.append(f"- {r}")
        else:
            lines.append("- none_detected")
        lines += [
            "",
            "## Mapping Notes",
            "- session_inference: per-subject date-sorted filenames -> session index",
            "- filename_pattern: subject_YYYYMMDD.mat",
        ]
        md_path.write_text("\n".join(lines) + "\n")

    print(f"[preflight] root={root}", flush=True)
    print(f"[preflight] mat_count={len(mats)}", flush=True)
    if recommended:
        print(
            f"[preflight] recommended_key_base={recommended} shape={recommended_shape}",
            flush=True,
        )
    print(f"[preflight] sample_unit={sample_unit}", flush=True)
    print(f"[preflight] report={json_path}", flush=True)


if __name__ == "__main__":
    main()
