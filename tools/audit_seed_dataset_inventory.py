import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EXPECTED_SUBJECTS = [str(i) for i in range(1, 16)]


def _parse_subject_session(name: str, kind: str) -> Tuple[Optional[str], Optional[str]]:
    base = Path(name).stem
    if kind in {"cnt", "npz"}:
        m = re.match(r"(\d+)[_-](\d+)", base)
        if m:
            return m.group(1), m.group(2)
    if kind == "fif":
        m = re.match(r"(\d+)[_-](\d+)", base)
        if m:
            return m.group(1), m.group(2)
    if kind == "mat":
        m = re.match(r"(\d+)_(\d+)", base)
        if m:
            return m.group(1), m.group(2)
    return None, None


def _coverage_for_files(files: List[Path], kind: str) -> Dict[str, object]:
    subjects = set()
    sessions = set()
    pairs = set()
    per_subject_sessions: Dict[str, set] = defaultdict(set)

    for path in files:
        subj, sess = _parse_subject_session(path.name, kind)
        if subj is None or sess is None:
            continue
        subjects.add(subj)
        sessions.add(sess)
        pairs.add((subj, sess))
        per_subject_sessions[subj].add(sess)

    missing_subjects = [s for s in EXPECTED_SUBJECTS if s not in subjects]

    expected_pairs = None
    missing_pairs_count = None
    session_vals = []
    for s in sessions:
        if str(s).isdigit():
            session_vals.append(int(s))
    if session_vals and max(session_vals) <= 3:
        expected_sessions = {1, 2, 3}
        expected_pairs = len(EXPECTED_SUBJECTS) * len(expected_sessions)
        missing_pairs_count = sum(
            1
            for subj in EXPECTED_SUBJECTS
            for sess in expected_sessions
            if (subj, str(sess)) not in pairs
        )
    elif per_subject_sessions:
        expected_pairs = len(EXPECTED_SUBJECTS) * 3
        missing_pairs_count = sum(
            max(0, 3 - len(per_subject_sessions.get(subj, set())))
            for subj in EXPECTED_SUBJECTS
        )

    return {
        "subjects_present": sorted(subjects, key=lambda x: int(x) if str(x).isdigit() else x),
        "sessions_present": sorted(sessions),
        "pairs_count": int(len(pairs)),
        "expected_pairs": int(expected_pairs) if expected_pairs is not None else None,
        "missing_subjects": missing_subjects,
        "missing_pairs_count": int(missing_pairs_count) if missing_pairs_count is not None else None,
    }


def _find_dirs_with_ext(seed_root: Path, ext: str) -> List[Path]:
    roots = set()
    for dirpath, _, filenames in os.walk(seed_root):
        if any(f.lower().endswith(ext) for f in filenames):
            roots.add(Path(dirpath))
    return sorted(roots)


def _summarize_roots(roots: List[Path], kind: str, ext: str) -> List[Dict[str, object]]:
    out = []
    for root in roots:
        files = sorted(root.glob(f"*{ext}"))
        if not files:
            continue
        cov = _coverage_for_files(files, kind)
        out.append(
            {
                "path": str(root),
                "file_count": int(len(files)),
                "coverage": cov,
            }
        )
    return out


def _find_named_dirs(seed_root: Path, name: str) -> List[Path]:
    return sorted(p for p in seed_root.rglob(name) if p.is_dir())


def _find_china_info(seed_root: Path) -> Optional[str]:
    for p in seed_root.rglob("China_information.xlsx"):
        if p.is_file():
            return str(p)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="SEED_1 dataset inventory audit.")
    parser.add_argument("--seed-root", required=True, help="SEED root directory")
    parser.add_argument(
        "--out-json",
        default="logs/seed_inventory.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    seed_root = Path(args.seed_root).resolve()
    if not seed_root.is_dir():
        raise FileNotFoundError(f"seed_root not found: {seed_root}")

    raw_cnt_roots = _summarize_roots(_find_dirs_with_ext(seed_root, ".cnt"), "cnt", ".cnt")
    raw_fif_roots = _summarize_roots(_find_dirs_with_ext(seed_root, ".fif"), "fif", ".fif")

    mat_roots = []
    for name in ("ExtractedFeatures_1s", "ExtractedFeatures_4s"):
        for root in _find_named_dirs(seed_root, name):
            files = sorted(root.glob("*.mat"))
            if not files:
                continue
            cov = _coverage_for_files(files, "mat")
            mat_roots.append(
                {
                    "path": str(root),
                    "kind": name,
                    "file_count": int(len(files)),
                    "coverage": cov,
                }
            )

    npz_roots = []
    for name in ("eeg_used_1s", "eeg_used_4s"):
        for root in _find_named_dirs(seed_root, name):
            files = sorted(root.glob("*.npz"))
            if not files:
                continue
            cov = _coverage_for_files(files, "npz")
            npz_roots.append(
                {
                    "path": str(root),
                    "kind": name,
                    "file_count": int(len(files)),
                    "coverage": cov,
                }
            )

    china_info = _find_china_info(seed_root)

    report = {
        "seed_root": str(seed_root),
        "raw_cnt_roots": raw_cnt_roots,
        "raw_fif_roots": raw_fif_roots,
        "extracted_mat_roots": mat_roots,
        "multimodal_npz_roots": npz_roots,
        "china_info_xlsx": {
            "exists": bool(china_info),
            "path": china_info,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"[seed_inventory] seed_root={seed_root}", flush=True)
    for group_name, roots in [
        ("raw_cnt_roots", raw_cnt_roots),
        ("raw_fif_roots", raw_fif_roots),
        ("mat_roots", mat_roots),
        ("npz_roots", npz_roots),
    ]:
        print(f"[seed_inventory] {group_name} count={len(roots)}", flush=True)
        for entry in roots:
            cov = entry["coverage"]
            print(
                f"  - {entry['path']} files={entry['file_count']} "
                f"subjects={len(cov['subjects_present'])} sessions={len(cov['sessions_present'])} "
                f"pairs={cov['pairs_count']} missing_pairs={cov['missing_pairs_count']}",
                flush=True,
            )
    if china_info:
        print(f"[seed_inventory] china_info_xlsx={china_info}", flush=True)
    else:
        print("[seed_inventory] china_info_xlsx=missing", flush=True)


if __name__ == "__main__":
    main()
