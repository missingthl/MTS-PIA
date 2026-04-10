from __future__ import annotations

import glob
import gc
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from datasets.seed_raw_trials import build_trial_index
from manifold_raw.features import (
    BandSpec,
    bandpass,
    cov_shrink,
    logmap_spd,
    parse_band_spec,
    vec_utri,
    window_slices,
)
from manifold_raw.scale_trace import print_stats, trace_enabled
from manifold_raw.spd_eps import compute_spd_eps


@dataclass
class ManifoldRawV1Runner:
    raw_manifest: Optional[str]
    seed_raw_root: str
    raw_window_sec: float = 4.0
    raw_window_hop_sec: float = 4.0
    raw_resample_fs: float = 0.0
    raw_bands: str = "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50"
    raw_cov: str = "shrinkage_oas"
    raw_logmap_eps: float = 1e-6
    raw_seq_save_format: str = "vec_utri"
    spd_eps: float = 1e-5
    spd_eps_mode: str = "relative_trace"
    spd_eps_alpha: float = 1e-2
    spd_eps_floor_mult: float = 1e-6
    spd_eps_ceil_mult: float = 1e-1
    clf: str = "ridge"
    trial_protocol: str = "session_holdout"
    out_prefix: Optional[str] = None
    raw_manifest_used: Optional[str] = None
    raw_backend: str = "cnt"
    raw_chunk_by: str = "none"
    raw_max_subjects: int = 0
    raw_subject_list: Optional[str] = None
    raw_mem_debug: int = 0
    raw_mem_interval: int = 0
    raw_scale_debug: int = 0
    raw_scale_debug_sec: float = 10.0
    raw_save_trial: str = "auto"
    raw_filter_chunk: int = 0
    raw_resample_chunk: int = 0
    raw_time_unit: Optional[str] = None
    raw_trial_offset_sec: float = -3.0
    _mem_warned: bool = False
    _time_unit_logged: bool = False
    _time_unit_warned: bool = False

    def _load_manifest(self) -> List[dict]:
        manifest_path = None
        if self.raw_manifest:
            if "*" in self.raw_manifest or "?" in self.raw_manifest:
                matches = sorted(glob.glob(self.raw_manifest))
                if matches:
                    manifest_path = matches[-1]
            elif os.path.isfile(self.raw_manifest):
                manifest_path = self.raw_manifest

        if manifest_path:
            self.raw_manifest_used = manifest_path
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "trials" not in data:
                    raise ValueError(f"manifest dict missing 'trials': {manifest_path}")
                return data["trials"]
            if isinstance(data, list):
                if data and "t_start_s" in data[0]:
                    return data
                if data and "out_path" in data[0]:
                    return self._expand_convert_manifest(data)
                return data
            raise ValueError(f"unsupported manifest format: {manifest_path}")

        # Build manifest from raw root without loading CNT data.
        cnt_files = sorted(
            [
                str(p)
                for p in Path(self.seed_raw_root).iterdir()
                if p.suffix.lower() == ".cnt"
            ]
        )
        time_txt_path = os.path.join(self.seed_raw_root, "time.txt")
        stim_xlsx_path = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
        manifest = []
        for cnt in cnt_files:
            manifest.extend(
                build_trial_index(
                    cnt,
                    time_txt_path,
                    stim_xlsx_path,
                    time_unit=self.raw_time_unit,
                )
            )
        rows = [
            {
                "subject": t.subject,
                "session": t.session,
                "trial": t.trial,
                "label": t.label,
                "t_start_s": t.t_start_s,
                "t_end_s": t.t_end_s,
                "source_cnt_path": t.source_cnt_path,
                "time_unit": getattr(t, "time_unit", None),
            }
            for t in manifest
        ]
        return rows

    def _expand_convert_manifest(self, entries: List[dict]) -> List[dict]:
        from datasets.seed_raw_trials import build_trial_index

        rows: List[dict] = []
        for entry in entries:
            cnt_path = entry.get("cnt_path")
            out_path = entry.get("out_path")
            if not cnt_path:
                raise ValueError("convert manifest entry missing cnt_path")
            if self.raw_backend == "fif":
                if not out_path or not os.path.isfile(out_path):
                    raise FileNotFoundError(
                        f"Expected FIF missing for CNT: {cnt_path} (out_path={out_path})"
                    )
            time_txt = os.path.join(os.path.dirname(cnt_path), "time.txt")
            if not os.path.isfile(time_txt):
                time_txt = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
            stim_xlsx = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
            trials = build_trial_index(
                cnt_path,
                time_txt,
                stim_xlsx,
                time_unit=self.raw_time_unit,
            )
            for t in trials:
                source_path = out_path if self.raw_backend == "fif" else cnt_path
                rows.append(
                    {
                        "subject": t.subject,
                        "session": t.session,
                        "trial": t.trial,
                        "label": t.label,
                        "t_start_s": t.t_start_s,
                        "t_end_s": t.t_end_s,
                        "source_cnt_path": source_path,
                        "cnt_path": cnt_path,
                        "time_unit": getattr(t, "time_unit", None),
                    }
                )
        return rows

    def _out_prefix(self) -> str:
        if self.out_prefix:
            return self.out_prefix
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join("logs", f"manifold_raw_v1_seed1_{ts}")

    @staticmethod
    def _subject_sort_key(subject: str) -> Tuple[int, str]:
        try:
            return (0, f"{int(subject):04d}")
        except (ValueError, TypeError):
            return (1, str(subject))

    def _rss_gb(self) -> float:
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        except Exception:
            import resource
            import sys

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                rss_bytes = float(usage)
            else:
                rss_bytes = float(usage) * 1024.0
            if not self._mem_warned:
                print("[mem] psutil not available; using ru_maxrss (peak), not live RSS")
                self._mem_warned = True
            return rss_bytes / (1024**3)

    def _log_mem(self, tag: str, extra: Optional[str] = None) -> None:
        if not self.raw_mem_debug:
            return
        msg = f"[mem][{tag}] rss={self._rss_gb():.3f}GB"
        if extra:
            msg = f"{msg} {extra}"
        print(msg, flush=True)

    def _build_splits(self, meta_trials: List[dict]) -> List[dict]:
        protocol = (self.trial_protocol or "").lower()
        by_subject: Dict[str, List[int]] = defaultdict(list)
        for idx, meta in enumerate(meta_trials):
            by_subject[str(meta["subject"])].append(idx)

        splits: List[dict] = []
        if protocol == "session_holdout":
            for subject, idxs in sorted(by_subject.items(), key=lambda x: self._subject_sort_key(x[0])):
                train_idx: List[int] = []
                test_idx: List[int] = []
                for i in idxs:
                    session = int(meta_trials[i]["session"])
                    if session in (1, 2):
                        train_idx.append(i)
                    elif session == 3:
                        test_idx.append(i)
                if not train_idx or not test_idx:
                    print(f"[raw_v1] skip subject={subject} (train={len(train_idx)} test={len(test_idx)})")
                    continue
                splits.append(
                    {
                        "name": f"subj_{subject}_s12_vs_s3",
                        "subject": subject,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                    }
                )
        elif protocol == "loso_subject":
            all_idx = list(range(len(meta_trials)))
            all_idx_set = set(all_idx)
            for subject, idxs in sorted(by_subject.items(), key=lambda x: self._subject_sort_key(x[0])):
                test_idx = idxs
                train_idx = list(all_idx_set.difference(idxs))
                if not train_idx or not test_idx:
                    print(f"[raw_v1] skip subject={subject} (train={len(train_idx)} test={len(test_idx)})")
                    continue
                splits.append(
                    {
                        "name": f"loso_{subject}",
                        "subject": subject,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                    }
                )
        else:
            raise ValueError(f"Unknown trial protocol: {self.trial_protocol}")

        return splits

    def _resample_chunked(self, data: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
        import mne

        n_ch = data.shape[0]
        chunk = int(self.raw_resample_chunk or 0)
        if chunk <= 0 or chunk >= n_ch:
            return mne.filter.resample(
                np.asarray(data, dtype=np.float64),
                up=fs_to,
                down=fs_from,
                axis=-1,
                npad="auto",
                n_jobs=1,
            )

        out_list = []
        for start in range(0, n_ch, chunk):
            end = min(start + chunk, n_ch)
            chunk_data = mne.filter.resample(
                np.asarray(data[start:end], dtype=np.float64),
                up=fs_to,
                down=fs_from,
                axis=-1,
                npad="auto",
                n_jobs=1,
            )
            out_list.append(chunk_data)
        return np.concatenate(out_list, axis=0)

    def run(self) -> None:
        if self.raw_backend != "fif":
            raise ValueError(
                "manifold_raw_v1_frozen: CNT backend is not allowed. "
                "Use offline conversion to FIF and set --seed-raw-backend fif."
            )
        out_prefix = self._out_prefix()
        manifest = self._load_manifest()
        if not manifest:
            raise ValueError("raw manifest is empty")

        bands = parse_band_spec(self.raw_bands)
        if not bands:
            raise ValueError("raw band spec is empty")

        print(f"[raw_v1] raw_root={os.path.abspath(self.seed_raw_root)}")
        print(
            f"[raw_v1] window_sec={self.raw_window_sec} hop_sec={self.raw_window_hop_sec}"
        )
        print(f"[raw_v1] bands={[f'{b.name}:{b.lo}-{b.hi}' for b in bands]}")
        print(f"[raw_v1] cov={self.raw_cov} logmap_eps={self.raw_logmap_eps}")
        seq_save_format = (self.raw_seq_save_format or "vec_utri").lower()
        if seq_save_format not in {"vec_utri", "cov_spd"}:
            raise ValueError(
                f"raw_seq_save_format must be vec_utri or cov_spd, got {self.raw_seq_save_format}"
            )
        spd_eps_mode = (self.spd_eps_mode or "absolute").lower()
        if spd_eps_mode not in {"absolute", "relative_trace", "relative_diag"}:
            raise ValueError(f"spd_eps_mode must be absolute/relative_trace/relative_diag, got {self.spd_eps_mode}")
        if spd_eps_mode == "absolute":
            if self.spd_eps is None or float(self.spd_eps) <= 0:
                raise ValueError(f"spd_eps must be > 0, got {self.spd_eps}")
        else:
            if self.spd_eps_alpha is None or float(self.spd_eps_alpha) <= 0:
                raise ValueError(f"spd_eps_alpha must be > 0, got {self.spd_eps_alpha}")
            if self.spd_eps_floor_mult is None or float(self.spd_eps_floor_mult) <= 0:
                raise ValueError(
                    f"spd_eps_floor_mult must be > 0, got {self.spd_eps_floor_mult}"
                )
            if self.spd_eps_ceil_mult is None or float(self.spd_eps_ceil_mult) <= 0:
                raise ValueError(
                    f"spd_eps_ceil_mult must be > 0, got {self.spd_eps_ceil_mult}"
                )
            if float(self.spd_eps_floor_mult) > float(self.spd_eps_ceil_mult):
                raise ValueError(
                    "spd_eps_floor_mult must be <= spd_eps_ceil_mult "
                    f"(got {self.spd_eps_floor_mult} > {self.spd_eps_ceil_mult})"
                )
        self.spd_eps_mode = spd_eps_mode
        self.raw_seq_save_format = seq_save_format
        print(
            f"[raw_v1] seq_save_format={self.raw_seq_save_format} spd_eps={self.spd_eps}"
        )
        print(
            f"[raw_v1] spd_eps_mode={spd_eps_mode} alpha={self.spd_eps_alpha} "
            f"floor_mult={self.spd_eps_floor_mult} ceil_mult={self.spd_eps_ceil_mult}"
        )
        print(f"[raw_v1] protocol={self.trial_protocol}")
        print(f"[raw_v1] raw_backend={self.raw_backend}")
        print(
            f"[raw_v1] filter_chunk={int(self.raw_filter_chunk or 0)} "
            f"resample_chunk={int(self.raw_resample_chunk or 0)}"
        )
        print(f"[raw_v1] trial_offset_sec={float(self.raw_trial_offset_sec):.3f}")

        rows_all = [dict(row) for row in manifest]
        subjects_all = sorted(
            {str(r["subject"]) for r in rows_all},
            key=self._subject_sort_key,
        )
        subject_list: List[str]
        if self.raw_subject_list:
            requested = [s.strip() for s in self.raw_subject_list.split(",") if s.strip()]
            subject_list = [s for s in requested if s in subjects_all]
            missing = [s for s in requested if s not in subjects_all]
            if missing:
                print(f"[raw_v1] Warning: subjects not found: {missing}")
        else:
            subject_list = list(subjects_all)
        if self.raw_max_subjects and self.raw_max_subjects > 0:
            subject_list = subject_list[: int(self.raw_max_subjects)]

        rows_selected = [r for r in rows_all if str(r["subject"]) in subject_list]
        if self.raw_chunk_by == "subject":
            subjects_to_run = list(subject_list)
        else:
            subjects_to_run = ["__all__"]

        for i, row in enumerate(rows_selected):
            row["_index"] = i

        print(f"[raw_v1] subjects_selected={len(subject_list)}")
        print(f"[raw_v1] trials_selected={len(rows_selected)}")

        save_trial = self.raw_save_trial.lower()
        if save_trial == "auto":
            save_trial = "yes" if len(rows_selected) <= 50 else "no"
        if save_trial not in {"yes", "no"}:
            raise ValueError(f"raw_save_trial must be yes/no/auto, got {self.raw_save_trial}")
        print(f"[raw_v1] save_trial={save_trial} out_prefix={out_prefix}")

        by_cnt_global: Dict[str, List[dict]] = defaultdict(list)
        for row in rows_selected:
            cnt_path = row.get("source_cnt_path") or row.get("cnt_path")
            if not cnt_path:
                raise ValueError("manifest row missing source_cnt_path/cnt_path")
            by_cnt_global[cnt_path].append(row)

        n_trials = len(rows_selected)
        X_memmap: Optional[np.memmap] = None
        file_paths: List[Optional[str]] = [None] * n_trials
        seq_lens = np.zeros(n_trials, dtype=np.int64)
        feat_dim: Optional[int] = None
        seq_feature_shape: Optional[Tuple[int, ...]] = None
        y_memmap = np.memmap(
            f"{out_prefix}_y_trial.memmap",
            dtype=np.int64,
            mode="w+",
            shape=(n_trials,),
        )
        meta_trials: List[Optional[dict]] = [None] * n_trials
        skip_trials: List[dict] = []
        channel_names: List[str] = []
        channel_hash = None

        trial0_debug_done = False
        sequence_mode = True
        scale_trace_done = False
        eps_by_band: Dict[str, List[float]] = defaultdict(list)

        fs_raw_global = None
        fs_used_global = None
        trial_counter = 0
        for subject in subjects_to_run:
            if self.raw_chunk_by == "subject":
                rows_subset = [r for r in rows_selected if str(r["subject"]) == subject]
                by_cnt: Dict[str, List[dict]] = defaultdict(list)
                for row in rows_subset:
                    cnt_path = row.get("source_cnt_path") or row.get("cnt_path")
                    by_cnt[cnt_path].append(row)
                subject_tag = f"subject={subject}"
            else:
                by_cnt = dict(by_cnt_global)
                subject_tag = "subject=all"

            self._log_mem("SUBJECT_START", subject_tag)

            for cnt_path, rows in sorted(by_cnt.items()):
                self._log_mem("CNT_START", os.path.basename(cnt_path))
                if self.raw_backend == "fif" and not cnt_path.lower().endswith(".fif"):
                    raise ValueError(f"Expected FIF input, got: {cnt_path}")
                raw = load_one_raw(cnt_path, backend=self.raw_backend, preload=False)
                raw62, meta = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
                self._log_mem("CNT_LOADED", os.path.basename(cnt_path))

                names = meta["selected_names"]
                if not channel_names:
                    channel_names = list(names)
                    import hashlib

                    channel_hash = hashlib.sha1(
                        ",".join(channel_names).encode("utf-8")
                    ).hexdigest()
                elif channel_names != list(names):
                    raise ValueError("channel order mismatch across CNT files")

                fs_raw = float(raw62.info.get("sfreq", 0.0))
                n_times = int(raw62.n_times)
                fs_used = fs_raw
                if self.raw_resample_fs and fs_raw > 0:
                    fs_used = float(self.raw_resample_fs)
                if fs_raw_global is None:
                    fs_raw_global = fs_raw
                    fs_used_global = fs_used
                    print(f"[raw_v1] fs_raw={fs_raw_global} fs_used={fs_used_global}")
                else:
                    if fs_used_global is not None and abs(fs_used - fs_used_global) > 1e-3:
                        print(
                            f"[raw_v1] Warning: fs_used mismatch {fs_used} vs {fs_used_global} "
                            f"for {os.path.basename(cnt_path)}"
                        )

                if not scale_trace_done and trace_enabled(self.raw_scale_debug):
                    trace_sec = float(self.raw_scale_debug_sec or 10.0)
                    trace_stop = int(round(trace_sec * fs_raw)) if fs_raw > 0 else 0
                    if trace_stop > 0:
                        raw_slice = raw.get_data(start=0, stop=trace_stop)
                        nchan_raw = int(raw.info.get("nchan", raw_slice.shape[0]))
                        print(
                            f"[scale][after_load_one_raw] dtype={raw_slice.dtype} "
                            f"nchan={nchan_raw} sfreq={fs_raw}",
                            flush=True,
                        )
                        unit = None
                        unit_mul = None
                        chs = raw.info.get("chs")
                        if chs:
                            unit = chs[0].get("unit")
                            unit_mul = chs[0].get("unit_mul")
                        orig_units = getattr(raw, "_orig_units", None)
                        if isinstance(orig_units, dict):
                            sample = list(orig_units.items())[:5]
                            orig_units_summary = f"sample={sample} total={len(orig_units)}"
                        else:
                            orig_units_summary = str(orig_units)
                        print(
                            f"[scale][after_load_one_raw] unit={unit} unit_mul={unit_mul} "
                            f"orig_units={orig_units_summary}",
                            flush=True,
                        )
                        print_stats("after_load_one_raw", raw_slice, force=True)

                        raw62_slice = raw62.get_data(start=0, stop=trace_stop)
                        nchan_62 = int(raw62.info.get("nchan", raw62_slice.shape[0]))
                        selected = meta.get("selected_names", [])
                        selected_norm = [str(n).upper() for n in selected]
                        has_eog = any("EOG" in n for n in selected_norm)
                        has_ref = any(
                            n in {"REF", "A1", "A2", "M1", "M2"} or "REF" in n
                            for n in selected_norm
                        )
                        print(
                            f"[scale][after_build_eeg62_view] dtype={raw62_slice.dtype} "
                            f"nchan={nchan_62} selected={len(selected)} "
                            f"has_eog={has_eog} has_ref={has_ref}",
                            flush=True,
                        )
                        print_stats("after_build_eeg62_view", raw62_slice, force=True)

                        trace_seg = raw62_slice
                        fs_trace = fs_raw
                        if fs_used != fs_raw and fs_raw > 0:
                            trace_seg = self._resample_chunked(
                                trace_seg, fs_from=fs_raw, fs_to=fs_used
                            )
                            fs_trace = fs_used
                        for band in bands:
                            band_data = bandpass(
                                trace_seg,
                                fs_trace,
                                band,
                                chunk_size=int(self.raw_filter_chunk or 0),
                            )
                            print_stats(f"after_bandpass_{band.name}", band_data, force=True)
                        scale_trace_done = True

                for row in rows:
                    offset_sec = float(self.raw_trial_offset_sec or 0.0)
                    t_start_raw = float(row["t_start_s"])
                    t_end_raw = float(row["t_end_s"])
                    t_start = t_start_raw + offset_sec
                    t_end = t_end_raw + offset_sec
                    start_idx_raw = int(round(t_start_raw * fs_raw))
                    end_idx_raw = int(round(t_end_raw * fs_raw))
                    start_idx = int(round(t_start * fs_raw))
                    end_idx = int(round(t_end * fs_raw))
                    skip_reason = None
                    if start_idx < 0:
                        skip_reason = "start_before_0"
                    elif end_idx <= start_idx:
                        skip_reason = "non_positive_duration"
                    elif start_idx >= n_times:
                        skip_reason = "start_after_end"
                    if skip_reason:
                        skip_trials.append(
                            {
                                "subject": row["subject"],
                                "session": row["session"],
                                "trial": row["trial"],
                                "label": int(row["label"]),
                                "start_idx_raw": start_idx_raw,
                                "end_idx_raw": end_idx_raw,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "t_start_s_raw": t_start_raw,
                                "t_end_s_raw": t_end_raw,
                                "t_start_s_adj": t_start,
                                "t_end_s_adj": t_end,
                                "trial_offset_sec": offset_sec,
                                "reason": skip_reason,
                            }
                        )
                        print(
                            f"[raw_v1][skip] subject={row['subject']} session={row['session']} "
                            f"trial={row['trial']} reason={skip_reason} "
                            f"start_idx={start_idx} end_idx={end_idx}",
                            flush=True,
                        )
                        continue
                    start_idx_clamped = max(0, min(start_idx, n_times - 1))
                    end_idx_clamped = max(start_idx_clamped + 1, min(end_idx, n_times))
                    if end_idx_clamped <= start_idx_clamped:
                        skip_reason = "clamped_empty"
                        skip_trials.append(
                            {
                                "subject": row["subject"],
                                "session": row["session"],
                                "trial": row["trial"],
                                "label": int(row["label"]),
                                "start_idx_raw": start_idx_raw,
                                "end_idx_raw": end_idx_raw,
                                "start_idx": start_idx_clamped,
                                "end_idx": end_idx_clamped,
                                "t_start_s_raw": t_start_raw,
                                "t_end_s_raw": t_end_raw,
                                "t_start_s_adj": t_start,
                                "t_end_s_adj": t_end,
                                "trial_offset_sec": offset_sec,
                                "reason": skip_reason,
                            }
                        )
                        print(
                            f"[raw_v1][skip] subject={row['subject']} session={row['session']} "
                            f"trial={row['trial']} reason={skip_reason} "
                            f"start_idx={start_idx_clamped} end_idx={end_idx_clamped}",
                            flush=True,
                        )
                        continue
                    if not self._time_unit_logged:
                        time_unit = row.get("time_unit") or self.raw_time_unit
                        if time_unit is None and not self._time_unit_warned:
                            print(
                                "[raw_v1][time] Warning: time_unit not set; "
                                "assuming samples@1000 (legacy behavior)",
                                flush=True,
                            )
                            self._time_unit_warned = True
                            time_unit = "samples@1000"
                        duration_sec = float(end_idx_clamped - start_idx_clamped) / fs_raw
                        print(
                            "[raw_v1][time] "
                            f"raw_sfreq={fs_raw:.2f} time_unit={time_unit} "
                            f"trial_offset_sec={offset_sec:.3f} "
                            f"start_idx={start_idx_clamped} end_idx={end_idx_clamped} "
                            f"trial_duration_sec={duration_sec:.3f}",
                            flush=True,
                        )
                        self._time_unit_logged = True
                    seg = raw62.get_data(start=start_idx_clamped, stop=end_idx_clamped).astype(np.float32)

                    if fs_used != fs_raw:
                        seg_resampled = self._resample_chunked(seg, fs_from=fs_raw, fs_to=fs_used)
                        seg = seg_resampled.astype(np.float32, copy=False)

                    n_samples = seg.shape[1]
                    win_slices = window_slices(
                        n_samples,
                        fs_used,
                        self.raw_window_sec,
                        self.raw_window_hop_sec,
                    )
                    n_windows = len(win_slices)

                    n_channels = seg.shape[0]
                    band_window_vecs = None
                    band_window_covs = None
                    if self.raw_seq_save_format == "vec_utri":
                        tri_dim = n_channels * (n_channels + 1) // 2
                        band_window_vecs = [
                            np.empty((n_windows, tri_dim), dtype=np.float32) for _ in bands
                        ]
                    else:
                        band_window_covs = [
                            np.empty((n_windows, n_channels, n_channels), dtype=np.float32)
                            for _ in bands
                        ]

                    for b_idx, band in enumerate(bands):
                        band_data = bandpass(
                            seg,
                            fs_used,
                            band,
                            chunk_size=int(self.raw_filter_chunk or 0),
                        )
                        for w_idx, (w_start, w_end) in enumerate(win_slices):
                            window = band_data[:, w_start:w_end]
                            cov = cov_shrink(window, method=self.raw_cov)
                            if self.raw_seq_save_format == "cov_spd":
                                cov = 0.5 * (cov + cov.T)
                                eps_val, _base = compute_spd_eps(
                                    cov,
                                    mode=self.spd_eps_mode,
                                    absolute=float(self.spd_eps),
                                    alpha=float(self.spd_eps_alpha),
                                    floor_mult=float(self.spd_eps_floor_mult),
                                    ceil_mult=float(self.spd_eps_ceil_mult),
                                )
                                eps_by_band[band.name].append(float(eps_val))
                                cov = cov + np.eye(n_channels, dtype=cov.dtype) * float(eps_val)
                                cov = cov.astype(np.float32, copy=False)
                                if not trial0_debug_done and b_idx == 0 and w_idx == 0:
                                    eigvals = np.linalg.eigvalsh(cov)
                                    print(
                                        f"[raw_v1][trial0][band0][window0] cov_eig_min={eigvals.min():.6e} "
                                        f"cov_eig_max={eigvals.max():.6e}"
                                    )
                                band_window_covs[b_idx][w_idx] = cov
                            else:
                                if not trial0_debug_done and b_idx == 0 and w_idx == 0:
                                    eigvals = np.linalg.eigvalsh(cov)
                                    print(
                                        f"[raw_v1][trial0][band0][window0] cov_eig_min={eigvals.min():.6e} "
                                        f"cov_eig_max={eigvals.max():.6e}"
                                    )
                                logm = logmap_spd(cov, self.raw_logmap_eps)
                                vec = vec_utri(logm).astype(np.float32)
                                if not trial0_debug_done and b_idx == 0 and w_idx == 0:
                                    print(
                                        f"[raw_v1][trial0][band0][window0] vec stats "
                                        f"min={vec.min():.6f} max={vec.max():.6f} "
                                        f"mean={vec.mean():.6f} std={vec.std():.6f}"
                                    )
                                band_window_vecs[b_idx][w_idx] = vec
                        del band_data

                    if self.raw_seq_save_format == "cov_spd":
                        trial_seq = np.stack(band_window_covs, axis=1)
                        if trial_seq.ndim != 4:
                            raise ValueError("Expected per-trial sequence features to be 4D")
                    else:
                        trial_seq = np.concatenate(band_window_vecs, axis=1)
                        if trial_seq.ndim != 2:
                            raise ValueError("Expected per-trial sequence features to be 2D")
                    if not trial0_debug_done:
                        if self.raw_seq_save_format == "cov_spd":
                            print(
                                f"[raw_v1] trial0 windows={n_windows} X_trial_shape={trial_seq.shape[1:]}"
                            )
                        else:
                            print(
                                f"[raw_v1] trial0 windows={n_windows} X_trial_dim={trial_seq.shape[1]}"
                            )
                        trial0_debug_done = True

                    idx = int(row["_index"])
                    trial_path = f"{out_prefix}_trial_{idx}.npy"
                    np.save(trial_path, trial_seq)
                    file_paths[idx] = trial_path
                    seq_lens[idx] = int(trial_seq.shape[0])
                    if self.raw_seq_save_format == "cov_spd":
                        feature_shape = tuple(int(v) for v in trial_seq.shape[1:])
                        if seq_feature_shape is None:
                            seq_feature_shape = feature_shape
                        elif seq_feature_shape != feature_shape:
                            raise ValueError("sequence feature shape mismatch across trials")
                    else:
                        if feat_dim is None:
                            feat_dim = int(trial_seq.shape[1])
                        elif feat_dim != int(trial_seq.shape[1]):
                            raise ValueError("feature_dim mismatch across trials")

                    y_memmap[idx] = int(row["label"])
                    trial_id = f"{row['subject']}_s{row['session']}_t{row['trial']}"
                    meta_trials[idx] = {
                        "subject": row["subject"],
                        "session": row["session"],
                        "trial": row["trial"],
                        "trial_id": trial_id,
                        "label": int(row["label"]),
                        "t_start_s_raw": t_start_raw,
                        "t_end_s_raw": t_end_raw,
                        "t_start_s_adj": t_start,
                        "t_end_s_adj": t_end,
                        "start_idx_raw": start_idx_raw,
                        "end_idx_raw": end_idx_raw,
                        "start_idx": start_idx_clamped,
                        "end_idx": end_idx_clamped,
                        "trial_offset_sec": offset_sec,
                        "time_unit": row.get("time_unit") or self.raw_time_unit,
                        "n_windows": int(n_windows),
                        "len_T": int(n_samples),
                        "source_cnt_path": cnt_path,
                    }

                    trial_counter += 1
                    if self.raw_mem_interval and trial_counter % int(self.raw_mem_interval) == 0:
                        self._log_mem("TRIAL_PROGRESS", f"trial={trial_counter}")

                    del seg
                    del trial_seq
                    gc.collect()

                self._log_mem("CNT_DONE_BEFORE_CLOSE", os.path.basename(cnt_path))
                if hasattr(raw, "close"):
                    raw.close()
                del raw
                del raw62
                gc.collect()
                self._log_mem("CNT_DONE_AFTER_CLOSE", os.path.basename(cnt_path))

            self._log_mem("SUBJECT_DONE_AFTER_CLOSE", subject_tag)
            if self.raw_chunk_by == "subject":
                gc.collect()

        if X_memmap is not None:
            X_memmap.flush()
        y_memmap.flush()

        valid_indices = [i for i, m in enumerate(meta_trials) if m is not None]
        if not valid_indices:
            raise ValueError("No trials were processed; all trials were skipped")
        skip_count = len(meta_trials) - len(valid_indices)
        if skip_count:
            print(
                f"[raw_v1] skipped_trials={skip_count} (see meta skip_trials)",
                flush=True,
            )

        file_paths = [file_paths[i] for i in valid_indices]
        seq_lens = np.asarray(seq_lens, dtype=np.int64)[valid_indices]
        meta_trials = [meta_trials[i] for i in valid_indices]
        y_trials = np.asarray(y_memmap, dtype=np.int64)[valid_indices]

        eps_stats = None
        eps_meta_path = None
        if self.raw_seq_save_format == "cov_spd":
            eps_stats = {}
            for band in bands:
                vals = np.asarray(eps_by_band.get(band.name, []), dtype=np.float64)
                if vals.size == 0:
                    eps_stats[band.name] = {
                        "count": 0,
                        "min": None,
                        "max": None,
                        "p50": None,
                        "p95": None,
                    }
                    continue
                eps_stats[band.name] = {
                    "count": int(vals.size),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                }
            eps_meta = {
                "spd_eps_mode": self.spd_eps_mode,
                "spd_eps": float(self.spd_eps),
                "spd_eps_alpha": float(self.spd_eps_alpha),
                "spd_eps_floor_mult": float(self.spd_eps_floor_mult),
                "spd_eps_ceil_mult": float(self.spd_eps_ceil_mult),
                "bands": [b.name for b in bands],
                "eps_stats": eps_stats,
            }
            eps_meta_path = f"{out_prefix}_eps.meta.json"
            with open(eps_meta_path, "w", encoding="utf-8") as f:
                json.dump(eps_meta, f, ensure_ascii=True, indent=2)

        if sequence_mode:
            if any(p is None for p in file_paths):
                raise ValueError("Some trials were not saved; file_paths contains None")
            seq_lens = np.asarray(seq_lens, dtype=np.int64)
            feat_dim_val = int(feat_dim) if feat_dim is not None else 0
            max_len = int(seq_lens.max()) if seq_lens.size else 0
            min_len = int(seq_lens.min()) if seq_lens.size else 0
            feat_desc = (
                f"feature_shape={seq_feature_shape}"
                if seq_feature_shape is not None
                else f"feat_dim={feat_dim_val}"
            )
            print(
                f"[raw_v1] X_seq_list n_trials={len(file_paths)} "
                f"{feat_desc} min_len={min_len} max_len={max_len}"
            )
        else:
            raise ValueError("sequence_mode disabled but no X_memmap writer is configured")

        if sequence_mode:
            acc_mean = None
            acc_std = None
            f1_mean = None
            f1_std = None
            split_rows: List[dict] = []

            if save_trial == "yes":
                np.save(f"{out_prefix}_seq_lens.npy", seq_lens)
                np.save(f"{out_prefix}_y_trial.npy", np.asarray(y_trials))
                manifest_path = f"{out_prefix}_seq_manifest.json"
                manifest = {
                    "trials": meta_trials,
                    "file_paths": file_paths,
                }
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)

            meta = {
                "dataset": "seed1",
                "raw_root": os.path.abspath(self.seed_raw_root),
                "raw_manifest": self.raw_manifest_used or self.raw_manifest,
                "raw_chunk_by": self.raw_chunk_by,
                "raw_subject_list": self.raw_subject_list,
                "raw_max_subjects": int(self.raw_max_subjects) if self.raw_max_subjects else 0,
                "raw_save_trial": save_trial,
                "raw_mem_debug": int(self.raw_mem_debug),
                "raw_backend": self.raw_backend,
                "x_memmap_path": None,
                "y_memmap_path": f"{out_prefix}_y_trial.memmap",
                "fs_raw": fs_raw_global,
                "fs_used": fs_used_global,
                "trial_offset_sec": float(self.raw_trial_offset_sec),
                "window_sec": self.raw_window_sec,
                "hop_sec": self.raw_window_hop_sec,
                "bands": [f"{b.name}:{b.lo}-{b.hi}" for b in bands],
                "cov": self.raw_cov,
                "logmap_eps": self.raw_logmap_eps,
                "raw_seq_save_format": self.raw_seq_save_format,
                "spd_eps": float(self.spd_eps) if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_mode": self.spd_eps_mode if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_alpha": float(self.spd_eps_alpha) if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_floor_mult": float(self.spd_eps_floor_mult)
                if self.raw_seq_save_format == "cov_spd"
                else None,
                "spd_eps_ceil_mult": float(self.spd_eps_ceil_mult)
                if self.raw_seq_save_format == "cov_spd"
                else None,
                "spd_eps_stats": eps_stats,
                "spd_eps_meta_path": eps_meta_path,
                "clf": self.clf,
                "trial_protocol": self.trial_protocol,
                "n_trials": int(len(file_paths)),
                "sequence_mode": True,
                "feature_dim": int(feat_dim_val),
                "sequence_feature_shape": list(seq_feature_shape) if seq_feature_shape is not None else None,
                "sequence_max_len": int(max_len),
                "sequence_min_len": int(min_len),
                "labels_unique": sorted(set(y_trials.tolist())),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "channel_names": channel_names,
                "channel_hash": channel_hash,
                "skip_count": int(skip_count),
                "skip_trials": skip_trials,
                "trials": meta_trials,
                "seq_manifest_path": f"{out_prefix}_seq_manifest.json" if save_trial == "yes" else None,
                "splits": split_rows,
            }
            with open(f"{out_prefix}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            report = {
                "protocol": self.trial_protocol,
                "window_sec": self.raw_window_sec,
                "hop_sec": self.raw_window_hop_sec,
                "resample_fs": self.raw_resample_fs,
                "trial_offset_sec": float(self.raw_trial_offset_sec),
                "cov": self.raw_cov,
                "logmap_eps": self.raw_logmap_eps,
                "raw_seq_save_format": self.raw_seq_save_format,
                "spd_eps": float(self.spd_eps) if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_mode": self.spd_eps_mode if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_alpha": float(self.spd_eps_alpha) if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_floor_mult": float(self.spd_eps_floor_mult)
                if self.raw_seq_save_format == "cov_spd"
                else None,
                "spd_eps_ceil_mult": float(self.spd_eps_ceil_mult)
                if self.raw_seq_save_format == "cov_spd"
                else None,
                "clf": self.clf,
                "raw_backend": self.raw_backend,
                "sequence_mode": True,
                "sequence_max_len": int(max_len),
                "sequence_min_len": int(min_len),
                "n_splits": 0,
                "skip_count": int(skip_count),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "splits": split_rows,
            }
            report_json = f"{out_prefix}_report.json"
            report_csv = f"{out_prefix}_report.csv"
            with open(report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            with open(report_csv, "w", encoding="utf-8") as f:
                f.write("name,subject,n_train,n_test,acc,macro_f1\n")
            return

        splits = self._build_splits(meta_trials)
        print(f"[raw_v1] n_splits={len(splits)}")

        split_rows: List[dict] = []
        acc_list: List[float] = []
        f1_list: List[float] = []
        for split in splits:
            train_idx = np.asarray(split["train_idx"], dtype=np.int64)
            test_idx = np.asarray(split["test_idx"], dtype=np.int64)
            X_tr, y_tr = X_trials[train_idx], y_trials[train_idx]
            X_te, y_te = X_trials[test_idx], y_trials[test_idx]

            clf_name = self.clf.lower()
            if clf_name == "ridge":
                clf = RidgeClassifier()
            elif clf_name == "svm_linear":
                clf = LinearSVC()
            else:
                raise ValueError(f"Unknown clf: {self.clf}")

            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            acc = float(accuracy_score(y_te, pred))
            f1 = float(f1_score(y_te, pred, average="macro"))
            acc_list.append(acc)
            f1_list.append(f1)

            split_rows.append(
                {
                    "name": split["name"],
                    "subject": split["subject"],
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "acc": acc,
                    "macro_f1": f1,
                }
            )
            print(
                f"[raw_v1][{split['name']}] train={len(train_idx)} test={len(test_idx)} "
                f"acc={acc:.4f} macro_f1={f1:.4f}"
            )

        acc_mean = float(np.mean(acc_list)) if acc_list else 0.0
        acc_std = float(np.std(acc_list)) if acc_list else 0.0
        f1_mean = float(np.mean(f1_list)) if f1_list else 0.0
        f1_std = float(np.std(f1_list)) if f1_list else 0.0
        print(
            f"[raw_v1] overall acc={acc_mean:.4f}±{acc_std:.4f} "
            f"macro_f1={f1_mean:.4f}±{f1_std:.4f}"
        )

        if save_trial == "yes":
            np.save(f"{out_prefix}_X_trial.npy", np.asarray(X_trials))
            np.save(f"{out_prefix}_y_trial.npy", np.asarray(y_trials))

        meta = {
            "dataset": "seed1",
            "raw_root": os.path.abspath(self.seed_raw_root),
            "raw_manifest": self.raw_manifest_used or self.raw_manifest,
            "raw_chunk_by": self.raw_chunk_by,
            "raw_subject_list": self.raw_subject_list,
            "raw_max_subjects": int(self.raw_max_subjects) if self.raw_max_subjects else 0,
            "raw_save_trial": save_trial,
            "raw_mem_debug": int(self.raw_mem_debug),
            "raw_backend": self.raw_backend,
            "x_memmap_path": f"{out_prefix}_X_trial.memmap" if X_memmap is not None else None,
            "y_memmap_path": f"{out_prefix}_y_trial.memmap",
            "fs_raw": fs_raw_global,
            "fs_used": fs_used_global,
            "trial_offset_sec": float(self.raw_trial_offset_sec),
            "window_sec": self.raw_window_sec,
            "hop_sec": self.raw_window_hop_sec,
                "bands": [f"{b.name}:{b.lo}-{b.hi}" for b in bands],
                "cov": self.raw_cov,
                "logmap_eps": self.raw_logmap_eps,
                "spd_eps": float(self.spd_eps) if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_mode": self.spd_eps_mode if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_alpha": float(self.spd_eps_alpha) if self.raw_seq_save_format == "cov_spd" else None,
                "spd_eps_floor_mult": float(self.spd_eps_floor_mult)
                if self.raw_seq_save_format == "cov_spd"
                else None,
                "spd_eps_ceil_mult": float(self.spd_eps_ceil_mult)
                if self.raw_seq_save_format == "cov_spd"
                else None,
                "spd_eps_stats": eps_stats,
                "spd_eps_meta_path": eps_meta_path,
                "clf": self.clf,
                "trial_protocol": self.trial_protocol,
                "n_trials": int(X_trials.shape[0]),
            "feature_dim": int(X_trials.shape[1]),
            "labels_unique": sorted(set(y_trials.tolist())),
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "channel_names": channel_names,
            "channel_hash": channel_hash,
            "skip_count": int(skip_count),
            "skip_trials": skip_trials,
            "trials": meta_trials,
            "splits": split_rows,
        }
        with open(f"{out_prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        report = {
            "protocol": self.trial_protocol,
            "window_sec": self.raw_window_sec,
            "hop_sec": self.raw_window_hop_sec,
            "resample_fs": self.raw_resample_fs,
            "trial_offset_sec": float(self.raw_trial_offset_sec),
            "cov": self.raw_cov,
            "logmap_eps": self.raw_logmap_eps,
            "clf": self.clf,
            "raw_backend": self.raw_backend,
            "spd_eps": float(self.spd_eps) if self.raw_seq_save_format == "cov_spd" else None,
            "spd_eps_mode": self.spd_eps_mode if self.raw_seq_save_format == "cov_spd" else None,
            "spd_eps_alpha": float(self.spd_eps_alpha) if self.raw_seq_save_format == "cov_spd" else None,
            "spd_eps_floor_mult": float(self.spd_eps_floor_mult)
            if self.raw_seq_save_format == "cov_spd"
            else None,
            "spd_eps_ceil_mult": float(self.spd_eps_ceil_mult)
            if self.raw_seq_save_format == "cov_spd"
            else None,
            "n_splits": len(split_rows),
            "skip_count": int(skip_count),
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "splits": split_rows,
        }
        report_json = f"{out_prefix}_report.json"
        report_csv = f"{out_prefix}_report.csv"
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(report_csv, "w", encoding="utf-8") as f:
            f.write("name,subject,n_train,n_test,acc,macro_f1\n")
            for row in split_rows:
                f.write(
                    f"{row['name']},{row['subject']},{row['n_train']},"
                    f"{row['n_test']},{row['acc']:.6f},{row['macro_f1']:.6f}\n"
                )
