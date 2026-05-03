from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class LocalTangentResult:
    bases: List[Optional[np.ndarray]]
    local_neighbor_count: np.ndarray
    local_tangent_dim: np.ndarray
    local_explained_variance_ratio: np.ndarray
    tangent_available: np.ndarray
    fallback_flag: np.ndarray
    fallback_reason: List[str]


def _as_2d_float(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf values.")
    return arr


def _choose_tangent_dim(
    eigvals: np.ndarray,
    *,
    tangent_dim: int | str,
    explained_var_threshold: float,
    max_tangent_dim: int,
    z_dim: int,
) -> Tuple[int, float]:
    vals = np.asarray(eigvals, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size <= 0:
        return 0, np.nan
    upper = int(max(1, min(int(max_tangent_dim), int(vals.size), int(z_dim))))
    if isinstance(tangent_dim, str):
        if tangent_dim != "auto":
            raise ValueError("tangent_dim must be an integer or 'auto'.")
        total = float(np.sum(vals))
        if total <= 1e-12:
            return 0, np.nan
        cumulative = np.cumsum(vals) / total
        r = int(np.searchsorted(cumulative, float(explained_var_threshold), side="left") + 1)
        r = max(1, min(r, upper))
    else:
        r = max(1, min(int(tangent_dim), upper))
    ratio = float(np.sum(vals[:r]) / (np.sum(vals) + 1e-12))
    return r, ratio


def estimate_local_tangent_spaces(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    k_neighbors: int = 8,
    tangent_dim: int | str = "auto",
    class_conditioned: bool = True,
    distance: str = "euclidean",
    min_neighbors: int = 3,
    explained_var_threshold: float = 0.90,
    max_tangent_dim: int = 10,
    eps: float = 1e-12,
) -> LocalTangentResult:
    """Estimate train-only local tangent bases from same-class covariance states.

    The estimator is intentionally conservative: every anchor uses only train
    split states and, by default, only same-class neighbors.  Scarce classes are
    marked with explicit fallback metadata instead of aborting the full audit.
    """

    if distance != "euclidean":
        raise ValueError("Local tangent audit v1 only supports euclidean distance.")
    if not class_conditioned:
        raise ValueError("Local tangent audit v1 expects class_conditioned=True.")
    X = _as_2d_float(Z, name="Z")
    labels = np.asarray(y, dtype=np.int64).ravel()
    if labels.shape[0] != X.shape[0]:
        raise ValueError("y length must match Z rows.")

    n, z_dim = X.shape
    bases: List[Optional[np.ndarray]] = []
    neighbor_count = np.zeros(n, dtype=np.int64)
    tangent_dims = np.zeros(n, dtype=np.int64)
    explained = np.full(n, np.nan, dtype=np.float64)
    available = np.zeros(n, dtype=bool)
    fallback = np.zeros(n, dtype=bool)
    reasons: List[str] = []

    k_eff = max(1, int(k_neighbors))
    min_eff = max(1, int(min_neighbors))

    for i in range(n):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]
        if same.size <= 0:
            bases.append(None)
            reasons.append("singleton_class_no_same_class_neighbor")
            fallback[i] = True
            continue

        dists = np.linalg.norm(X[same] - X[i][None, :], axis=1)
        order = np.argsort(dists, kind="mergesort")
        chosen = same[order[: min(k_eff, same.size)]]
        neighbor_count[i] = int(chosen.size)
        if chosen.size < min_eff:
            fallback[i] = True
            reasons.append("insufficient_same_class_neighbors")
        else:
            reasons.append("")

        neigh = X[chosen]
        centered = neigh - np.mean(neigh, axis=0, keepdims=True)
        if centered.shape[0] <= 0 or float(np.linalg.norm(centered)) <= eps:
            bases.append(None)
            reasons[-1] = reasons[-1] or "zero_local_variance"
            fallback[i] = True
            continue

        # SVD is stable for the common n_neighbors << z_dim regime.
        _, svals, vt = np.linalg.svd(centered, full_matrices=False)
        eigvals = (svals * svals) / max(float(centered.shape[0] - 1), 1.0)
        r, ratio = _choose_tangent_dim(
            eigvals,
            tangent_dim=tangent_dim,
            explained_var_threshold=explained_var_threshold,
            max_tangent_dim=max_tangent_dim,
            z_dim=z_dim,
        )
        if r <= 0:
            bases.append(None)
            reasons[-1] = reasons[-1] or "zero_local_variance"
            fallback[i] = True
            continue

        U = vt[:r].T.astype(np.float64)
        # Numerical guard: SVD already returns orthonormal rows in V^T.
        norms = np.linalg.norm(U, axis=0, keepdims=True)
        U = U / (norms + eps)
        bases.append(U)
        tangent_dims[i] = int(r)
        explained[i] = ratio
        available[i] = True

    return LocalTangentResult(
        bases=bases,
        local_neighbor_count=neighbor_count,
        local_tangent_dim=tangent_dims,
        local_explained_variance_ratio=explained,
        tangent_available=available,
        fallback_flag=fallback,
        fallback_reason=reasons,
    )


def compute_tangent_alignment(
    direction: np.ndarray,
    U: Optional[np.ndarray],
    *,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    d = np.asarray(direction, dtype=np.float64).ravel()
    norm2 = float(np.dot(d, d))
    if U is None or norm2 <= eps or not np.isfinite(norm2):
        return np.nan, np.nan
    basis = np.asarray(U, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != d.shape[0] or basis.shape[1] <= 0:
        return np.nan, np.nan
    proj = basis.T @ d
    alignment = float(np.dot(proj, proj) / (norm2 + eps))
    alignment = float(np.clip(alignment, 0.0, 1.0))
    return alignment, float(np.clip(1.0 - alignment, 0.0, 1.0))


def top_response_template_ids(
    z: np.ndarray,
    bank: np.ndarray,
    *,
    policy: str,
    seed: int,
    anchor_index: int,
    pairs: int = 1,
) -> np.ndarray:
    D = _as_2d_float(bank, name="bank")
    zz = np.asarray(z, dtype=np.float64).ravel()
    responses = np.abs(zz @ D.T)
    order = np.lexsort((np.arange(D.shape[0]), -responses))
    if policy == "top1":
        return order[: int(pairs)]
    if policy.startswith("topk_uniform_top"):
        top_k = int(policy.split("top")[-1])
        top_ids = order[: min(top_k, D.shape[0])]
        rng = np.random.default_rng(int(anchor_index) + int(seed))
        return rng.choice(top_ids, size=(int(pairs),), replace=True)
    raise ValueError(f"Unsupported PIA audit policy: {policy}")


def build_alignment_rows(
    *,
    dataset: str,
    seed: int,
    method: str,
    Z: np.ndarray,
    y: np.ndarray,
    tangent: LocalTangentResult,
    direction_bank: Optional[np.ndarray],
    pca_bank: Optional[np.ndarray],
    multiplier: int = 10,
    policy: str = "top1",
    k_dir: int = 10,
    actual_candidate_rows: Optional[pd.DataFrame] = None,
    audit_source: str = "policy_replay",
) -> List[Dict[str, object]]:
    X = _as_2d_float(Z, name="Z")
    labels = np.asarray(y, dtype=np.int64).ravel()
    rows: List[Dict[str, object]] = []
    n, z_dim = X.shape
    mult = max(1, int(multiplier))

    def _top5_alignment_stats(anchor_index: int, selected_template_id: int) -> Dict[str, object]:
        if direction_bank is None:
            return {
                "top1_alignment": np.nan,
                "top5_alignment_mean": np.nan,
                "top5_alignment_std": np.nan,
                "selected_alignment_rank_within_top5": np.nan,
                "selected_alignment_minus_top5_mean": np.nan,
                "top5_response_mean": np.nan,
                "top5_response_std": np.nan,
            }
        D = _as_2d_float(direction_bank, name="direction_bank")
        responses = np.abs(X[anchor_index] @ D.T)
        order = np.lexsort((np.arange(D.shape[0]), -responses))
        top5 = order[: min(5, D.shape[0])]
        alignments = []
        for tid in top5:
            align, _ = compute_tangent_alignment(D[int(tid)], tangent.bases[anchor_index])
            alignments.append(align)
        arr = np.asarray(alignments, dtype=np.float64)
        top1 = float(arr[0]) if arr.size and np.isfinite(arr[0]) else np.nan
        selected_alignment = np.nan
        if int(selected_template_id) >= 0:
            selected_alignment, _ = compute_tangent_alignment(D[int(selected_template_id)], tangent.bases[anchor_index])
        if np.isfinite(selected_alignment) and arr.size:
            # Rank selected direction by alignment inside the high-response top5 set.
            selected_rank = int(1 + np.sum(arr > selected_alignment + 1e-12))
        else:
            selected_rank = np.nan
        return {
            "top1_alignment": top1,
            "top5_alignment_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,
            "top5_alignment_std": float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan,
            "selected_alignment_rank_within_top5": selected_rank,
            "selected_alignment_minus_top5_mean": float(selected_alignment - np.nanmean(arr))
            if np.isfinite(selected_alignment) and np.isfinite(arr).any()
            else np.nan,
            "top5_response_mean": float(np.mean(responses[top5])) if top5.size else np.nan,
            "top5_response_std": float(np.std(responses[top5])) if top5.size else np.nan,
        }

    def add_row(
        *,
        anchor_index: int,
        candidate_order: int,
        direction: np.ndarray,
        direction_source: str,
        template_id: int,
        template_rank: int,
        template_response_abs: float,
        sign: float,
    ) -> None:
        alignment, leakage = compute_tangent_alignment(direction, tangent.bases[anchor_index])
        top5_stats = _top5_alignment_stats(anchor_index, template_id)
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "anchor_index": int(anchor_index),
                "class_id": int(labels[anchor_index]),
                "candidate_order": int(candidate_order),
                "direction_source": direction_source,
                "template_id": int(template_id),
                "template_rank": int(template_rank),
                "template_response_abs": float(template_response_abs),
                "template_sign": float(sign),
                "local_neighbor_count": int(tangent.local_neighbor_count[anchor_index]),
                "local_tangent_dim": int(tangent.local_tangent_dim[anchor_index]),
                "local_explained_variance_ratio": float(tangent.local_explained_variance_ratio[anchor_index])
                if np.isfinite(tangent.local_explained_variance_ratio[anchor_index])
                else np.nan,
                "tangent_available": bool(tangent.tangent_available[anchor_index]),
                "tangent_alignment": alignment,
                "normal_leakage": leakage,
                "direction_norm": float(np.linalg.norm(direction)),
                "z_dim": int(z_dim),
                "fallback_flag": bool(tangent.fallback_flag[anchor_index]),
                "fallback_reason": str(tangent.fallback_reason[anchor_index]),
                "audit_source": str(audit_source),
                "actual_candidate_audit_available": bool(audit_source == "actual_candidate_audit"),
                **top5_stats,
            }
        )

    if method in {"csta_topk_uniform_top5", "csta_top1_current"}:
        if direction_bank is None:
            raise ValueError("PIA methods require direction_bank.")
        D = _as_2d_float(direction_bank, name="direction_bank")
        audit_policy = "topk_uniform_top5" if method == "csta_topk_uniform_top5" else "top1"
        actual_df = actual_candidate_rows.copy() if actual_candidate_rows is not None and not actual_candidate_rows.empty else None
        if actual_df is not None:
            required = {"anchor_index", "candidate_order", "template_id", "template_sign"}
            missing = required - set(actual_df.columns)
            if missing:
                raise ValueError(f"actual_candidate_rows missing required columns: {sorted(missing)}")
            actual_df = actual_df.sort_values(["anchor_index", "candidate_order", "slot_index" if "slot_index" in actual_df else "candidate_order"])
        for i in range(n):
            responses = np.abs(X[i] @ D.T)
            order = np.lexsort((np.arange(D.shape[0]), -responses))
            rank_lookup = {int(tid): int(rank) for rank, tid in enumerate(order.tolist())}
            if actual_df is not None:
                selected_specs = actual_df[actual_df["anchor_index"].astype(int) == int(i)]
            else:
                ids = top_response_template_ids(X[i], D, policy=audit_policy, seed=seed, anchor_index=i, pairs=1)
                template_id = int(ids[0])
                selected_specs = pd.DataFrame(
                    [
                        {
                            "candidate_order": int(c),
                            "template_id": template_id,
                            "template_rank": rank_lookup.get(template_id, -1),
                            "template_sign": 1.0 if c % 2 == 0 else -1.0,
                            "template_response_abs": float(responses[template_id]),
                        }
                        for c in range(mult)
                    ]
                )
            for _, spec in selected_specs.iterrows():
                c = int(spec.get("candidate_order", 0))
                template_id = int(spec.get("template_id", -1))
                if template_id < 0 or template_id >= D.shape[0]:
                    continue
                sign = float(spec.get("template_sign", 1.0))
                add_row(
                    anchor_index=i,
                    candidate_order=c,
                    direction=sign * D[template_id],
                    direction_source="pia_selected",
                    template_id=template_id,
                    template_rank=int(spec.get("template_rank", rank_lookup.get(template_id, -1))),
                    template_response_abs=float(spec.get("template_response_abs", responses[template_id])),
                    sign=sign,
                )

            # Add same-shape post-hoc comparators for CSTA methods without
            # changing the anchor ids.  These rows do not represent additional
            # CSTA samples; they are diagnostic reference directions.
            for c in range(mult):
                rng = np.random.default_rng(int(seed) * 1000003 + i * 1009 + c)
                direction = rng.standard_normal(z_dim)
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                add_row(
                    anchor_index=i,
                    candidate_order=c,
                    direction=direction,
                    direction_source="random_cov",
                    template_id=-1,
                    template_rank=-1,
                    template_response_abs=np.nan,
                    sign=1.0,
                )
            if pca_bank is not None:
                P = _as_2d_float(pca_bank, name="pca_bank")
                for c in range(mult):
                    pca_id = int(c % min(int(k_dir), P.shape[0]))
                    sign = 1.0 if c % 2 == 0 else -1.0
                    add_row(
                        anchor_index=i,
                        candidate_order=c,
                        direction=sign * P[pca_id],
                        direction_source="pca_cov",
                        template_id=pca_id,
                        template_rank=pca_id,
                        template_response_abs=float(abs(X[i] @ P[pca_id].T)),
                        sign=sign,
                    )
        return rows

    if method == "random_cov_state" or policy == "comparators_only":
        for i in range(n):
            global_i = int(i)
            for c in range(mult):
                rng = np.random.default_rng(int(seed) * 1000003 + global_i * 1009 + c)
                direction = rng.standard_normal(z_dim)
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                add_row(
                    anchor_index=i,
                    candidate_order=c,
                    direction=direction,
                    direction_source="random_cov",
                    template_id=-1,
                    template_rank=-1,
                    template_response_abs=np.nan,
                    sign=1.0,
                )

    if (method == "pca_cov_state" or policy == "comparators_only") and pca_bank is not None:
        P = _as_2d_float(pca_bank, name="pca_bank")
        for i in range(n):
            for c in range(mult):
                template_id = int(c % min(int(k_dir), P.shape[0]))
                sign = 1.0 if c % 2 == 0 else -1.0
                add_row(
                    anchor_index=i,
                    candidate_order=c,
                    direction=sign * P[template_id],
                    direction_source="pca_cov",
                    template_id=template_id,
                    template_rank=template_id,
                    template_response_abs=float(abs(X[i] @ P[template_id].T)),
                    sign=sign,
                )

    return rows


def summarize_candidate_audit(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    df = rows.copy()
    df["tangent_alignment"] = pd.to_numeric(df["tangent_alignment"], errors="coerce")
    df["normal_leakage"] = pd.to_numeric(df["normal_leakage"], errors="coerce")
    summaries = []
    for (dataset, seed, method), sub in df.groupby(["dataset", "seed", "method"], dropna=False):
        by_source = sub.groupby("direction_source")["tangent_alignment"].agg(["mean", "std"]).to_dict("index")
        leakage = sub.groupby("direction_source")["normal_leakage"].mean().to_dict()
        selected_mean = by_source.get("pia_selected", {}).get("mean", np.nan)
        random_mean = by_source.get("random_cov", {}).get("mean", np.nan)
        pca_mean = by_source.get("pca_cov", {}).get("mean", np.nan)
        summaries.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "n_train": int(sub["anchor_index"].nunique()),
                "z_dim": int(sub["z_dim"].iloc[0]) if "z_dim" in sub else 0,
                "n_classes": int(sub["class_id"].nunique()),
                "tangent_available_rate": float(sub.drop_duplicates("anchor_index")["tangent_available"].mean()),
                "insufficient_neighbor_count": int(
                    sub.drop_duplicates("anchor_index")["fallback_reason"].eq("insufficient_same_class_neighbors").sum()
                ),
                "fallback_count": int(sub.drop_duplicates("anchor_index")["fallback_flag"].sum()),
                "selected_alignment_mean": float(selected_mean) if pd.notna(selected_mean) else np.nan,
                "selected_alignment_std": float(by_source.get("pia_selected", {}).get("std", np.nan)),
                "random_alignment_mean": float(random_mean) if pd.notna(random_mean) else np.nan,
                "random_alignment_std": float(by_source.get("random_cov", {}).get("std", np.nan)),
                "pca_alignment_mean": float(pca_mean) if pd.notna(pca_mean) else np.nan,
                "pca_alignment_std": float(by_source.get("pca_cov", {}).get("std", np.nan)),
                "selected_minus_random_alignment": float(selected_mean - random_mean)
                if pd.notna(selected_mean) and pd.notna(random_mean)
                else np.nan,
                "selected_minus_pca_alignment": float(selected_mean - pca_mean)
                if pd.notna(selected_mean) and pd.notna(pca_mean)
                else np.nan,
                "selected_normal_leakage_mean": float(leakage.get("pia_selected", np.nan)),
                "random_normal_leakage_mean": float(leakage.get("random_cov", np.nan)),
                "pca_normal_leakage_mean": float(leakage.get("pca_cov", np.nan)),
                "actual_candidate_audit_available": bool(sub["actual_candidate_audit_available"].any())
                if "actual_candidate_audit_available" in sub
                else False,
                "audit_source": ",".join(sorted(str(x) for x in sub.get("audit_source", pd.Series(dtype=str)).dropna().unique())),
                "top1_alignment_mean": float(pd.to_numeric(sub.get("top1_alignment", np.nan), errors="coerce").mean())
                if "top1_alignment" in sub
                else np.nan,
                "top5_alignment_mean": float(pd.to_numeric(sub.get("top5_alignment_mean", np.nan), errors="coerce").mean())
                if "top5_alignment_mean" in sub
                else np.nan,
                "top5_alignment_std_mean": float(pd.to_numeric(sub.get("top5_alignment_std", np.nan), errors="coerce").mean())
                if "top5_alignment_std" in sub
                else np.nan,
                "selected_alignment_rank_within_top5_mean": float(
                    pd.to_numeric(sub[sub["direction_source"] == "pia_selected"].get("selected_alignment_rank_within_top5", np.nan), errors="coerce").mean()
                )
                if "selected_alignment_rank_within_top5" in sub
                else np.nan,
                "selected_alignment_minus_top5_mean": float(
                    pd.to_numeric(sub[sub["direction_source"] == "pia_selected"].get("selected_alignment_minus_top5_mean", np.nan), errors="coerce").mean()
                )
                if "selected_alignment_minus_top5_mean" in sub
                else np.nan,
                "top5_response_mean": float(pd.to_numeric(sub.get("top5_response_mean", np.nan), errors="coerce").mean())
                if "top5_response_mean" in sub
                else np.nan,
                "top5_response_std_mean": float(pd.to_numeric(sub.get("top5_response_std", np.nan), errors="coerce").mean())
                if "top5_response_std" in sub
                else np.nan,
            }
        )
    return pd.DataFrame(summaries)
