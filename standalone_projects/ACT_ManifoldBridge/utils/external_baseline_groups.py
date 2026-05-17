"""Paper-facing groups for external augmentation baselines.

These groups are intentionally separate from runtime phases.  A phase describes
when an arm entered the experiment matrix; a paper group describes how it should
be presented in the external-baseline comparison table.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


TEMPORAL_VICINAL_HEURISTIC = "temporal_vicinal_heuristic"
DEEP_GENERATIVE = "deep_generative"
ALIGNMENT_STRUCTURE_PRESERVING = "alignment_structure_preserving"
INTERNAL_CONTROL = "internal_control"
PROPOSED_OR_INTERNAL_MECHANISM = "proposed_or_internal_mechanism"
OTHER = "other"


EXTERNAL_BASELINE_GROUPS: Dict[str, Tuple[str, ...]] = {
    TEMPORAL_VICINAL_HEURISTIC: (
        "raw_aug_jitter",
        "raw_aug_scaling",
        "raw_aug_timewarp",
        "raw_aug_magnitude_warping",
        "raw_aug_window_warping",
        "raw_aug_window_slicing",
        "raw_mixup",
        "raw_smote_flatten_balanced",
        "manifold_mixup",
    ),
    DEEP_GENERATIVE: (
        "timevae_classwise_optional",
        "timegan_classwise",
        "timevqvae_classwise",
        "diffusionts_classwise",
    ),
    ALIGNMENT_STRUCTURE_PRESERVING: (
        "dba_sameclass",
        "wdba_sameclass",
        "spawner_sameclass_style",
        "jobda_cleanroom",
        "rgw_sameclass",
        "dgw_sameclass",
    ),
    INTERNAL_CONTROL: (
        "no_aug",
        "random_cov_state",
        "pca_cov_state",
    ),
}


_METHOD_TO_GROUP: Dict[str, str] = {
    method: group
    for group, methods in EXTERNAL_BASELINE_GROUPS.items()
    for method in methods
}


def group_for_method(method: str) -> str:
    """Return the paper-facing external-baseline group for a method name."""

    if method.startswith("csta_") or method.startswith(
        (
            "ag_",
            "cs_flow_",
            "latent_",
            "task_guided_",
            "lc_",
            "spg_",
            "ecl_",
            "rn_ecl_",
            "gi_spg_",
        )
    ):
        return PROPOSED_OR_INTERNAL_MECHANISM
    return _METHOD_TO_GROUP.get(method, OTHER)


def methods_for_group(group: str) -> Tuple[str, ...]:
    """Return the methods that belong to a paper-facing group."""

    return EXTERNAL_BASELINE_GROUPS.get(group, ())


def known_groups() -> Tuple[str, ...]:
    """Return all stable group names accepted by CLI helpers."""

    return (
        TEMPORAL_VICINAL_HEURISTIC,
        DEEP_GENERATIVE,
        ALIGNMENT_STRUCTURE_PRESERVING,
        INTERNAL_CONTROL,
        PROPOSED_OR_INTERNAL_MECHANISM,
        OTHER,
    )


def filter_methods_by_group(methods: Iterable[str], group: str) -> Tuple[str, ...]:
    """Filter method names by paper-facing group."""

    return tuple(method for method in methods if group_for_method(method) == group)
