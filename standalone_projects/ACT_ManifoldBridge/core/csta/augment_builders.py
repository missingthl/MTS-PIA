from __future__ import annotations

"""Compatibility exports for CSTA augmentation builders.

Implementation now lives in mechanism-specific modules:

- ``act_builder``: base ACT/LRAES/zPIA realized augmentation and feedback scoring.
- ``template_pool_builder``: zPIA/PIA top1/topK/template-pool augmentation.
- ``rc4_osf_builders``: legacy RC4/OSF fused and multi-z variants.

Keeping this shim avoids touching historical imports while making the actual
algorithm paths easier to audit.
"""

from .act_builder import (
    _attach_feedback_scores_to_aug_out,
    _build_act_realized_augmentations,
    _score_aug_margins,
)
from .rc4_osf_builders import (
    _apply_rc4_safe_governance,
    _build_rc4_fused_aug_out,
    _build_rc4_multiz_fused_aug_out,
    _clone_args_with_updates,
    _project_rank1_structure_out,
    _project_spectral_structure_out,
    _summarize_osf_audit_rows,
    _summarize_spectral_audit_rows,
)
from .template_pool_builder import (
    _build_zpia_template_pool_aug_out,
    _template_response_diagnostics,
)

__all__ = [
    "_apply_rc4_safe_governance",
    "_attach_feedback_scores_to_aug_out",
    "_build_act_realized_augmentations",
    "_build_rc4_fused_aug_out",
    "_build_rc4_multiz_fused_aug_out",
    "_build_zpia_template_pool_aug_out",
    "_clone_args_with_updates",
    "_project_rank1_structure_out",
    "_project_spectral_structure_out",
    "_score_aug_margins",
    "_summarize_osf_audit_rows",
    "_summarize_spectral_audit_rows",
    "_template_response_diagnostics",
]
