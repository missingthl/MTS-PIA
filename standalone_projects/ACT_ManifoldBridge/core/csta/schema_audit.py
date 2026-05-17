from __future__ import annotations

"""Lightweight schema checks for CSTA result-field plumbing.

The project still uses dict-based metadata for compatibility with historical
CSV outputs.  These checks make that looser protocol safer by reporting field
duplication and passthrough coverage without changing runtime behavior.
"""

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

from core.csta import result_schema


FIELD_GROUPS: Mapping[str, Sequence[str]] = {
    "AG_RESULT_FIELDS": result_schema.AG_RESULT_FIELDS,
    "CS_FLOW_RESULT_FIELDS": result_schema.CS_FLOW_RESULT_FIELDS,
    "LATENT_RESIDUAL_RESULT_FIELDS": result_schema.LATENT_RESIDUAL_RESULT_FIELDS,
    "TASK_GUIDED_LATENT_RESULT_FIELDS": result_schema.TASK_GUIDED_LATENT_RESULT_FIELDS,
    "LC_LATENT_RESULT_FIELDS": result_schema.LC_LATENT_RESULT_FIELDS,
    "SPG_RESULT_FIELDS": result_schema.SPG_RESULT_FIELDS,
    "GI_SPG_RESULT_FIELDS": result_schema.GI_SPG_RESULT_FIELDS,
    "SPG_CFM_RESULT_FIELDS": result_schema.SPG_CFM_RESULT_FIELDS,
}

# Shared fields that intentionally appear in more than one mechanism family.
ALLOWED_CROSS_GROUP_DUPLICATES = {
    "spg_zhead_train_acc",
}


def _duplicates(values: Iterable[str]) -> Dict[str, int]:
    return {key: count for key, count in Counter(values).items() if count > 1}


def audit_result_schema() -> Dict[str, object]:
    per_group_duplicates = {name: _duplicates(fields) for name, fields in FIELD_GROUPS.items()}
    all_fields: List[str] = []
    for fields in FIELD_GROUPS.values():
        all_fields.extend(fields)
    cross_group_duplicates = _duplicates(all_fields)
    unexpected_cross_group_duplicates = {
        key: count
        for key, count in cross_group_duplicates.items()
        if key not in ALLOWED_CROSS_GROUP_DUPLICATES
    }
    return {
        "field_group_count": len(FIELD_GROUPS),
        "total_group_field_entries": len(all_fields),
        "unique_group_fields": len(set(all_fields)),
        "per_group_duplicates": per_group_duplicates,
        "cross_group_duplicates": cross_group_duplicates,
        "allowed_cross_group_duplicates": sorted(ALLOWED_CROSS_GROUP_DUPLICATES),
        "unexpected_cross_group_duplicates": unexpected_cross_group_duplicates,
        "ok": not any(per_group_duplicates.values()) and not unexpected_cross_group_duplicates,
    }


def audit_runner_passthrough_fields(passthrough_fields: Sequence[str]) -> Dict[str, object]:
    passthrough_duplicates = _duplicates(passthrough_fields)
    schema_fields = set(result_schema.CSTA_GENERATION_ENGINE_FIELDS)
    passthrough_set = set(passthrough_fields)
    missing_generation_fields = sorted(schema_fields - passthrough_set)
    return {
        "passthrough_field_entries": len(passthrough_fields),
        "passthrough_unique_fields": len(passthrough_set),
        "passthrough_duplicates": passthrough_duplicates,
        "missing_generation_fields": missing_generation_fields,
        "ok": not passthrough_duplicates and not missing_generation_fields,
    }
