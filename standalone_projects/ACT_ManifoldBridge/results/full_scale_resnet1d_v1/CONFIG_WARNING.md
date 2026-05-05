# full_scale_resnet1d_v1 — CONFIG WARNING

**IMPORTANT**: This matrix uses **eta_safe = 0.5**, which is NOT the canonical
Final20 configuration (eta_safe = 0.75).

## Do not use these results in the paper main table.

The canonical Final20 results with the locked configuration (eta_safe=0.75) are at:

- `../final20_minimal_baseline_v1/resnet1d_s123/`
- `../final20_main_comparison_v1/resnet1d_s123/`

This directory is retained for historical reference and as supplementary evidence
that the method is not highly sensitive to eta_safe (CSTA-U5 at e0.5 = 0.7287
vs e0.75 = 0.7279, negligible difference).
