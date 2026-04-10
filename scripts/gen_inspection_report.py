
import os
import json
import numpy as np
import pandas as pd

ROOT = "promoted_results/phase13c4/step1/seed1/seed0"
BANDGATE_DIR = os.path.join(ROOT, "bandgate")
PROBE_PATH = os.path.join(ROOT, "_probes/input_probe.json")
OUT_PATH = os.path.join(ROOT, "INSPECTION_REPORT.md")

def load_json(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def main():
    # 1. Load Artifacts
    gate_stats = load_json(os.path.join(BANDGATE_DIR, "gate_stats.json"))
    report = load_json(os.path.join(BANDGATE_DIR, "manifold/report.json"))
    probe = load_json(PROBE_PATH)
    
    lines = []
    lines.append("# Promote Inspection: Phase 13C-4 Step 1 (Seed 0)")
    lines.append(f"**Date**: {pd.Timestamp.now()}")
    lines.append("")
    
    # 2. Artifact Paths
    lines.append("## 1. Artifacts Inspected")
    lines.append(f"- Gate Stats: `{os.path.join(BANDGATE_DIR, 'gate_stats.json')}`")
    lines.append(f"- Report: `{os.path.join(BANDGATE_DIR, 'manifold/report.json')}`")
    lines.append(f"- Probe: `{PROBE_PATH}`")
    lines.append("")
    
    # 3. Runtime Probe & Metadata
    lines.append("## 2. Runtime Configuration")
    lines.append("| Metric | Value | Source |")
    lines.append("|---|---|---|")
    
    if probe:
        lines.append(f"| bands_mode | {probe.get('bands_mode')} | Probe |")
        lines.append(f"| x_shape | {probe.get('x_shape')} | Probe |")
        lines.append(f"| contiguous | {probe.get('x_is_contiguous')} | Probe |")
        lines.append(f"| stride | {probe.get('x_stride')} | Probe |")
        lines.append(f"| T_eff | {probe.get('T_eff')} | Probe |")
    else:
        lines.append("| Probe | MISSING | Probe |")
        
    if report:
        meta = report.get('metadata', {})
        lines.append(f"| T (Report) | {meta.get('T')} | Report |")
        lines.append(f"| T_eff (Report) | {meta.get('T_eff')} | Report |")
    lines.append("")
    
    # 4. Gate Weight Summary
    lines.append("## 3. Gate Weight Summary")
    if gate_stats:
        # Aggregate all epochs? Or just last?
        # gate_stats is list of dicts: {'epoch', 'w_mean': [5], 'w_std': [5]}
        # We'll take the mean of w_mean across all entries to see global behavior
        # Also check if any band is dominating (>0.8 for example)
        
        all_means = np.array([d['w_mean'] for d in gate_stats])
        grand_mean = all_means.mean(axis=0)
        grand_std = all_means.std(axis=0)
        min_w = all_means.min()
        max_w = all_means.max()
        
        lines.append(f"- **Grand Mean**: {np.array2string(grand_mean, precision=4, separator=', ')}")
        lines.append(f"- **Grand Std**: {np.array2string(grand_std, precision=4, separator=', ')}")
        lines.append(f"- **Range**: [{min_w:.4f}, {max_w:.4f}]")
        
        # Dominance Check
        dom_idx = np.where(grand_mean > 0.8)[0]
        if len(dom_idx) > 0:
            lines.append(f"- **Collapse Warning**: Band {dom_idx} dominates.")
        else:
            lines.append("- **Distribution**: Balanced (No single band > 0.8 mean).")
    else:
        lines.append("MISSING gate stats.")
    lines.append("")
            
    # 5. Checklist
    lines.append("## 4. Pass/Fail Checklist")
    pass_mode = (probe and probe.get('bands_mode') == "all5_timecat")
    pass_shape = (probe and probe.get('x_shape') == [8, 5, 62, 24]) # B=8 usually
    pass_probe = (probe is not None)
    
    lines.append(f"- [{'x' if pass_mode else ' '}] Correct Mode Recorded (all5_timecat)")
    lines.append(f"- [{'x' if pass_shape else ' '}] Shape Matches Expected ([B, 5, 62, 24])")
    lines.append(f"- [{'x' if pass_probe else ' '}] Probe JSON Generated")
    
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Report Generated: {OUT_PATH}")

if __name__ == "__main__":
    main()
