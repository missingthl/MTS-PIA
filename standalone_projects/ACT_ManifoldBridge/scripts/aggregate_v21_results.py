import pandas as pd
import os

results_root = "/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results/v2.1_test"
mapping = [
    {"ds": "atrialfibrillation", "arm": "Pure", "folder": "atrial_ref_pure"},
    {"ds": "atrialfibrillation", "arm": "Hybrid (v1)", "folder": "atrial_ref_hybrid"},
    {"ds": "atrialfibrillation", "arm": "v2.1 (τ=0)", "folder": "atrial_gated_t0"},
    {"ds": "atrialfibrillation", "arm": "v2.1 (τ=0.1)", "folder": "atrial_gated_t01"},
    {"ds": "heartbeat", "arm": "Hybrid (v1)", "folder": "heartbeat_ref_hybrid"},
    {"ds": "heartbeat", "arm": "v2.1 (τ=0)", "folder": "heartbeat_gated_t0"},
]

records = []
for item in mapping:
    csv_path = os.path.join(results_root, item["folder"], "sweep_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure we only take the relevant dataset from the csv
        df = df[df["dataset"] == item["ds"]]
        if len(df) > 0:
            records.append({
                "Dataset": item["ds"],
                "Arm": item["arm"],
                "Base F1 (avg)": f"{df['base_f1'].mean():.4f}",
                "ACT F1 (avg)": f"{df['act_f1'].mean():.4f}",
                "Gain % (avg)": f"{df['f1_gain_pct'].mean():+.2f}%",
                "Mean w_aug": f"{df.get('mean_aug_ce_weight', 1.0).mean():.3f}",
                "Zero Weight %": f"{df.get('zero_weight_fraction', 0.0).mean()*100:.1f}%"
            })

summary_df = pd.DataFrame(records)
header = "| " + " | ".join(summary_df.columns) + " |"
separator = "| " + " | ".join(["---"] * len(summary_df.columns)) + " |"
rows = []
for _, row in summary_df.iterrows():
    rows.append("| " + " | ".join([str(val) for val in row.values]) + " |")

md_output = "# ACT v2.1 Analysis: Evidence Table\n\n"
md_output += "> [!NOTE]\n"
md_output += "> This table summarizes the performance of ACT v2.1 across different regimes. \n"
md_output += "> `v2.1 (τ=0)` represents the proposed 'Augmentation Feedback Control' strategy.\n\n"
md_output += "\n".join([header, separator] + rows)

output_path = os.path.join(results_root, "ACT_v2.1_EVIDENCE_TABLE.md")
with open(output_path, "w") as f:
    f.write(md_output)

print(f"Table successfully saved to {output_path}")
print("\n" + md_output)
