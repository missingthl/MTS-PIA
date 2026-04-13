
import os
import csv
import glob
import pandas as pd
import numpy as np

def summarize():
    log_dir = "logs"
    out_file = "experiments/phase8_rebaseline/reports/window_learning_curves_summary.md"
    
    # Pattern: E*_seed*_diagnostics.csv
    files = glob.glob(os.path.join(log_dir, "E*_seed*_diagnostics.csv"))
    
    summary_lines = []
    summary_lines.append("# Phase 8.2: Learning Curves Summary\n")
    
    for fpath in sorted(files):
        name = os.path.basename(fpath).replace("_diagnostics.csv", "")
        try:
            df = pd.read_csv(fpath)
            # Check last 10 epochs
            last_10 = df.tail(10)
            
            if len(last_10) == 0:
                continue
                
            train_acc_mean = last_10['train_acc'].mean() if 'train_acc' in df else (last_10['train_correct']/last_10['train_total']).mean() if 'train_correct' in df else 0.0
            # ManifoldDeepRunner logs 'val_acc'. 'train_loss'?
            # Check columns in file manually if needed, but 'train_loss' and 'val_acc' are standard.
            
            train_loss_mean = last_10['train_loss'].mean()
            val_acc_mean = last_10['val_acc'].mean()
            val_acc_max = last_10['val_acc'].max()
            
            summary_lines.append(f"## {name}")
            summary_lines.append(f"- **Epochs**: {len(df)}")
            summary_lines.append(f"- **Last 10 Mean Train Loss**: {train_loss_mean:.4f}")
            if 'train_acc' in df.columns: # ManifoldDeepRunner doesn't log train_acc in diagnostics CSV usually?
                # Step 2160 view shows it logs `stats`.
                # `stats` comes from `model.last_diagnostics`.
                # But `train_loss`, `val_acc` are added to stats before writing!
                # `train_acc` is NOT added to stats in the view (Step 2160).
                # `stats['train_loss'] = train_loss`
                # `stats['val_acc'] = val_acc`
                # So `train_acc` is missing from CSV!
                # User asked for "train_acc".
                # I should have added it.
                # But I can't go back in time for the running process.
                summary_lines.append(f"- **Last 10 Mean Train Acc**: (Not collected)")
            else:
                 summary_lines.append(f"- **Last 10 Mean Train Acc**: (Not collected)")

            summary_lines.append(f"- **Last 10 Mean Val Acc**: {val_acc_mean:.4f}")
            summary_lines.append(f"- **Peak Val Acc**: {val_acc_max:.4f}")
            
            # Trend Check
            slope = 0
            if len(last_10) > 1:
                slope = np.polyfit(range(len(last_10)), last_10['val_acc'], 1)[0]
            summary_lines.append(f"- **Val Acc Trend**: {'Rising' if slope > 1e-4 else 'Flat/Falling'} (Slope: {slope:.2e})")
            summary_lines.append("")
            
        except Exception as e:
            summary_lines.append(f"## {name} (Error: {e})\n")
            
    with open(out_file, "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"Summary written to {out_file}")

if __name__ == "__main__":
    summarize()
