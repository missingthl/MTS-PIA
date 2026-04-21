import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_waveforms(npz_path, out_dir='results/viz_harvest/plots', num_dim=3):
    data = np.load(npz_path)
    X_orig = data['X_orig_raw'] # [N, C, T]
    X_aug = data['X_aug_raw']   # [M, C, T]
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Pick a few representative samples for Class 0
    # BasicMotions: dimensions are often 6 (accel + gyro)
    num_vars = min(num_dim, X_orig.shape[1])
    
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 3*num_vars), sharex=True, dpi=300)
    if num_vars == 1:
        axes = [axes]
        
    t = np.arange(X_orig.shape[2])
    
    for c in range(num_vars):
        # Plot several original samples in light gray
        for i in range(min(5, len(X_orig))):
            axes[c].plot(t, X_orig[i, c, :], color='gray', alpha=0.2, linewidth=1)
        
        # Plot mean of original
        axes[c].plot(t, np.mean(X_orig[:, c, :], axis=0), color='black', label='Orig Mean', linewidth=2)
        
        # Plot augmented samples in color
        aug_color = 'tab:orange'
        for j in range(min(5, len(X_aug))):
            axes[c].plot(t, X_aug[j, c, :], color=aug_color, alpha=0.3, linewidth=1)
            
        # Plot mean of augmented
        axes[c].plot(t, np.mean(X_aug[:, c, :], axis=0), color='red', label='ACT Aug Mean', linewidth=2)
        
        axes[c].set_ylabel(f"Ch {c}")
        axes[c].grid(True, alpha=0.2)
        if c == 0:
            axes[c].legend()

    plt.suptitle(f"ACT Waveform Synthesis: {os.path.basename(npz_path)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(out_dir, "waveform_comparison.png")
    plt.savefig(save_path)
    print(f"Waveform plot saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--dim', type=int, default=3)
    args = parser.parse_args()
    plot_waveforms(args.path, num_dim=args.dim)
