import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import argparse

def plot_manifold(npz_path, out_dir='results/viz_harvest/plots'):
    data = np.load(npz_path)
    Z_orig = data['Z_orig'] # [N, D]
    y_orig = data['y_orig']
    Z_aug = data['Z_aug']   # [M, D]
    y_aug = data['y_aug']
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Concatenate for joint projection
    Z_all = np.concatenate([Z_orig, Z_aug], axis=0)
    
    print(f"Projecting {Z_all.shape} latent vectors...")
    # Use PCA first for stability if D is large
    if Z_all.shape[1] > 50:
        pca = PCA(n_components=50)
        Z_low = pca.fit_transform(Z_all)
    else:
        Z_low = Z_all
        
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(Z_low)
    
    X_orig = X_embedded[:len(Z_orig)]
    X_aug = X_embedded[len(Z_orig):]
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Plot original classes
    unique_y = np.unique(y_orig)
    colors = plt.cm.get_cmap('Set1', len(unique_y))
    
    for i, label in enumerate(unique_y):
        mask = (y_orig == label)
        plt.scatter(X_orig[mask, 0], X_orig[mask, 1], 
                    color=colors(i), label=f'Class {label} (Orig)', 
                    alpha=0.8, edgecolors='black', linewidth=0.5, s=80)
        
        # Plot secondary points for augmented data
        mask_aug = (y_aug == label)
        plt.scatter(X_aug[mask_aug, 0], X_aug[mask_aug, 1], 
                    color=colors(i), marker='X', alpha=0.4, 
                    edgecolors='white', linewidth=0.2,
                    s=60, label=f'Class {label} (ACT)')

    plt.title(f"ACT Manifold Bridge Projection (basicmotions)", fontsize=16)
    plt.xlabel("Manifold Dimension 1", fontsize=12)
    plt.ylabel("Manifold Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "latent_manifold_tsne.png")
    plt.savefig(save_path)
    print(f"Manifold plot saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    plot_manifold(args.path)
