import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Add TimeVQVAE to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIME_VQVAE_PATH = os.path.join(PROJECT_ROOT, "external/TimeVQVAE/src")
if TIME_VQVAE_PATH not in sys.path:
    sys.path.append(TIME_VQVAE_PATH)

from timevqvae.vqvae import VQVAE
from timevqvae.maskgit import MaskGIT, PriorModelConfig

class TimeVQVAENumpyDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, C, T)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # TimeVQVAE expects class_condition as (batch, 1) during training
        return self.X[idx], self.y[idx].unsqueeze(0)

def fit_sample_timevqvae(X_train_ct, y_train, multiplier, seed, device="cpu", 
                         vqvae_epochs=100, maskgit_epochs=100, batch_size=64):
    """
    X_train_ct: (N, C, T)
    """
    N, C, T = X_train_ct.shape
    num_classes = len(np.unique(y_train))
    device_obj = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    
    # 1. Setup Models
    # Heuristics for TimeVQVAE params based on T
    # n_fft should be small for short sequences
    n_fft = 4
    if T < 16: n_fft = 2
    
    vqvae = VQVAE(
        in_channels=C,
        input_length=T,
        n_fft=n_fft,
        init_dim=4,
        hid_dim=64,
        downsampled_width_l=min(8, T // 4),
        downsampled_width_h=min(32, T // 2),
        encoder_n_resnet_blocks=2,
        decoder_n_resnet_blocks=2,
        codebook_size_l=256,
        codebook_size_h=256,
        kmeans_init=True,
        codebook_dim=8,
    ).to(device_obj)
    
    dataset = TimeVQVAENumpyDataset(X_train_ct, y_train)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, N), shuffle=True)
    
    # 2. Train VQVAE (Stage 1)
    print(f"Training TimeVQVAE Stage 1 (VQVAE) for {vqvae_epochs} epochs...")
    opt_vqvae = Adam(vqvae.parameters(), lr=1e-3)
    vqvae.train()
    for epoch in range(vqvae_epochs):
        total_loss = 0
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device_obj)
            opt_vqvae.zero_grad()
            out = vqvae(x_batch)
            recons_loss = sum(out.recons_loss.values())
            vq_loss = out.vq_losses['LF']['loss'] + out.vq_losses['HF']['loss']
            loss = recons_loss + vq_loss
            loss.backward()
            opt_vqvae.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{vqvae_epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    # 3. Setup MaskGIT (Stage 2)
    # CRITICAL: MaskGIT reads num_tokens from vqvae during __init__.
    # VQVAE only sets num_tokens after its first forward pass.
    # We already did forward passes during training, but let's be safe and ensure it's evaluated once more.
    vqvae.eval()
    with torch.no_grad():
        x_dummy, _ = next(iter(dataloader))
        vqvae(x_dummy.to(device_obj))

    maskgit = MaskGIT(
        vqvae=vqvae,
        lf_choice_temperature=10.0,
        hf_choice_temperature=0.0,
        lf_num_sampling_steps=10,
        hf_num_sampling_steps=10,
        lf_codebook_size=256,
        hf_codebook_size=256,
        transformer_embedding_dim=64,
        lf_prior_model_config=PriorModelConfig(
            hidden_dim=64, n_layers=2, heads=2, ff_mult=1, use_rmsnorm=True, p_unconditional=0.2
        ),
        hf_prior_model_config=PriorModelConfig(
            hidden_dim=32, n_layers=1, heads=1, ff_mult=1, use_rmsnorm=True, p_unconditional=0.2
        ),
        classifier_free_guidance_scale=1.0,
        n_classes=num_classes,
    ).to(device_obj)
    
    # 4. Train MaskGIT (Stage 2)
    print(f"Training TimeVQVAE Stage 2 (MaskGIT) for {maskgit_epochs} epochs...")
    opt_maskgit = Adam(maskgit.parameters(), lr=1e-3)
    maskgit.train()
    for epoch in range(maskgit_epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device_obj)
            y_batch = y_batch.to(device_obj)
            opt_maskgit.zero_grad()
            losses = maskgit(x_batch, y_batch)
            loss = losses.total_mask_prediction_loss
            loss.backward()
            opt_maskgit.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{maskgit_epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    # 4. Sample
    print(f"Sampling augmented data with TimeVQVAE (multiplier={multiplier})...")
    maskgit.eval()
    X_aug_list = []
    y_aug_list = []
    
    with torch.no_grad():
        for cls in sorted(np.unique(y_train)):
            n_cls = np.sum(y_train == cls)
            num_to_sample = int(n_cls * multiplier)
            if num_to_sample == 0: continue
            
            # Sample in batches
            batch_gen = 64
            samples_c = []
            for i in range(0, num_to_sample, batch_gen):
                curr_n = min(batch_gen, num_to_sample - i)
                token_ids_l, token_ids_h = maskgit.iterative_decoding(
                    num_samples=curr_n,
                    mode="cosine",
                    class_condition=int(cls),
                    device=device_obj,
                )
                x_l = maskgit.decode_token_ind_to_timeseries(token_ids_l, frequency="lf")
                x_h = maskgit.decode_token_ind_to_timeseries(token_ids_h, frequency="hf")
                x_gen = x_l + x_h
                samples_c.append(x_gen.detach().cpu().numpy())
            
            X_aug_list.append(np.concatenate(samples_c, axis=0))
            y_aug_list.append(np.full(num_to_sample, cls))
            
    if not X_aug_list:
        return np.empty((0, C, T)), np.empty((0,))
        
    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    
    return X_aug, y_aug
