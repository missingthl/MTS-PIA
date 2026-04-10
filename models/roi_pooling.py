
import torch
import torch.nn as nn
from datasets.seed1_channel_order import SEED1_CHANNEL_NAMES_62

# ROI Definition (Hard-coded Biology)
# K = 13 ROIs mapping rough scalp regions
ROIS = {
    "Pre-Frontal": ["Fp1", "Fpz", "Fp2", "AF3", "AF4"],
    "Left Frontal": ["F7", "F5", "F3", "F1", "FT7", "FC5", "FC3", "FC1"],
    "Right Frontal": ["F2", "F4", "F6", "F8", "FC2", "FC4", "FC6", "FT8"],
    "Mid Frontal": ["Fz", "FCz"],
    "Left Central": ["T7", "C5", "C3", "C1", "TP7", "CP5", "CP3", "CP1"],
    "Right Central": ["C2", "C4", "C6", "T8", "CP2", "CP4", "CP6", "TP8"],
    "Mid Central": ["Cz", "CPz"],
    "Left Parietal": ["P7", "P5", "P3", "P1"],
    "Right Parietal": ["P2", "P4", "P6", "P8"],
    "Mid Parietal": ["Pz"],
    "Left Occipital": ["PO7", "PO5", "PO3", "CB1", "O1"],
    "Right Occipital": ["PO4", "PO6", "PO8", "O2", "CB2"],
    "Mid Occipital": ["POz", "Oz"]
}

class ROIPooling(nn.Module):
    def __init__(self, mode="mean", strict=True):
        super().__init__()
        self.mode = mode
        self.roi_names = list(ROIS.keys())
        self.k_rois = len(self.roi_names)
        self.channel_map = {name: i for i, name in enumerate(SEED1_CHANNEL_NAMES_62)}
        
        # Build indices
        self.roi_indices = []
        covered_indices = set()
        
        for roi in self.roi_names:
            channels = ROIS[roi]
            indices = [self.channel_map[ch] for ch in channels if ch in self.channel_map]
            if strict and len(indices) != len(channels):
                missing = set(channels) - set(SEED1_CHANNEL_NAMES_62)
                raise ValueError(f"ROI {roi} has missing channels: {missing}")
            self.roi_indices.append(indices)
            covered_indices.update(indices)
            
        if strict and len(covered_indices) != 62:
             missing_coverage = set(range(62)) - covered_indices
             raise ValueError(f"ROI Pooling Coverage Incomplete! Missing indices: {missing_coverage}")
             
        # Register indices as buffers so they move with device? 
        # Actually easier to just keep them as lists and construct masks or gather indices on forward
        # For efficiency, let's pre-compute a binary mask if we do mean pooling
        # Mask: (K, 62)
        mask = torch.zeros(self.k_rois, 62)
        for i, indices in enumerate(self.roi_indices):
            mask[i, indices] = 1.0
            if mode == "mean":
                mask[i] /= len(indices) # Average
        
        self.register_buffer("roi_mask", mask)

    def forward(self, x):
        """
        Input: (B, 62, T)
        Output: (B, K=13, T)
        """
        # x: (B, 62, T) -> (B, T, 62) for matmul? 
        # Actually (K, 62) @ (62, T) -> (K, T) works if we permute/broadcast correctly.
        # Check dims: mask is (K, 62). x is (B, 62, T).
        # We want (B, K, T) = (B, K, 62) @ (B, 62, T) roughly.
        # Using torch.matmul:
        # mask.unsqueeze(0) -> (1, K, 62)
        # result = torch.matmul(mask, x) -> (B, K, T)
        
        return torch.matmul(self.roi_mask, x)

    def get_roi_info(self):
        return {
            "K": self.k_rois,
            "names": self.roi_names,
            "mapping": ROIS
        }
