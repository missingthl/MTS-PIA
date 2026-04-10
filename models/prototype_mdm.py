
import torch
import numpy as np

def logm_spd(C, eps=1e-8):
    """
    Compute Matrix Logarithm of SPD matrices C using eigen-decomposition.
    C: (..., N, N) symmetric positive definite
    Returns: (..., N, N) symmetric matrix
    """
    L, V = torch.linalg.eigh(C)
    L = torch.clamp(L, min=eps)
    log_L = torch.diag_embed(torch.log(L))
    log_C = V @ log_L @ V.transpose(-2, -1)
    # Symmetrize numerical errors
    log_C = 0.5 * (log_C + log_C.transpose(-2, -1))
    return log_C

def expm_sym(S):
    """
    Compute Matrix Exponential of symmetric matrices S using eigen-decomposition.
    S: (..., N, N) symmetric
    Returns: (..., N, N) SPD matrix
    """
    L, V = torch.linalg.eigh(S)
    expm_L = torch.diag_embed(torch.exp(L))
    expm_S = V @ expm_L @ V.transpose(-2, -1)
    # Symmetrize
    expm_S = 0.5 * (expm_S + expm_S.transpose(-2, -1))
    return expm_S

def logeuclid_mean(Cs, eps=1e-8):
    """
    Compute Log-Euclidean Mean of a batch of SPD matrices.
    Cs: (B, N, N)
    Returns: (N, N) Mean SPD matrix
    Mean = expm( mean( logm(Cs) ) )
    """
    log_Cs = logm_spd(Cs, eps=eps) # (B, N, N)
    mean_log = torch.mean(log_Cs, dim=0) # (N, N)
    mean_C = expm_sym(mean_log) # (N, N)
    return mean_C, mean_log

def logeuclid_dist_pairwise(A, B):
    """
    Compute Log-Euclidean distance between set A and set B.
    A: (N_A, D, D)
    B: (N_B, D, D) or (D, D)
    Returns: (N_A, N_B) matrix of distances or (N_A,) if B is single.
    
    dist(C1, C2) = || logm(C1) - logm(C2) ||_F
    """
    # Just compute vectorized norm of difference in log domain
    # Assume A and B are already in log domain? 
    # Or expect SPD? The request implies "dist" takes matrices.
    pass 
    # To be efficient, caller should pass log matrices if calling repeatedly.
    # But for prototype classifier:
    # Train: Compute logm(Protos) once.
    # Test: Compute logm(TestSamples).
    # Distance: Norm of diff.
    
    # Let's make this function accept LOG matrices for efficiency if flagged, or check inputs.
    # But keeping it simple for now: Input SPD.
    raise NotImplementedError("Use logeuclid_dist_log_domain for efficiency.")

def logeuclid_dist_log_domain(log_A, log_B):
    """
    Compute distances between log-matrices.
    log_A: (N_A, D, D) or (D, D)
    log_B: (N_B, D, D) or (D, D)
    Returns: distances
    """
    # Ensure dimensions
    if log_A.ndim == 2: log_A = log_A.unsqueeze(0)
    if log_B.ndim == 2: log_B = log_B.unsqueeze(0)
    
    # Broadcast subtract: (N_A, 1, D, D) - (1, N_B, D, D) -> (N_A, N_B, D, D)
    diff = log_A.unsqueeze(1) - log_B.unsqueeze(0)
    
    # Frobenius Norm: sqrt(sum(diff**2))
    # dim=(-2, -1)
    # Flatten last two dims?
    dist = torch.norm(diff, p='fro', dim=(-2, -1)) # (N_A, N_B)
    
    return dist
