# ACT: Augmented Covariance Transport (Reference Protocol v1)

This repository contains the official implementation of **Augmented Covariance Transport (ACT)**, a framework for geometric data augmentation via the SPD Manifold Bridge. This version serves as the **ACT v1 Reference Protocol**.

## 🚀 Theoretical Narrative: Manifold Transversality
ACT reimagines data augmentation as a **preconditioning problem** in the latent geometric space. A key finding of the Reference Protocol is that ACT candidates exhibit a **Conflict Rate $\approx 0.5$**, implying they are **Transversal** to the current optimization path. 

- **ACT-Theory (Pure ACL)**: Acts as lateral representation regularization.
- **ACT-Perform (Hybrid ACL)**: Injects semantic anchors to shape decision boundaries.

## 📐 Mathematical Formulation

### 1. Sample Representation (Manifold Embedding)
Trials $x \in \mathbb{R}^{C \times T}$ are mapped to the SPD tangent space through matrix logarithms:
$$z(x) = \text{vec}_{utri}(\log \Sigma(x) - \bar{L})$$

### 2. Class-Conditioned LRAES Basis
The class-conditional basis $U_c$ resolves the "expansion-risk" trade-off via generalized eigenvalue decomposition:
$$M_c = (S_{\text{expand},c} + \lambda I) - \beta(S_{\text{risk},c} + \lambda I)$$
$$U_c = [u_{c,1}, \dots, u_{c,K}] \leftarrow \text{top-K eigenvectors of } M_c$$

### 3. Geometric Bridge
Candidates in the tangent space $\Delta z_{i,m}$ are realized back to raw signals via the Whitening-Coloring Bridge $B$:
$$x_{i,m}^{\text{cand}} = B(x_i, \Sigma_i, \Sigma_{i,m}^{\text{cand}})$$

### 4. ACL Scoring (Hard Positive Mining)
We score candidates based on alignment $\Phi$, entropy shift $H$, and manifold fidelity $\tau$:
$$s_{\text{hp}} = \text{Normalize}(\alpha \Phi + (1-\alpha) H) \cdot \exp(-\tau \cdot \mathcal{D}_{\text{LogEuc}})$$

---

## ⚖️ Dual-Track Objectives
1.  **Hybrid Mode (ACT-Perform)**: $\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{CE}}^{\text{orig}} + \mathcal{L}_{\text{CE}}^{\text{aug}} + \lambda_{\text{acl}}\mathcal{L}_{\text{SupCon}}$
2.  **Pure Mode (ACT-Theory)**: $\mathcal{L}_{\text{pure}} = \mathcal{L}_{\text{CE}}^{\text{orig}} + \lambda_{\text{acl}}\mathcal{L}_{\text{SupCon}}$

## 📂 Project Structure
- `run_act_pilot.py`: Main entry for both tracks.
- `core/`: Core geometry engine (Bridge, PIA, LRAES).
- `scripts/`: Analytical tools for v2 taxonomy and failure taxonomy auditing.

## ⚡ Quick Start
```bash
python run_act_pilot.py --all-datasets --seeds 1,2,3 --pipeline gcg_acl --acl-aug-ce-mode selected
```
