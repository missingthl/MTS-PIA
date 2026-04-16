# MBA: Manifold Bridge Augmentation

This is a standalone, self-consistent engineering project for **Manifold Bridge Augmentation (MBA)**. 
It isolates the geometric data augmentation logic from the MTS-PIA framework.

## Narrative
MBA focuses on **Data Augmentation**. We perform geometric operations in the latent SPD manifold (Log-Euclidean space) and then use a **Whitening-Coloring Bridge** to map these variations back to the raw signal domain.

## Project Structure
- `run_mba_pilot.py`: Main entry point for experiments.
- `core/`: Core mathematical logic (Bridge, PIA, Curriculum).
- `utils/`: Utility functions for datasets and evaluation.
- `data/`: Directory for datasets (recommend symlinking from original sources).

## Usage
```bash
python run_mba_pilot.py --dataset natops --seeds 1 --rounds 3
```

## Requirements
- `torch >= 2.0`
- `numpy`
- `pandas`
- `scikit-learn`
- `aeon`
