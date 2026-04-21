# ACT: Manifold Bridge Augmentation

This is a standalone, self-consistent engineering project for **Manifold Bridge Augmentation (ACT)**. 
It isolates the geometric data augmentation logic from the MTS-PIA framework.

## Narrative
ACT focuses on **Data Augmentation**. We perform geometric operations in the latent SPD manifold (Log-Euclidean space) and then use a **Whitening-Coloring Bridge** to map these variations back to the raw signal domain.

## Project Structure
- `run_act_pilot.py`: Main entry point for experiments.
- `core/`: Core mathematical logic (Bridge, PIA, Curriculum).
- `utils/`: Utility functions for datasets and evaluation.
- `results/`: Contains verified experimental results and the paper report.

## Usage
```bash
python run_act_pilot.py --all-datasets --seeds 1,2,3 --algo lraes --model resnet1d
```

## Scientific Results
The framework has been validated across **21 MTS datasets**. The final results are stored in `results/paper_report/sweep_results.csv`. Key performance gains were observed on:
- **Handwriting**: +5.7% (avg)
- **JapaneseVowels**: +4.8% (avg)
- **UWaveGestureLibrary**: +2.7% (avg)
- **NATOPS**: +2.8% (avg)

## Requirements
- `torch >= 2.0.0`
- `numpy >= 1.26.0`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `aeon >= 1.0.0`

## Setup
You can set up the environment using either Pip or Conda:

**Option 1: Pip**
```bash
pip install -r requirements.txt
```

**Option 2: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate act
```
