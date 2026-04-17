# MBA: Manifold Bridge Augmentation

This is a standalone, self-consistent engineering project for **Manifold Bridge Augmentation (MBA)**. 
It isolates the geometric data augmentation logic from the MTS-PIA framework.

## Narrative
MBA focuses on **Data Augmentation**. We perform geometric operations in the latent SPD manifold (Log-Euclidean space) and then use a **Whitening-Coloring Bridge** to map these variations back to the raw signal domain.

## Project Structure
- `run_mba_pilot.py`: Main entry point for experiments.
- `core/`: Core mathematical logic (Bridge, PIA, Curriculum).
- `utils/`: Utility functions for datasets and evaluation.
- `results/`: Contains verified experimental results and the paper report.

## Usage
```bash
python run_mba_pilot.py --all-datasets --seeds 1,2,3 --algo lraes --model resnet1d
```

## Scientific Results
The framework has been validated across **21 MTS datasets**. The final results are stored in `results/paper_report/sweep_results.csv`. Key performance gains were observed on:
- **Handwriting**: +5.7% (avg)
- **JapaneseVowels**: +4.8% (avg)
- **UWaveGestureLibrary**: +2.7% (avg)
- **NATOPS**: +2.8% (avg)

## Requirements
- `torch >= 2.0`
- `numpy`
- `pandas`
- `scikit-learn`
- `aeon`
