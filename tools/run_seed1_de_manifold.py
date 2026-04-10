import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.seed_official_de import load_seed_official_de
from runners.manifold_runner import ManifoldRunner


def _reconstruct_trials(
    X_train: np.ndarray,
    X_test: np.ndarray,
    trial_index: List[dict],
) -> Tuple[List[np.ndarray], np.ndarray, List[str], List[np.ndarray], np.ndarray, List[str]]:
    train_trials = []
    train_labels = []
    train_ids = []
    test_trials = []
    test_labels = []
    test_ids = []

    train_pos = 0
    test_pos = 0
    for row in trial_index:
        n_win = int(row["n_windows"])
        label = int(row["label"])
        trial_id = str(row["trial_id"])
        if row["split"] == "train":
            trial = X_train[train_pos : train_pos + n_win]
            train_pos += n_win
            train_trials.append(trial)
            train_labels.append(label)
            train_ids.append(trial_id)
        else:
            trial = X_test[test_pos : test_pos + n_win]
            test_pos += n_win
            test_trials.append(trial)
            test_labels.append(label)
            test_ids.append(trial_id)

    return (
        train_trials,
        np.asarray(train_labels, dtype=int),
        train_ids,
        test_trials,
        np.asarray(test_labels, dtype=int),
        test_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manifold classifier on SEED official DE.")
    parser.add_argument("--seed-de-root", required=True)
    parser.add_argument("--seed-de-var", default="de_LDS1")
    parser.add_argument("--seed-manifest", default="logs/seed_raw_trial_manifest_full.json")
    parser.add_argument("--classifier", default="svm", choices=["svm", "logreg"])
    parser.add_argument("--mode", default="band", choices=["band", "flat", "pca"])
    parser.add_argument("--align-mode", default="ra", choices=["none", "ra", "tsa", "domain_ra", "uda"])
    parser.add_argument("--domain-level", default="none", choices=["none", "subject", "session"])
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, trial_index, skipped = load_seed_official_de(
        seed_de_root=args.seed_de_root,
        seed_de_var=args.seed_de_var,
        manifest_path=args.seed_manifest,
        freeze_align=True,
        seed_de_window=None,
    )
    if skipped:
        print(f"[seed1][de] skipped_files={skipped}")

    (
        trials_train,
        y_trial_train,
        trial_id_train,
        trials_test,
        y_trial_test,
        trial_id_test,
    ) = _reconstruct_trials(X_train, X_test, trial_index)

    runner = ManifoldRunner(
        mode=args.mode,
        eps=float(args.eps),
        classifier=args.classifier,
        align_mode=args.align_mode,
        domain_level=None if args.domain_level == "none" else args.domain_level,
        input_kind="signal",
        n_bands=5,
    )

    out = runner.fit_predict(
        fold=type(
            "Fold",
            (),
            {
                "trials_train": trials_train,
                "y_trial_train": y_trial_train,
                "trial_id_train": np.asarray(trial_id_train),
                "trials_test": trials_test,
                "y_trial_test": y_trial_test,
                "trial_id_test": np.asarray(trial_id_test),
            },
        )()
    )

    y_test = out["y_trial_test"]
    y_train = out["y_trial_train"]
    proba_test = out["trial_proba_test"]
    proba_train = out["trial_proba_train"]
    preds_test = np.argmax(proba_test, axis=1)
    preds_train = np.argmax(proba_train, axis=1)

    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "trial_acc": float(accuracy_score(y_test, preds_test)),
        "trial_macro_f1": float(f1_score(y_test, preds_test, average="macro")),
        "train_trial_acc": float(accuracy_score(y_train, preds_train)),
        "train_trial_macro_f1": float(f1_score(y_train, preds_train, average="macro")),
        "n_trials_train": int(len(trials_train)),
        "n_trials_test": int(len(trials_test)),
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_test": int(X_test.shape[0]),
    }

    run_config = {
        "seed_de_root": args.seed_de_root,
        "seed_de_var": args.seed_de_var,
        "seed_manifest": args.seed_manifest,
        "mode": args.mode,
        "classifier": args.classifier,
        "align_mode": args.align_mode,
        "domain_level": args.domain_level,
        "eps": float(args.eps),
        "split_rule": "trial_0_8_train_9_14_test (per subject/session)",
        "feature_source": "official DE (standardized via load_seed_official_de)",
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print(
        f"[manifold][seed1][de] trial_acc={metrics['trial_acc']:.4f} "
        f"trial_macro_f1={metrics['trial_macro_f1']:.4f} "
        f"train_acc={metrics['train_trial_acc']:.4f}"
    )
    print(f"[manifold][seed1][de] out_dir={out_dir}")


if __name__ == "__main__":
    main()
