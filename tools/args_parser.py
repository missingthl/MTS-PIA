from __future__ import annotations

import argparse
from typing import Optional, List, Tuple
from dataclasses import dataclass

def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIA Unified Demo: DCNet + Manifold Stream Experiments")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="seedv", help="dataset name: seedv / seed1 / dreamer")
    
    # Streams
    parser.add_argument("--stream", type=str, default="spatial", 
                        choices=["spatial", "manifold", "dual", "manifold_deep", "official_baseline"],
                        help="experiment stream to run")
    
    # Backend
    parser.add_argument("--backend", type=str, default="torch", choices=["torch"], help="backend framework (torch only)")
    parser.add_argument("--torch-device", type=str, default=None, help="torch device override (e.g. cuda:0)")

    # SEED-V Split
    parser.add_argument("--seedv-split", type=str, default="trial", choices=["trial", "session", "subject"],
                       help="seedv split protocol")
    parser.add_argument("--seedv-subject-mode", type=str, default="loso", choices=["loso", "kfold"],
                       help="subject split mode for seedv")
    parser.add_argument("--seedv-subject-k", type=int, default=5, help="k for subject kfold")
    parser.add_argument("--seedv-subject-seed", type=int, default=0, help="seed for subject kfold shuffle")

    # SEED-1 Specific
    parser.add_argument("--seed-de-root", type=str, default="data/SEED/SEED_EEG/ExtractedFeatures_1s",
                       help="SEED official DE root")
    parser.add_argument("--seed-de-var", type=str, default="de_LDS1", help="SEED official DE variable name")
    parser.add_argument("--seed-de-window", type=str, default=None, help="optional window selector for SEED official DE")
    parser.add_argument("--seed-de-mode", type=str, default="official", choices=["official", "author"],
                       help="seed1 DE source")
    parser.add_argument("--seed-manifest", type=str, default="logs/seed_raw_trial_manifest_full.json",
                       help="seed raw trial manifest for alignment")
    parser.add_argument("--seed-freeze-align", dest="seed_freeze_align", action="store_true", default=True,
                       help="freeze SEED1 alignment order")
    parser.add_argument("--seed-no-freeze-align", dest="seed_freeze_align", action="store_false")
    
    # Spatial Stream (DCNet)
    parser.add_argument("--epochs", type=int, default=40, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="batch size")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="evaluation batch size")
    parser.add_argument("--spatial-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--spatial-head", type=str, default="softmax", choices=["softmax", "snn"], help="classifier head")
    parser.add_argument("--spatial-input", type=str, default="flat", choices=["flat", "topo"], help="input format")
    parser.add_argument("--spatial-classifier", type=str, default="conv", choices=["conv", "linear"], help="classifier type")
    parser.add_argument("--spatial-deconv-first", type=int, default=None, help="first deconv filters")
    parser.add_argument("--spatial-bn-eps", type=float, default=1e-5, help="batch norm epsilon")
    parser.add_argument("--spatial-bn-momentum", type=float, default=0.1, help="batch norm momentum")
    parser.add_argument("--spatial-adam-eps", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--spatial-init", type=str, default="default", help="weight initialization")
    parser.add_argument("--spatial-clipnorm", type=float, default=1.0, help="gradient clipping norm")
    parser.add_argument("--spatial-align-baseline", action="store_true", help="override spatial params to match baseline DCNet training defaults")
    parser.add_argument("--freeze-bn", action="store_true", help="freeze batch norm stats")
    
    # SNN Head
    parser.add_argument("--snn-C", type=float, default=4.0, help="SNN C parameter")
    parser.add_argument("--snn-nodes", type=int, default=3, help="SNN nodes")
    parser.add_argument("--snn-activation", type=str, default="sigmoid", help="SNN activation")

    # Manifold Stream
    parser.add_argument("--manifold-mode", type=str, default="band", choices=["band", "flat"], help="manifold mode")
    parser.add_argument("--manifold-classifier", type=str, default="svm", choices=["svm", "mdm", "fgmdm"], help="manifold classifier")
    parser.add_argument("--manifold-source", type=str, default="de", choices=["de", "raw"], help="feature source")
    parser.add_argument("--manifold-pca-dim", type=float, default=0.98, help="PCA dim or variance ratio")
    parser.add_argument("--manifold-C", type=float, default=1.0, help="SVM C parameter")
    parser.add_argument("--manifold-C-grid", type=str, default=None, help="SVM C grid search")
    parser.add_argument("--manifold-eps", type=float, default=1e-4, help="regularization eps")
    parser.add_argument("--manifold-min-eig", type=float, default=1e-6, help="min eigenvalue threshold")
    parser.add_argument("--manifold-ra", type=str, default="euc", choices=["euc", "airm", "logeuclid"], help="riemannian alignment metric")
    parser.add_argument("--manifold-align", type=str, default="euclid", help="alignment strategy")
    parser.add_argument("--manifold-domain", type=str, default="none", help="domain adaptation level")
    parser.add_argument("--manifold-mdm-metric", type=str, default="riemann", help="MDM metric")
    parser.add_argument("--manifold-mdm-mean", type=str, default="riemann", help="MDM mean")
    parser.add_argument("--manifold-no-standardize", action="store_true", help="disable standardization")
    parser.add_argument("--manifold-debug", action="store_true", help="enable manifold debug prints")
    parser.add_argument("--manifold-max-iter", type=int, default=100, help="max iterations for optimization")
    
    # Manifold Raw Specifics
    parser.add_argument("--manifold-raw-repr", type=str, default="cov", choices=["cov", "fb_cov", "signal"], help="raw representation")
    parser.add_argument("--manifold-raw-bands", type=str, default="1-4,4-8,8-10,10-13,13-30,30-45", help="band spec for fb_cov")
    parser.add_argument("--manifold-raw-cache", type=str, default="auto", help="cache policy")
    parser.add_argument("--manifold-raw-channel-policy", type=str, default="strict", help="channel selection policy")
    parser.add_argument("--manifold-n-bands", type=int, default=5, help="number of frequency bands") # Default for DE
    
    # Dual Stream Fusion
    parser.add_argument("--fusion-alpha", type=float, default=0.5, help="fusion alpha weight")
    parser.add_argument("--trial-agg", type=str, default="mean", choices=["mean", "max", "topk"], help="sample to trial aggregation")
    parser.add_argument("--trial-agg-topk", type=float, default=0.3, help="topk ratio")

    # Raw Processing (Seed Raw)
    parser.add_argument("--seed-raw-debug", action="store_true", help="debug seed raw channels")
    parser.add_argument("--seed-raw-root", type=str, default="data/SEED/SEED_EEG/SEED_RAW_EEG", help="seed raw root")
    parser.add_argument("--seed-raw-backend", type=str, default="fif", choices=["cnt", "fif"], help="raw backend")
    parser.add_argument("--seed-raw-fs", type=int, default=None, help="override fs")
    parser.add_argument("--seed-raw-channel-policy", type=str, default="strict", help="channel policy")
    parser.add_argument("--debug-raw", type=int, default=1, help="debug raw trials count")
    parser.add_argument("--raw-mode", type=str, default="v1", choices=["debug", "v1"], help="raw runner mode")
    parser.add_argument("--raw-manifest", type=str, default="logs/seed1_raw_manifest_*.json", help="raw manifest")
    parser.add_argument("--raw-window-sec", type=float, default=4.0, help="window len")
    parser.add_argument("--raw-window-hop-sec", type=float, default=4.0, help="window hop")
    parser.add_argument("--raw-resample-fs", type=float, default=0.0, help="resample fs")
    parser.add_argument("--raw-time-unit", type=str, default=None, help="time unit")
    parser.add_argument("--raw-trial-offset-sec", type=float, default=0.0, help="trial time offset")
    parser.add_argument("--raw-bands", type=str, default=None, help="raw bands override")
    parser.add_argument("--raw-cov", type=str, default="scm", help="cov estimator")
    parser.add_argument("--raw-logmap-eps", type=float, default=1e-4, help="logmap eps")
    parser.add_argument("--raw-seq-save-format", type=str, default="pt", help="sequence save format")
    parser.add_argument("--spd-eps", type=float, default=1e-4, help="SPD epsilon")
    parser.add_argument("--spd-eps-mode", type=str, default="fixed", help="SPD eps mode")
    parser.add_argument("--spd-eps-alpha", type=float, default=0.0, help="SPD eps alpha")
    parser.add_argument("--spd-eps-floor-mult", type=float, default=1.0, help="SPD eps floor mult")
    parser.add_argument("--spd-eps-ceil-mult", type=float, default=1.0, help="SPD eps ceil mult")
    parser.add_argument("--clf", type=str, default="svm", help="raw classifier")
    parser.add_argument("--trial-protocol", type=str, default=None, help="trial protocol override")
    parser.add_argument("--out-prefix", type=str, default=None, help="output prefix")
    parser.add_argument("--raw-chunk-by", type=str, default="subject", help="chunk by")
    parser.add_argument("--raw-max-subjects", type=int, default=0, help="max subjects")
    parser.add_argument("--raw-subject-list", type=str, default=None, help="subject list")
    parser.add_argument("--raw-mem-debug", type=int, default=0, help="mem debug")
    parser.add_argument("--raw-mem-interval", type=int, default=0, help="mem interval")
    parser.add_argument("--raw-save-trial", action="store_true", help="save trial")
    parser.add_argument("--raw-filter-chunk", type=int, default=0, help="filter chunk")
    parser.add_argument("--raw-resample-chunk", type=int, default=0, help="resample chunk")
    parser.add_argument("--raw-workers", type=int, default=1, help="raw workers (subprocess)")
    parser.add_argument("--raw-cnt-subprocess", type=int, default=0, help="cnt subprocess")
    parser.add_argument("--raw-runner", type=str, default="inproc", choices=["inproc", "subprocess"], help="manifold_raw runner")
    parser.add_argument("--raw-stop-on-error", action="store_true", help="stop subprocess run on first subject failure")
    parser.add_argument("--raw-shrinkage", type=str, default=None, help="shrinkage for covariance")

    # Manifold Deep
    parser.add_argument("--manifold-deep-manifest", type=str, default=None, help="deep manifest")
    parser.add_argument("--manifold-deep-epochs", type=int, default=100, help="deep epochs")
    parser.add_argument("--manifold-deep-batch-size", type=int, default=32, help="deep batch size")
    parser.add_argument("--manifold-deep-lr", type=float, default=1e-4, help="deep lr")
    parser.add_argument("--manifold-deep-weight-decay", type=float, default=1e-4, help="deep weight decay")
    parser.add_argument("--manifold-deep-val-ratio", type=float, default=0.1, help="deep val ratio")
    parser.add_argument("--manifold-deep-split-by", type=str, default="trial", help="deep split by")
    parser.add_argument("--manifold-deep-split-seed", type=int, default=42, help="deep split seed")
    parser.add_argument("--manifold-deep-grad-accum-steps", type=int, default=1, help="grad accum steps")
    parser.add_argument("--manifold-deep-embed-dim", type=int, default=128, help="embed dim")
    parser.add_argument("--manifold-deep-num-heads", type=int, default=4, help="num heads")
    parser.add_argument("--manifold-deep-num-layers", type=int, default=2, help="num layers")
    parser.add_argument("--manifold-deep-max-grad-norm", type=float, default=1.0, help="max grad norm")
    parser.add_argument("--manifold-deep-ref-mean-path", type=str, default=None, help="ref mean path")
    parser.add_argument("--manifold-deep-num-workers", type=int, default=4, help="num workers")
    parser.add_argument("--manifold-deep-prefetch-factor", type=int, default=2, help="prefetch factor")
    parser.add_argument("--manifold-deep-no-pin-memory", action="store_true", help="no pin memory")
    parser.add_argument("--manifold-deep-no-persistent-workers", action="store_true", help="no persistent workers")
    parser.add_argument("--manifold-deep-dataset-type", type=str, default="tsm", help="dataset type")
    parser.add_argument("--manifold-deep-model-type", type=str, default="manifold_net", help="model type")
    parser.add_argument("--manifold-deep-spatial-encoder", type=str, default="grid", help="spatial encoder")
    parser.add_argument("--manifold-deep-smoke-test", action="store_true", help="smoke test")
    parser.add_argument("--manifold-deep-graph-num-layers", type=int, default=2, help="graph num layers")
    parser.add_argument("--manifold-deep-graph-num-heads", type=int, default=4, help="graph num heads")
    parser.add_argument("--manifold-deep-graph-ffn-mult", type=float, default=4.0, help="graph ffn mult")
    parser.add_argument("--manifold-deep-graph-dropout", type=float, default=0.1, help="graph dropout")
    parser.add_argument("--manifold-deep-graph-use-max", action="store_true", help="graph use max")
    parser.add_argument("--manifold-deep-tsm-smooth-mode", type=str, default=None, help="tsm smooth mode")
    parser.add_argument("--manifold-deep-tsm-ema-alpha", type=float, default=0.1, help="tsm ema alpha")
    parser.add_argument("--manifold-deep-tsm-kalman-qr", type=float, default=0.01, help="tsm kalman qr")
    parser.add_argument("--manifold-deep-seed", type=int, default=42, help="deep seed")
    parser.add_argument("--manifold-deep-debug-mode", action="store_true", help="deep debug mode")
    parser.add_argument("--manifold-deep-overfit-small-k", type=int, default=None, help="overfit small k")
    parser.add_argument("--manifold-deep-debug-batches", type=int, default=None, help="debug batches")
    parser.add_argument("--manifold-deep-debug-disable-dropout", dest="manifold_deep_debug_disable_dropout", action="store_true", default=True, help="disable dropout")
    parser.add_argument("--manifold-deep-debug-enable-dropout", dest="manifold_deep_debug_disable_dropout", action="store_false", help="keep dropout enabled")
    parser.add_argument("--manifold-deep-debug-weight-decay", type=float, default=None, help="debug weight decay")
    parser.add_argument("--manifold-deep-debug-lr", type=float, default=None, help="debug lr")
    parser.add_argument("--manifold-deep-probe-layer", type=int, default=None, help="probe layer")
    parser.add_argument("--manifold-deep-probe-epochs", type=int, default=None, help="probe epochs")

    # Official Baseline
    parser.add_argument("--official-root", type=str, default=None, help="official root")
    parser.add_argument("--official-feature-base", type=str, default=None, help="feature base")
    parser.add_argument("--official-agg-mode", type=str, default="mean", help="agg mode")
    parser.add_argument("--official-split-by", type=str, default="trial", help="split by")
    parser.add_argument("--official-val-ratio", type=float, default=0.1, help="val ratio")
    parser.add_argument("--official-split-seed", type=int, default=42, help="split seed")
    parser.add_argument("--official-model", type=str, default="svm", help="model type")
    parser.add_argument("--official-manifest", type=str, default=None, help="manifest path")
    parser.add_argument("--official-run-dir", type=str, default=None, help="run dir")

    # Common
    parser.add_argument("--no-check-input", action="store_true", help="disable input checks")
    parser.add_argument("--no-input-log", action="store_true", help="disable input stats logging")
    parser.add_argument("--no-terminate-on-nan", action="store_true", help="disable terminate on nan")
    parser.add_argument("--folds", type=int, default=3, help="number of folds")
    parser.add_argument("--seed-de-debug", action="store_true", help="debug seed de")
    parser.add_argument("--de-lds", action="store_true", help="enable LDS smoothing on DE")
    parser.add_argument("--de-lds-level", type=str, default="trial", help="lds level")
    parser.add_argument("--de-lds-method", type=str, default="standard", help="lds method")
    parser.add_argument("--de-lds-q", type=float, default=1e-4, help="lds q")
    parser.add_argument("--de-lds-r", type=float, default=1.0, help="lds r")
    parser.add_argument("--de-lds-em-iters", type=int, default=3, help="lds em iters")
    parser.add_argument("--de-lds-em-tol", type=float, default=1e-4, help="lds em tol")

    return parser
