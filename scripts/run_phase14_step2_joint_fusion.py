import os
import sys
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runners.joint_fusion_runner import JointFusionRunner, JointFusionConfig


def main():
    cfg = JointFusionConfig(
        seeds=[0, 4],
        dataset="seed1",
        out_root="promoted_results/phase14/step2/seed1",
        backbone_lr=1e-5,
        head_lr=1e-3,
        weight_decay=0.0,
        epochs=30,
        batch_size=8,
    )

    runner = JointFusionRunner(cfg)
    try:
        runner.run_all()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
