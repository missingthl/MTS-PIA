"""Unified Route B policy-driven closed-loop framework (Standalone Version)."""

from .augmentation_admission import (
    HybridAdmissionConfig,
    HybridAdmissionResult,
    apply_hybrid_admission,
)
from .bridge import BridgeConfig, apply_bridge
from .evaluator import EvaluatorPosterior, MiniRocketEvalConfig, evaluate_bridge
# from .dual_stream_classifier import DualStreamModelConfig
# from .dual_stream_dataset import DualStreamSplit, DualStreamState, build_dual_stream_state
# from .dual_stream_evaluator import DualStreamEvalConfig, DualStreamEvalResult, evaluate_dual_stream
# from .pia_core import PIACore, PIACoreArtifacts, PIACoreConfig, PIAOperatorResult
from .trajectory_classifier import (
    DynamicGRUClassifier,
    DynamicMeanPoolClassifier,
    StaticLinearClassifier,
    TrajectoryModelConfig,
)
# from .trajectory_pia_evaluator import TrajectoryPIAEvalConfig, TrajectoryPIAEvalResult, evaluate_trajectory_pia_t2a
from .trajectory_pia_operator import (
    TrajectoryPIAOperator,
    TrajectoryPIAOperatorArtifacts,
    TrajectoryPIAOperatorConfig,
)
from .trajectory_feedback_pool import TrajectoryFeedbackPoolConfig, TrajectoryFeedbackPoolResult, build_trajectory_feedback_pool
from .trajectory_feedback_pool_windows import (
    TrajectoryWindowFeedbackPoolConfig,
    TrajectoryWindowFeedbackPoolResult,
    TrajectoryWindowReferenceStats,
    build_window_feedback_pool,
    build_window_feedback_reference_stats,
)
from .trajectory_feedback_rebasis import TrajectoryFeedbackRebasisResult, fit_trajectory_feedback_rebasis
from .trajectory_feedback_rebasis_t4 import (
    TrajectoryClassConditionedBasisFamily,
    TrajectoryClassConditionedRebasisResult,
    fit_trajectory_class_conditioned_rebasis,
)
from .trajectory_feedback_rebasis_t7 import (
    TrajectoryClassConditionedOVRBasisFamily,
    TrajectoryClassConditionedOVRRebasisResult,
    fit_trajectory_class_conditioned_rebasis_ovr,
)
from .scp_prototype_memory import SCPPrototypeMemoryConfig, SCPPrototypeMemoryResult, build_scp_prototype_memory
from .scp_local_shaping import SCPLocalShapingConfig, SCPLocalShapingResult, apply_scp_local_shaping
from .trajectory_dual_role_policy import (
    TrajectoryDualRolePolicyResult,
    build_dual_role_augmented_trajectories,
)
from .trajectory_unified_window_policy import (
    TrajectoryUnifiedWindowPolicyResult,
    build_unified_window_augmented_trajectories,
)
from .trajectory_pia_operator_t2b import TrajectoryPIAT2B0Artifacts, TrajectoryPIAT2B0Config, TrajectoryPIAT2B0Operator
from .trajectory_dataset import TrajectoryTrialDataset, build_trajectory_datasets, collate_trajectory_batch
from .trajectory_evaluator import TrajectoryEvalConfig, TrajectoryEvalResult, evaluate_trajectory_classifier
from .trajectory_representation import (
    TrajectoryRepresentationConfig,
    TrajectoryRepresentationState,
    TrajectorySplit,
    build_trajectory_representation,
)
# from .regression import (
#     RegressionEvalResult,
#     RegressionRepresentationConfig,
#     RegressionRepresentationState,
#     RegressorConfig,
#     build_regression_representation,
#     build_regressor,
#     evaluate_regression,
# )
from .policy import (
    UnifiedPolicyConfig,
    apply_target_feedback,
    init_policy,
    policy_step,
    update_policy,
)
from .representation import RepresentationConfig, build_representation
from .types import (
    BridgeResult,
    PolicyAction,
    PolicyState,
    PolicyUpdateSummary,
    RepresentationState,
    TargetRoundState,
)

__all__ = [
    "BridgeConfig",
    "BridgeResult",
    # "DualStreamEvalConfig",
    # "DualStreamEvalResult",
    # "DualStreamModelConfig",
    # "DualStreamSplit",
    # "DualStreamState",
    "DynamicGRUClassifier",
    "DynamicMeanPoolClassifier",
    "EvaluatorPosterior",
    "HybridAdmissionConfig",
    "HybridAdmissionResult",
    "MiniRocketEvalConfig",
    "PolicyAction",
    "PolicyState",
    "PolicyUpdateSummary",
    # "PIACore",
    # "PIACoreArtifacts",
    # "PIACoreConfig",
    # "PIAOperatorResult",
    # "RegressionEvalResult",
    # "RegressionRepresentationConfig",
    # "RegressionRepresentationState",
    # "RegressorConfig",
    "RepresentationConfig",
    "RepresentationState",
    "SCPPrototypeMemoryConfig",
    "SCPPrototypeMemoryResult",
    "SCPLocalShapingConfig",
    "SCPLocalShapingResult",
    "StaticLinearClassifier",
    "TargetRoundState",
    "TrajectoryEvalConfig",
    "TrajectoryEvalResult",
    "TrajectoryModelConfig",
    # "TrajectoryPIAEvalConfig",
    # "TrajectoryPIAEvalResult",
    "TrajectoryPIAOperator",
    "TrajectoryPIAOperatorArtifacts",
    "TrajectoryPIAOperatorConfig",
    "TrajectoryFeedbackPoolConfig",
    "TrajectoryFeedbackPoolResult",
    "TrajectoryWindowFeedbackPoolConfig",
    "TrajectoryWindowFeedbackPoolResult",
    "TrajectoryWindowReferenceStats",
    "TrajectoryClassConditionedBasisFamily",
    "TrajectoryClassConditionedRebasisResult",
    "TrajectoryClassConditionedOVRBasisFamily",
    "TrajectoryClassConditionedOVRRebasisResult",
    "TrajectoryDualRolePolicyResult",
    "TrajectoryFeedbackRebasisResult",
    "TrajectoryUnifiedWindowPolicyResult",
    "TrajectoryPIAT2B0Artifacts",
    "TrajectoryPIAT2B0Config",
    "TrajectoryPIAT2B0Operator",
    "TrajectoryRepresentationConfig",
    "TrajectoryRepresentationState",
    "TrajectorySplit",
    "TrajectoryTrialDataset",
    "UnifiedPolicyConfig",
    "apply_bridge",
    # "build_regression_representation",
    # "build_regressor",
    # "build_dual_stream_state",
    "build_trajectory_datasets",
    "build_trajectory_representation",
    "build_trajectory_feedback_pool",
    "build_window_feedback_pool",
    "build_window_feedback_reference_stats",
    "build_scp_prototype_memory",
    "apply_scp_local_shaping",
    "fit_trajectory_class_conditioned_rebasis",
    "fit_trajectory_class_conditioned_rebasis_ovr",
    "build_dual_role_augmented_trajectories",
    "build_unified_window_augmented_trajectories",
    "apply_hybrid_admission",
    "apply_target_feedback",
    "build_representation",
    "collate_trajectory_batch",
    "evaluate_bridge",
    # "evaluate_dual_stream",
    # "evaluate_trajectory_pia_t2a",
    "evaluate_trajectory_classifier",
    # "evaluate_regression",
    "fit_trajectory_feedback_rebasis",
    "init_policy",
    "policy_step",
    "update_policy",
]
