"""
safeshipper — Production-grade harm prevalence measurement toolkit.

Modules
-------
classifier  : LLM-based harm detection system (Anthropic / OpenAI / Groq)
sampling    : Stratified sampling design for rare event estimation
prevalence  : Capture-recapture and classifier-adjusted prevalence estimators
calibration : Classifier calibration and error propagation
metrics     : Threshold-level classification metrics
simulation  : Synthetic corpus generator for methodology validation
actor       : Actor-level behavioral features and coordination network detection
"""

from .classifier import HarmClassifier, ClassificationResult, HARM_CATEGORIES
from .sampling import StratifiedHarmSampler, SampleDesign
from .prevalence import PrevalenceEstimator, PrevalenceEstimate
from .calibration import ClassifierCalibrator, CalibrationResult
from .metrics import compute_metrics_at_threshold, sweep_thresholds, ThresholdMetrics
from .simulation import SimulationConfig, generate_corpus, simulate_detection_systems
from .actor import (
    ActorFeatureExtractor,
    ActorFeatures,
    ActorRiskScore,
    NetworkCluster,
    detect_coordination_networks,
    simulate_actor_corpus,
)

__all__ = [
    "HarmClassifier",
    "ClassificationResult",
    "HARM_CATEGORIES",
    "StratifiedHarmSampler",
    "SampleDesign",
    "PrevalenceEstimator",
    "PrevalenceEstimate",
    "ClassifierCalibrator",
    "CalibrationResult",
    "compute_metrics_at_threshold",
    "sweep_thresholds",
    "ThresholdMetrics",
    "SimulationConfig",
    "generate_corpus",
    "simulate_detection_systems",
    "ActorFeatureExtractor",
    "ActorFeatures",
    "ActorRiskScore",
    "NetworkCluster",
    "detect_coordination_networks",
    "simulate_actor_corpus",
]
