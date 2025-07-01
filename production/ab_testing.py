"""A/B testing framework for TradingNanoLLM model comparison and optimization."""

import random
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from enum import Enum
import hashlib
import statistics

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentVariant:
    """Represents a single variant in an A/B test."""
    name: str
    description: str
    model_config: Dict[str, Any]
    traffic_allocation: float  # 0.0 to 1.0
    is_control: bool = False


@dataclass
class ExperimentMetric:
    """Represents a metric to track in experiments."""
    name: str
    description: str
    higher_is_better: bool = True
    significance_threshold: float = 0.05


@dataclass
class ExperimentResult:
    """Results for a single variant."""
    variant_name: str
    sample_size: int
    metric_values: Dict[str, List[float]]
    conversion_rate: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0


class TrafficSplitter:
    """Handles traffic splitting for A/B tests."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize traffic splitter.
        
        Args:
            seed: Random seed for reproducible splits
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def get_variant(self, user_id: str, variants: List[ExperimentVariant]) -> ExperimentVariant:
        """Determine which variant a user should see.
        
        Args:
            user_id: Unique identifier for the user/request
            variants: List of available variants
            
        Returns:
            Selected variant
        """
        # Use consistent hashing for deterministic assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        cumulative_allocation = 0.0
        for variant in variants:
            cumulative_allocation += variant.traffic_allocation
            if normalized_hash <= cumulative_allocation:
                return variant
        
        # Fallback to control variant
        control_variants = [v for v in variants if v.is_control]
        return control_variants[0] if control_variants else variants[0]


class StatisticalAnalyzer:
    """Performs statistical analysis on A/B test results."""
    
    @staticmethod
    def calculate_significance(control_values: List[float], 
                              treatment_values: List[float]) -> Tuple[bool, float]:
        """Calculate statistical significance between two groups.
        
        Args:
            control_values: Metric values for control group
            treatment_values: Metric values for treatment group
            
        Returns:
            Tuple of (is_significant, p_value)
        """
        try:
            from scipy import stats
            
            if len(control_values) < 30 or len(treatment_values) < 30:
                return False, 1.0  # Insufficient sample size
            
            # Perform Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False)
            
            return p_value < 0.05, p_value
            
        except ImportError:
            logger.warning("scipy not available, using simplified significance test")
            return StatisticalAnalyzer._simple_significance_test(control_values, treatment_values)
    
    @staticmethod
    def _simple_significance_test(control_values: List[float], 
                                 treatment_values: List[float]) -> Tuple[bool, float]:
        """Simple significance test without scipy."""
        if len(control_values) < 30 or len(treatment_values) < 30:
            return False, 1.0
        
        control_mean = statistics.mean(control_values)
        treatment_mean = statistics.mean(treatment_values)
        
        control_std = statistics.stdev(control_values)
        treatment_std = statistics.stdev(treatment_values)
        
        # Simple effect size calculation
        pooled_std = ((control_std ** 2 + treatment_std ** 2) / 2) ** 0.5
        effect_size = abs(treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Rough significance based on effect size and sample sizes
        min_effect_size = 0.5  # Cohen's medium effect size
        is_significant = effect_size >= min_effect_size
        
        # Rough p-value approximation
        p_value = max(0.01, 0.5 - effect_size * 0.4) if is_significant else 0.5
        
        return is_significant, p_value
    
    @staticmethod
    def calculate_lift(control_mean: float, treatment_mean: float) -> float:
        """Calculate percentage lift of treatment over control.
        
        Args:
            control_mean: Mean value for control group
            treatment_mean: Mean value for treatment group
            
        Returns:
            Percentage lift (positive means treatment is better)
        """
        if control_mean == 0:
            return 0.0
        return ((treatment_mean - control_mean) / control_mean) * 100


class ABTestExperiment:
    """Represents a single A/B test experiment."""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 variants: List[ExperimentVariant],
                 metrics: List[ExperimentMetric],
                 min_sample_size: int = 100,
                 max_duration_days: int = 30):
        """Initialize A/B test experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variants to test
            metrics: List of metrics to track
            min_sample_size: Minimum sample size per variant
            max_duration_days: Maximum experiment duration
        """
        self.name = name
        self.description = description
        self.variants = variants
        self.metrics = metrics
        self.min_sample_size = min_sample_size
        self.max_duration_days = max_duration_days
        
        self.status = ExperimentStatus.DRAFT
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Results storage
        self.results: Dict[str, ExperimentResult] = {}
        self.raw_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Traffic splitter
        self.traffic_splitter = TrafficSplitter()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Validate experiment setup
        self._validate_experiment()
    
    def _validate_experiment(self):
        """Validate experiment configuration."""
        # Check traffic allocation sums to 1.0
        total_allocation = sum(v.traffic_allocation for v in self.variants)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Check for exactly one control variant
        control_variants = [v for v in self.variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Experiment must have exactly one control variant")
        
        # Check variant names are unique
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("Variant names must be unique")
    
    def start(self):
        """Start the experiment."""
        if self.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in {self.status.value} status")
        
        self.status = ExperimentStatus.RUNNING
        self.start_time = time.time()
        
        # Initialize results for each variant
        for variant in self.variants:
            self.results[variant.name] = ExperimentResult(
                variant_name=variant.name,
                sample_size=0,
                metric_values={metric.name: [] for metric in self.metrics}
            )
        
        logger.info(f"Started A/B test experiment: {self.name}")
    
    def stop(self):
        """Stop the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot stop experiment in {self.status.value} status")
        
        self.status = ExperimentStatus.COMPLETED
        self.end_time = time.time()
        
        logger.info(f"Stopped A/B test experiment: {self.name}")
    
    def get_variant_for_user(self, user_id: str) -> ExperimentVariant:
        """Get the variant assignment for a user."""
        if self.status != ExperimentStatus.RUNNING:
            # Return control variant if not running
            control_variants = [v for v in self.variants if v.is_control]
            return control_variants[0]
        
        return self.traffic_splitter.get_variant(user_id, self.variants)
    
    def record_observation(self, user_id: str, variant_name: str, 
                          metrics: Dict[str, float], metadata: Dict[str, Any] = None):
        """Record an observation for the experiment.
        
        Args:
            user_id: User identifier
            variant_name: Which variant was used
            metrics: Metric values observed
            metadata: Additional metadata
        """
        if self.status != ExperimentStatus.RUNNING:
            return
        
        if metadata is None:
            metadata = {}
        
        observation = {
            "user_id": user_id,
            "timestamp": time.time(),
            "metrics": metrics,
            "metadata": metadata
        }
        
        with self.lock:
            # Store raw observation
            self.raw_data[variant_name].append(observation)
            
            # Update aggregated results
            if variant_name in self.results:
                result = self.results[variant_name]
                result.sample_size += 1
                
                # Update metric values
                for metric_name, value in metrics.items():
                    if metric_name in result.metric_values:
                        result.metric_values[metric_name].append(value)
                
                # Update derived metrics
                if "response_time" in metrics:
                    response_times = result.metric_values.get("response_time", [])
                    if response_times:
                        result.average_response_time = statistics.mean(response_times)
                
                if "error" in metrics:
                    errors = result.metric_values.get("error", [])
                    result.error_rate = sum(errors) / len(errors) if errors else 0.0
    
    def get_results(self) -> Dict[str, Any]:
        """Get current experiment results."""
        with self.lock:
            control_variant = next(v for v in self.variants if v.is_control)
            control_result = self.results.get(control_variant.name)
            
            if not control_result:
                return {"status": "no_data", "message": "No data collected yet"}
            
            analysis = {
                "experiment_name": self.name,
                "status": self.status.value,
                "start_time": self.start_time,
                "duration_hours": (time.time() - self.start_time) / 3600 if self.start_time else 0,
                "variants": {},
                "significance_tests": {},
                "recommendations": []
            }
            
            # Analyze each variant
            for variant in self.variants:
                result = self.results.get(variant.name)
                if not result:
                    continue
                
                variant_analysis = {
                    "name": variant.name,
                    "is_control": variant.is_control,
                    "sample_size": result.sample_size,
                    "traffic_allocation": variant.traffic_allocation,
                    "metrics": {}
                }
                
                # Calculate metric statistics
                for metric in self.metrics:
                    values = result.metric_values.get(metric.name, [])
                    if values:
                        variant_analysis["metrics"][metric.name] = {
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "std": statistics.stdev(values) if len(values) > 1 else 0,
                            "min": min(values),
                            "max": max(values),
                            "count": len(values)
                        }
                
                analysis["variants"][variant.name] = variant_analysis
                
                # Statistical significance vs control (if not control)
                if not variant.is_control and control_result:
                    for metric in self.metrics:
                        control_values = control_result.metric_values.get(metric.name, [])
                        treatment_values = result.metric_values.get(metric.name, [])
                        
                        if control_values and treatment_values:
                            is_significant, p_value = StatisticalAnalyzer.calculate_significance(
                                control_values, treatment_values
                            )
                            
                            control_mean = statistics.mean(control_values)
                            treatment_mean = statistics.mean(treatment_values)
                            lift = StatisticalAnalyzer.calculate_lift(control_mean, treatment_mean)
                            
                            significance_key = f"{variant.name}_vs_{control_variant.name}_{metric.name}"
                            analysis["significance_tests"][significance_key] = {
                                "variant": variant.name,
                                "metric": metric.name,
                                "is_significant": is_significant,
                                "p_value": p_value,
                                "lift_percent": lift,
                                "control_mean": control_mean,
                                "treatment_mean": treatment_mean,
                                "control_sample_size": len(control_values),
                                "treatment_sample_size": len(treatment_values)
                            }
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Check sample sizes
        min_samples_met = all(
            variant.get("sample_size", 0) >= self.min_sample_size 
            for variant in analysis["variants"].values()
        )
        
        if not min_samples_met:
            recommendations.append(
                f"Continue experiment until all variants reach minimum sample size of {self.min_sample_size}"
            )
        
        # Check for clear winners
        significant_improvements = []
        for test_name, test_result in analysis["significance_tests"].items():
            if test_result["is_significant"] and test_result["lift_percent"] > 0:
                significant_improvements.append(test_result)
        
        if significant_improvements:
            best_improvement = max(significant_improvements, key=lambda x: x["lift_percent"])
            recommendations.append(
                f"Consider deploying {best_improvement['variant']} - shows {best_improvement['lift_percent']:.1f}% "
                f"improvement in {best_improvement['metric']} (p={best_improvement['p_value']:.3f})"
            )
        
        # Check for performance issues
        for variant_name, variant_data in analysis["variants"].items():
            error_rate = variant_data.get("metrics", {}).get("error_rate", {}).get("mean", 0)
            if error_rate > 0.05:  # 5% error rate threshold
                recommendations.append(
                    f"Investigate {variant_name} - high error rate ({error_rate:.1%})"
                )
        
        # Duration recommendations
        if analysis["duration_hours"] > self.max_duration_days * 24:
            recommendations.append("Experiment has exceeded maximum duration - consider stopping")
        
        if not recommendations:
            recommendations.append("Continue monitoring - no clear winner yet")
        
        return recommendations
    
    def export_data(self, filepath: str):
        """Export experiment data to file."""
        export_data = {
            "experiment": {
                "name": self.name,
                "description": self.description,
                "variants": [asdict(v) for v in self.variants],
                "metrics": [asdict(m) for m in self.metrics],
                "status": self.status.value,
                "start_time": self.start_time,
                "end_time": self.end_time
            },
            "results": self.get_results(),
            "raw_data": dict(self.raw_data)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


class ABTestManager:
    """Manages multiple A/B test experiments."""
    
    def __init__(self):
        """Initialize A/B test manager."""
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.active_experiment: Optional[str] = None
    
    def create_experiment(self, 
                         name: str,
                         description: str,
                         variants: List[ExperimentVariant],
                         metrics: List[ExperimentMetric],
                         **kwargs) -> ABTestExperiment:
        """Create a new experiment."""
        if name in self.experiments:
            raise ValueError(f"Experiment {name} already exists")
        
        experiment = ABTestExperiment(name, description, variants, metrics, **kwargs)
        self.experiments[name] = experiment
        
        logger.info(f"Created experiment: {name}")
        return experiment
    
    def start_experiment(self, name: str):
        """Start an experiment."""
        if name not in self.experiments:
            raise ValueError(f"Experiment {name} not found")
        
        # Stop any currently active experiment
        if self.active_experiment:
            self.experiments[self.active_experiment].stop()
        
        experiment = self.experiments[name]
        experiment.start()
        self.active_experiment = name
        
        logger.info(f"Started experiment: {name}")
    
    def stop_experiment(self, name: str):
        """Stop an experiment."""
        if name not in self.experiments:
            raise ValueError(f"Experiment {name} not found")
        
        experiment = self.experiments[name]
        experiment.stop()
        
        if self.active_experiment == name:
            self.active_experiment = None
        
        logger.info(f"Stopped experiment: {name}")
    
    def get_active_experiment(self) -> Optional[ABTestExperiment]:
        """Get the currently active experiment."""
        if self.active_experiment:
            return self.experiments.get(self.active_experiment)
        return None
    
    def record_observation(self, user_id: str, metrics: Dict[str, float], 
                          metadata: Dict[str, Any] = None):
        """Record an observation for the active experiment."""
        active_exp = self.get_active_experiment()
        if not active_exp:
            return None
        
        variant = active_exp.get_variant_for_user(user_id)
        active_exp.record_observation(user_id, variant.name, metrics, metadata)
        
        return variant
    
    def get_experiment_results(self, name: str) -> Dict[str, Any]:
        """Get results for a specific experiment."""
        if name not in self.experiments:
            raise ValueError(f"Experiment {name} not found")
        
        return self.experiments[name].get_results()
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with basic info."""
        experiments_info = []
        
        for name, experiment in self.experiments.items():
            info = {
                "name": name,
                "description": experiment.description,
                "status": experiment.status.value,
                "variants": len(experiment.variants),
                "start_time": experiment.start_time,
                "is_active": name == self.active_experiment
            }
            
            # Add sample sizes if running
            if experiment.status == ExperimentStatus.RUNNING:
                total_samples = sum(r.sample_size for r in experiment.results.values())
                info["total_samples"] = total_samples
            
            experiments_info.append(info)
        
        return experiments_info
