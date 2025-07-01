"""Production deployment example for TradingNanoLLM with full monitoring integration."""

import time
import uuid
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

# Import production components
from .monitoring import (
    MetricsCollector, HealthChecker, AlertManager, MonitoringDashboard,
    setup_default_alerts, console_alert_handler
)
from .ab_testing import (
    ABTestManager, ExperimentVariant, ExperimentMetric
)

# Import core TradingNanoLLM components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_nanovllm import TradingLLM, SamplingParams
from trading_nanovllm.trading_utils import SentimentAnalyzer, TradeSignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionTradingService:
    """Production-ready TradingNanoLLM service with full monitoring and A/B testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize production trading service.
        
        Args:
            config: Service configuration dictionary
        """
        self.config = config
        
        # Core LLM components
        self.llm: Optional[TradingLLM] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.signal_generator: Optional[TradeSignalGenerator] = None
        
        # Production monitoring
        self.metrics_collector = MetricsCollector(retention_hours=24)
        self.health_checker: Optional[HealthChecker] = None
        self.alert_manager = AlertManager()
        self.dashboard: Optional[MonitoringDashboard] = None
        
        # A/B testing
        self.ab_test_manager = ABTestManager()
        
        # Service state
        self.is_running = False
        
        # Set up alerts and monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Set up monitoring and alerting."""
        # Add default alert rules
        setup_default_alerts(self.alert_manager)
        
        # Add console alert handler
        self.alert_manager.add_alert_handler(console_alert_handler)
        
        # Add file alert handler if specified
        if self.config.get("alert_log_file"):
            from .monitoring import file_alert_handler
            self.alert_manager.add_alert_handler(
                file_alert_handler(self.config["alert_log_file"])
            )
    
    def start(self):
        """Start the production service."""
        if self.is_running:
            logger.warning("Service is already running")
            return
        
        logger.info("Starting TradingNanoLLM Production Service")
        
        try:
            # Load model
            logger.info("Loading model...")
            self.llm = TradingLLM(
                model_path=self.config.get("model_path", "Qwen/Qwen2-0.5B-Instruct"),
                enforce_eager=self.config.get("enforce_eager", True),
                enable_prefix_caching=self.config.get("enable_caching", True),
                max_batch_size=self.config.get("max_batch_size", 16)
            )
            
            # Initialize trading utilities
            self.sentiment_analyzer = SentimentAnalyzer(self.llm)
            self.signal_generator = TradeSignalGenerator(self.llm)
            
            # Start health checking
            self.health_checker = HealthChecker(self.llm, check_interval=60)
            self.health_checker.start()
            
            # Initialize dashboard
            self.dashboard = MonitoringDashboard(
                self.metrics_collector,
                self.health_checker,
                self.alert_manager
            )
            
            # Set up default A/B test if configured
            if self.config.get("enable_ab_testing", False):
                self._setup_default_ab_test()
            
            self.is_running = True
            logger.info("✅ TradingNanoLLM Production Service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            raise
    
    def stop(self):
        """Stop the production service."""
        if not self.is_running:
            return
        
        logger.info("Stopping TradingNanoLLM Production Service")
        
        # Stop health checker
        if self.health_checker:
            self.health_checker.stop()
        
        # Stop any active A/B tests
        active_experiment = self.ab_test_manager.get_active_experiment()
        if active_experiment:
            self.ab_test_manager.stop_experiment(active_experiment.name)
        
        self.is_running = False
        logger.info("✅ Service stopped")
    
    def _setup_default_ab_test(self):
        """Set up a default A/B test for model comparison."""
        # Create variants for testing different configurations
        control_variant = ExperimentVariant(
            name="control",
            description="Default model configuration",
            model_config={
                "temperature": 0.6,
                "max_tokens": 256,
                "enable_caching": True
            },
            traffic_allocation=0.5,
            is_control=True
        )
        
        treatment_variant = ExperimentVariant(
            name="optimized",
            description="Optimized configuration",
            model_config={
                "temperature": 0.4,  # Lower temperature for more consistent results
                "max_tokens": 256,
                "enable_caching": True
            },
            traffic_allocation=0.5,
            is_control=False
        )
        
        # Define metrics to track
        metrics = [
            ExperimentMetric(
                name="response_time",
                description="Response time in seconds",
                higher_is_better=False
            ),
            ExperimentMetric(
                name="confidence_score",
                description="Model confidence score",
                higher_is_better=True
            ),
            ExperimentMetric(
                name="user_satisfaction",
                description="User satisfaction rating",
                higher_is_better=True
            )
        ]
        
        # Create and start experiment
        experiment = self.ab_test_manager.create_experiment(
            name="default_optimization_test",
            description="Test optimized vs default model configuration",
            variants=[control_variant, treatment_variant],
            metrics=metrics,
            min_sample_size=100,
            max_duration_days=14
        )
        
        self.ab_test_manager.start_experiment("default_optimization_test")
        logger.info("Started default A/B test experiment")
    
    @contextmanager
    def _record_request_metrics(self, endpoint: str, user_id: str = None):
        """Context manager to record request metrics."""
        start_time = time.time()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self.metrics_collector.record_error(str(type(e).__name__), endpoint)
            raise
        finally:
            # Record latency for successful requests
            if not error_occurred:
                latency = time.time() - start_time
                self.metrics_collector.record_latency(latency, endpoint)
                
                # Record A/B test observation if applicable
                if user_id and self.ab_test_manager.get_active_experiment():
                    metrics = {
                        "response_time": latency,
                        "error": 0.0  # No error occurred
                    }
                    self.ab_test_manager.record_observation(user_id, metrics)
    
    def analyze_sentiment(self, text: str, user_id: str = None, 
                         config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze sentiment with full production monitoring.
        
        Args:
            text: Text to analyze
            user_id: User identifier for A/B testing
            config_override: Override default configuration
            
        Returns:
            Sentiment analysis result with metadata
        """
        if not self.is_running:
            raise RuntimeError("Service is not running")
        
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        with self._record_request_metrics("sentiment_analysis", user_id):
            # Get A/B test variant if applicable
            variant = None
            active_experiment = self.ab_test_manager.get_active_experiment()
            if active_experiment:
                variant = active_experiment.get_variant_for_user(user_id)
                
                # Use variant configuration
                config_override = config_override or {}
                config_override.update(variant.model_config)
            
            # Apply configuration
            temperature = config_override.get("temperature", 0.6) if config_override else 0.6
            max_tokens = config_override.get("max_tokens", 128) if config_override else 128
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Perform analysis
            start_time = time.time()
            
            if config_override and config_override.get("use_confidence", False):
                result = self.sentiment_analyzer.analyze_with_confidence(text)
                sentiment = result["sentiment"]
                confidence = result["confidence"]
            else:
                from trading_nanovllm.trading_utils import analyze_sentiment
                sentiment = analyze_sentiment(text, self.llm, sampling_params)
                confidence = 0.8  # Default confidence
            
            response_time = time.time() - start_time
            
            # Record A/B test metrics
            if variant and active_experiment:
                ab_metrics = {
                    "response_time": response_time,
                    "confidence_score": confidence,
                    "error": 0.0
                }
                active_experiment.record_observation(user_id, variant.name, ab_metrics)
            
            # Record custom metrics
            self.metrics_collector.record_metric("sentiment_requests", 1.0)
            self.metrics_collector.record_metric("confidence_score", confidence)
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "response_time": response_time,
                "user_id": user_id,
                "variant": variant.name if variant else None,
                "timestamp": time.time()
            }
    
    def generate_trade_signal(self, market_data: Dict[str, Any], 
                            user_id: str = None,
                            config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate trade signal with full production monitoring.
        
        Args:
            market_data: Market data for analysis
            user_id: User identifier for A/B testing
            config_override: Override default configuration
            
        Returns:
            Trade signal result with metadata
        """
        if not self.is_running:
            raise RuntimeError("Service is not running")
        
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        with self._record_request_metrics("trade_signal", user_id):
            # Get A/B test variant if applicable
            variant = None
            active_experiment = self.ab_test_manager.get_active_experiment()
            if active_experiment:
                variant = active_experiment.get_variant_for_user(user_id)
                config_override = config_override or {}
                config_override.update(variant.model_config)
            
            # Perform analysis
            start_time = time.time()
            
            if config_override and config_override.get("detailed_analysis", True):
                result = self.signal_generator.generate_with_analysis(market_data)
                signal = result["signal"]
                confidence = result["confidence"] / 10.0  # Normalize to 0-1
                reasoning = result["reasoning"]
            else:
                from trading_nanovllm.trading_utils import generate_trade_signal
                sampling_params = SamplingParams(
                    temperature=config_override.get("temperature", 0.4) if config_override else 0.4,
                    max_tokens=config_override.get("max_tokens", 128) if config_override else 128
                )
                
                if isinstance(market_data, dict):
                    data_str = self.signal_generator._format_market_data(market_data)
                else:
                    data_str = str(market_data)
                
                signal = generate_trade_signal(data_str, self.llm, sampling_params)
                confidence = 0.7  # Default confidence
                reasoning = "Basic signal generation"
            
            response_time = time.time() - start_time
            
            # Record A/B test metrics
            if variant and active_experiment:
                ab_metrics = {
                    "response_time": response_time,
                    "confidence_score": confidence,
                    "error": 0.0
                }
                active_experiment.record_observation(user_id, variant.name, ab_metrics)
            
            # Record custom metrics
            self.metrics_collector.record_metric("signal_requests", 1.0)
            self.metrics_collector.record_metric("signal_confidence", confidence)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "response_time": response_time,
                "user_id": user_id,
                "variant": variant.name if variant else None,
                "timestamp": time.time()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        if not self.health_checker:
            return {"status": "unknown", "message": "Health checker not initialized"}
        
        return self.health_checker.get_health_status()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.dashboard:
            return {"error": "Dashboard not initialized"}
        
        return self.dashboard.get_dashboard_data()
    
    def get_ab_test_results(self) -> Dict[str, Any]:
        """Get A/B test results."""
        experiments = self.ab_test_manager.list_experiments()
        results = {}
        
        for exp_info in experiments:
            if exp_info["is_active"]:
                results[exp_info["name"]] = self.ab_test_manager.get_experiment_results(exp_info["name"])
        
        return results
    
    def export_monitoring_data(self, filepath: str):
        """Export comprehensive monitoring data."""
        export_data = {
            "service_info": {
                "config": self.config,
                "is_running": self.is_running,
                "export_timestamp": time.time()
            },
            "metrics": self.get_metrics_summary(),
            "health": self.get_health_status(),
            "ab_tests": self.get_ab_test_results(),
            "model_stats": self.llm.get_stats() if self.llm else {}
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported monitoring data to {filepath}")


def create_production_service(config: Dict[str, Any] = None) -> ProductionTradingService:
    """Create a production trading service with default configuration.
    
    Args:
        config: Service configuration (optional)
        
    Returns:
        Configured production service
    """
    default_config = {
        "model_path": "Qwen/Qwen2-0.5B-Instruct",
        "enforce_eager": True,
        "enable_caching": True,
        "max_batch_size": 16,
        "enable_ab_testing": True,
        "alert_log_file": "alerts.log"
    }
    
    if config:
        default_config.update(config)
    
    return ProductionTradingService(default_config)


# Example usage and testing
def main():
    """Example of running the production service."""
    print("🚀 TradingNanoLLM Production Service Example")
    print("=" * 60)
    
    # Create and start service
    config = {
        "model_path": "Qwen/Qwen2-0.5B-Instruct",
        "enable_ab_testing": True,
        "enable_caching": True
    }
    
    service = create_production_service(config)
    
    try:
        # Start service
        service.start()
        
        # Test sentiment analysis
        print("\n📊 Testing Sentiment Analysis...")
        sentiment_result = service.analyze_sentiment(
            "Apple reports record earnings with 20% revenue growth",
            user_id="test_user_1"
        )
        print(f"Sentiment: {sentiment_result['sentiment']}")
        print(f"Confidence: {sentiment_result['confidence']:.2f}")
        print(f"Response Time: {sentiment_result['response_time']:.3f}s")
        print(f"A/B Variant: {sentiment_result['variant']}")
        
        # Test trade signal generation
        print("\n📈 Testing Trade Signal Generation...")
        market_data = {
            "symbol": "AAPL",
            "price": 175.50,
            "rsi": 65,
            "macd": "Bullish",
            "volume": 85000000,
            "news": "Strong earnings report"
        }
        
        signal_result = service.generate_trade_signal(
            market_data,
            user_id="test_user_2"
        )
        print(f"Signal: {signal_result['signal']}")
        print(f"Confidence: {signal_result['confidence']:.2f}")
        print(f"Response Time: {signal_result['response_time']:.3f}s")
        print(f"A/B Variant: {signal_result['variant']}")
        
        # Show monitoring summary
        time.sleep(2)  # Let metrics accumulate
        print("\n🔍 Monitoring Summary:")
        service.dashboard.print_summary()
        
        # Show A/B test results
        print("\n🧪 A/B Test Results:")
        ab_results = service.get_ab_test_results()
        for exp_name, results in ab_results.items():
            print(f"Experiment: {exp_name}")
            print(f"Status: {results.get('status', 'unknown')}")
            if 'variants' in results:
                for variant_name, variant_data in results['variants'].items():
                    print(f"  {variant_name}: {variant_data.get('sample_size', 0)} samples")
        
        # Export monitoring data
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_file = f"production_monitoring_{timestamp}.json"
        service.export_monitoring_data(export_file)
        print(f"\n💾 Monitoring data exported to: {export_file}")
        
        print("\n✅ Production service test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during service test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean shutdown
        service.stop()


if __name__ == "__main__":
    main()
