"""Production monitoring and metrics system for TradingNanoLLM."""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a single metric measurement."""
    timestamp: float
    value: float
    tags: Dict[str, str]


@dataclass
class PerformanceMetrics:
    """Performance metrics for model inference."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class TradingMetrics:
    """Trading-specific metrics."""
    sentiment_accuracy: float = 0.0
    signal_accuracy: float = 0.0
    prediction_confidence: float = 0.0
    model_drift_score: float = 0.0
    response_quality: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics for monitoring."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to keep metrics in memory
        """
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.latencies = deque(maxlen=1000)
        self.errors = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value."""
        if tags is None:
            tags = {}
            
        metric = MetricValue(
            timestamp=time.time(),
            value=value,
            tags=tags
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def record_latency(self, latency: float, endpoint: str = "inference"):
        """Record request latency."""
        self.record_metric(f"latency_{endpoint}", latency, {"endpoint": endpoint})
        
        with self.lock:
            self.latencies.append(latency)
    
    def record_error(self, error_type: str, endpoint: str = "inference"):
        """Record an error."""
        self.record_metric("errors", 1.0, {"type": error_type, "endpoint": endpoint})
        
        with self.lock:
            self.errors.append(time.time())
    
    def get_performance_metrics(self, window_minutes: int = 5) -> PerformanceMetrics:
        """Get aggregated performance metrics for a time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self.lock:
            # Filter recent latencies
            recent_latencies = [lat for lat in self.latencies if lat > cutoff_time]
            recent_errors = [err for err in self.errors if err > cutoff_time]
            
            # Calculate metrics
            total_requests = len(recent_latencies) + len(recent_errors)
            successful_requests = len(recent_latencies)
            failed_requests = len(recent_errors)
            
            if total_requests == 0:
                return PerformanceMetrics()
            
            # Latency percentiles
            if recent_latencies:
                sorted_latencies = sorted(recent_latencies)
                avg_latency = sum(sorted_latencies) / len(sorted_latencies)
                p95_idx = int(0.95 * len(sorted_latencies))
                p99_idx = int(0.99 * len(sorted_latencies))
                p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1]
                p99_latency = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else sorted_latencies[-1]
            else:
                avg_latency = p95_latency = p99_latency = 0.0
            
            # Throughput (requests per second)
            throughput = total_requests / (window_minutes * 60)
            
            # Error rate
            error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
            
            return PerformanceMetrics(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_latency=avg_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                throughput=throughput,
                error_rate=error_rate
            )
    
    def get_metric_values(self, name: str, window_minutes: int = 5) -> List[MetricValue]:
        """Get metric values for a time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self.lock:
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def _cleanup_old_metrics(self):
        """Cleanup old metrics periodically."""
        while True:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                with self.lock:
                    for name, values in self.metrics.items():
                        # Remove old values
                        while values and values[0].timestamp < cutoff_time:
                            values.popleft()
                
                # Sleep for 1 hour before next cleanup
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                time.sleep(60)


class HealthChecker:
    """Health checking system for TradingNanoLLM."""
    
    def __init__(self, llm, check_interval: int = 60):
        """Initialize health checker.
        
        Args:
            llm: TradingLLM instance to monitor
            check_interval: Health check interval in seconds
        """
        self.llm = llm
        self.check_interval = check_interval
        self.health_status = {"status": "unknown", "last_check": None, "issues": []}
        self.running = False
        self.check_thread = None
    
    def start(self):
        """Start health checking."""
        if self.running:
            return
            
        self.running = True
        self.check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.check_thread.start()
        logger.info("Health checker started")
    
    def stop(self):
        """Stop health checking."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        logger.info("Health checker stopped")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_status.copy()
    
    def _health_check_loop(self):
        """Main health check loop."""
        while self.running:
            try:
                self._perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(10)  # Short retry on error
    
    def _perform_health_check(self):
        """Perform a single health check."""
        issues = []
        
        try:
            # Test basic inference
            from ..sampling import SamplingParams
            test_prompt = "Test health check prompt"
            test_params = SamplingParams(temperature=0.1, max_tokens=10)
            
            start_time = time.time()
            result = self.llm.generate([test_prompt], test_params)
            response_time = time.time() - start_time
            
            # Check response time
            if response_time > 10.0:  # 10 second threshold
                issues.append(f"Slow response time: {response_time:.2f}s")
            
            # Check if response is valid
            if not result or not result[0] or not result[0].get("text"):
                issues.append("Invalid response from model")
            
            # Check memory usage (if available)
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 90:
                    issues.append(f"High memory usage: {memory_percent:.1f}%")
            except ImportError:
                pass
            
            # Check GPU memory (if CUDA available)
            if self.llm.device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                        
                        if memory_allocated > 7.0:  # 7GB threshold
                            issues.append(f"High GPU memory usage: {memory_allocated:.1f}GB allocated")
                except Exception:
                    pass
            
            # Update health status
            status = "healthy" if not issues else "degraded" if len(issues) <= 2 else "unhealthy"
            
            self.health_status = {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "response_time": response_time,
                "issues": issues,
                "device": self.llm.device,
                "model_stats": self.llm.get_stats()
            }
            
        except Exception as e:
            self.health_status = {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "issues": [f"Health check failed: {str(e)}"],
                "error": str(e)
            }


class AlertManager:
    """Alert management system for production monitoring."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_rules = {}
        self.alert_handlers = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict], bool], 
                      severity: str = "warning", cooldown: int = 300):
        """Add an alert rule.
        
        Args:
            name: Alert rule name
            condition: Function that returns True if alert should fire
            severity: Alert severity (info, warning, critical)
            cooldown: Minimum seconds between alerts of same type
        """
        self.alert_rules[name] = {
            "condition": condition,
            "severity": severity,
            "cooldown": cooldown,
            "last_fired": 0
        }
    
    def add_alert_handler(self, handler: Callable[[Dict], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check if condition is met
                if rule["condition"](metrics):
                    # Check cooldown
                    if current_time - rule["last_fired"] >= rule["cooldown"]:
                        self._fire_alert(rule_name, rule["severity"], metrics)
                        rule["last_fired"] = current_time
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _fire_alert(self, name: str, severity: str, context: Dict[str, Any]):
        """Fire an alert."""
        alert = {
            "name": name,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        }
        
        # Store alert
        self.alert_history.append(alert)
        self.active_alerts[name] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def resolve_alert(self, name: str):
        """Resolve an active alert."""
        if name in self.active_alerts:
            del self.active_alerts[name]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self.active_alerts.values())


class MonitoringDashboard:
    """Simple monitoring dashboard for TradingNanoLLM."""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 health_checker: HealthChecker, alert_manager: AlertManager):
        """Initialize monitoring dashboard."""
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.alert_manager = alert_manager
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        performance_metrics = self.metrics_collector.get_performance_metrics()
        health_status = self.health_checker.get_health_status()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": asdict(performance_metrics),
            "health": health_status,
            "alerts": {
                "active": active_alerts,
                "count": len(active_alerts)
            },
            "system": self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        dashboard_data = self.get_dashboard_data()
        
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
    
    def print_summary(self):
        """Print a summary of current status."""
        data = self.get_dashboard_data()
        
        print("🔍 TradingNanoLLM Monitoring Summary")
        print("=" * 50)
        
        # Health status
        health = data["health"]
        status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "unknown": "❓"}
        print(f"Health: {status_emoji.get(health['status'], '❓')} {health['status'].upper()}")
        
        if health.get("issues"):
            print("Issues:")
            for issue in health["issues"]:
                print(f"  • {issue}")
        
        # Performance metrics
        perf = data["performance"]
        print(f"\n📊 Performance (5min window):")
        print(f"  Requests: {perf['total_requests']} (success: {perf['successful_requests']}, failed: {perf['failed_requests']})")
        print(f"  Latency: avg={perf['avg_latency']:.3f}s, p95={perf['p95_latency']:.3f}s")
        print(f"  Throughput: {perf['throughput']:.2f} req/s")
        print(f"  Error Rate: {perf['error_rate']:.1%}")
        
        # Alerts
        alerts = data["alerts"]
        if alerts["count"] > 0:
            print(f"\n🚨 Active Alerts ({alerts['count']}):")
            for alert in alerts["active"]:
                print(f"  • {alert['severity'].upper()}: {alert['name']}")
        else:
            print("\n✅ No active alerts")
        
        # System info
        if "error" not in data["system"]:
            sys_info = data["system"]
            print(f"\n💻 System:")
            print(f"  CPU: {sys_info['cpu_percent']:.1f}%")
            print(f"  Memory: {sys_info['memory_percent']:.1f}%")


# Default alert rules for TradingNanoLLM
def setup_default_alerts(alert_manager: AlertManager):
    """Set up default alert rules."""
    
    # High error rate
    alert_manager.add_alert_rule(
        "high_error_rate",
        lambda m: m.get("performance", {}).get("error_rate", 0) > 0.1,  # 10% error rate
        severity="critical",
        cooldown=300
    )
    
    # High latency
    alert_manager.add_alert_rule(
        "high_latency",
        lambda m: m.get("performance", {}).get("p95_latency", 0) > 5.0,  # 5 second P95
        severity="warning",
        cooldown=300
    )
    
    # Health degraded
    alert_manager.add_alert_rule(
        "health_degraded",
        lambda m: m.get("health", {}).get("status") in ["degraded", "unhealthy"],
        severity="warning",
        cooldown=600
    )
    
    # Low throughput
    alert_manager.add_alert_rule(
        "low_throughput",
        lambda m: m.get("performance", {}).get("throughput", 0) < 0.1,  # < 0.1 req/s
        severity="info",
        cooldown=600
    )


# Simple alert handlers
def console_alert_handler(alert: Dict[str, Any]):
    """Print alerts to console."""
    print(f"🚨 ALERT [{alert['severity'].upper()}]: {alert['name']} at {alert['timestamp']}")


def file_alert_handler(filepath: str):
    """Create a file-based alert handler."""
    def handler(alert: Dict[str, Any]):
        with open(filepath, 'a') as f:
            f.write(f"{json.dumps(alert)}\n")
    return handler
