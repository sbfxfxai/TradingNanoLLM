"""Real-world validation script for TradingNanoLLM with actual models."""

import time
import json
import argparse
import logging
from typing import Dict, List, Any
import warnings
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from trading_nanovllm import TradingLLM, SamplingParams
from trading_nanovllm.trading_utils import SentimentAnalyzer, TradeSignalGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealWorldValidator:
    """Comprehensive real-world validation suite."""
    
    def __init__(self, model_path: str = "Qwen/Qwen2-0.5B-Instruct"):
        """Initialize validator with specified model."""
        self.model_path = model_path
        self.llm = None
        self.results = {
            "model_info": {"path": model_path, "loaded": False},
            "performance": {},
            "accuracy": {},
            "trading_scenarios": {},
            "edge_cases": {},
            "optimization_impact": {}
        }
        
    def load_model(self, enable_optimizations: bool = True) -> bool:
        """Load the model and return success status."""
        try:
            print(f"🚀 Loading model: {self.model_path}")
            print("📥 This may take a few minutes for first-time download...")
            
            start_time = time.time()
            self.llm = TradingLLM(
                self.model_path,
                enforce_eager=not enable_optimizations,
                enable_prefix_caching=enable_optimizations,
                max_batch_size=16
            )
            load_time = time.time() - start_time
            
            self.results["model_info"]["loaded"] = True
            self.results["model_info"]["load_time"] = load_time
            self.results["model_info"]["device"] = self.llm.device
            self.results["model_info"]["dtype"] = str(self.llm.dtype)
            
            print(f"✅ Model loaded successfully!")
            print(f"   Device: {self.llm.device}")
            print(f"   Data Type: {self.llm.dtype}")
            print(f"   Load Time: {load_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.results["model_info"]["error"] = str(e)
            return False
    
    def test_performance_baseline(self) -> Dict[str, Any]:
        """Test basic performance metrics."""
        print("\n📊 Testing Performance Baseline")
        print("=" * 50)
        
        test_prompts = [
            "Analyze market sentiment: Apple stock rises 5% on strong iPhone sales.",
            "Generate trade signal: TSLA $250, RSI 75, MACD bullish, high volume.",
            "Risk assessment: Small-cap biotech stock, no revenue, clinical trials.",
            "Portfolio advice: Conservative investor, $100k, retirement in 10 years.",
            "Market outlook: Fed raises rates, inflation at 3%, unemployment low."
        ]
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
        
        # Single request latency
        print("🔍 Testing single request latency...")
        start_time = time.time()
        output = self.llm.generate([test_prompts[0]], sampling_params)[0]
        single_latency = time.time() - start_time
        single_tokens = len(output["text"].split())
        
        print(f"   Single Request: {single_latency:.3f}s")
        print(f"   Tokens Generated: {single_tokens}")
        print(f"   Tokens/Second: {single_tokens/single_latency:.2f}")
        
        # Batch throughput
        print("\n🚀 Testing batch throughput...")
        start_time = time.time()
        outputs = self.llm.generate(test_prompts, sampling_params)
        batch_time = time.time() - start_time
        total_tokens = sum(len(output["text"].split()) for output in outputs)
        
        print(f"   Batch Time: {batch_time:.3f}s")
        print(f"   Total Tokens: {total_tokens}")
        print(f"   Throughput: {total_tokens/batch_time:.2f} tokens/s")
        print(f"   Requests/Second: {len(test_prompts)/batch_time:.2f}")
        
        performance_results = {
            "single_request_latency": single_latency,
            "single_request_tokens": single_tokens,
            "single_request_tps": single_tokens / single_latency,
            "batch_time": batch_time,
            "batch_total_tokens": total_tokens,
            "batch_throughput": total_tokens / batch_time,
            "requests_per_second": len(test_prompts) / batch_time
        }
        
        self.results["performance"] = performance_results
        return performance_results
    
    def test_trading_scenarios(self) -> Dict[str, Any]:
        """Test realistic trading scenarios."""
        print("\n📈 Testing Trading Scenarios")
        print("=" * 50)
        
        scenarios = {
            "earnings_surprise": {
                "input": "Apple reports Q2 earnings: Revenue $97.3B vs $94.5B expected, EPS $1.52 vs $1.43 expected. iPhone revenue up 12% YoY.",
                "expected_sentiment": "Positive",
                "context": "Strong earnings beat"
            },
            "market_crash": {
                "input": "S&P 500 drops 8% in single session as banking crisis spreads, VIX spikes to 45, Treasury yields plummet.",
                "expected_sentiment": "Negative", 
                "context": "Market crash scenario"
            },
            "technical_breakout": {
                "input": "NVDA breaks above $900 resistance, RSI 68, MACD bullish crossover, volume 150% average, AI chip demand strong.",
                "expected_signal": "Buy",
                "context": "Technical breakout with fundamental support"
            },
            "oversold_bounce": {
                "input": "TSLA at $180 support level, RSI 25 (oversold), insider buying reported, short interest 15% of float.",
                "expected_signal": "Buy",
                "context": "Oversold conditions with positive catalysts"
            },
            "profit_taking": {
                "input": "Bitcoin at $65,000 all-time high, RSI 85, fear/greed index at 'extreme greed', leverage ratios elevated.",
                "expected_signal": "Sell",
                "context": "Overextended market conditions"
            }
        }
        
        sentiment_analyzer = SentimentAnalyzer(self.llm)
        signal_generator = TradeSignalGenerator(self.llm)
        
        scenario_results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"\n🎯 Testing: {scenario['context']}")
            print(f"Input: {scenario['input'][:80]}...")
            
            try:
                if "expected_sentiment" in scenario:
                    result = sentiment_analyzer.analyze_with_confidence(scenario["input"])
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]
                    
                    print(f"Result: {sentiment} (confidence: {confidence:.2f})")
                    
                    scenario_results[scenario_name] = {
                        "type": "sentiment",
                        "result": sentiment,
                        "confidence": confidence,
                        "expected": scenario["expected_sentiment"],
                        "correct": sentiment == scenario["expected_sentiment"]
                    }
                    
                elif "expected_signal" in scenario:
                    result = signal_generator.generate_with_analysis(scenario["input"])
                    signal = result["signal"]
                    confidence = result["confidence"]
                    reasoning = result["reasoning"][:100] + "..." if len(result["reasoning"]) > 100 else result["reasoning"]
                    
                    print(f"Result: {signal} (confidence: {confidence:.1f}/10)")
                    print(f"Reasoning: {reasoning}")
                    
                    scenario_results[scenario_name] = {
                        "type": "signal",
                        "result": signal,
                        "confidence": confidence,
                        "reasoning": result["reasoning"],
                        "expected": scenario["expected_signal"],
                        "correct": signal == scenario["expected_signal"]
                    }
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                scenario_results[scenario_name] = {"error": str(e)}
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in scenario_results.values() 
                                if isinstance(r, dict) and r.get("correct", False))
        total_predictions = len([r for r in scenario_results.values() 
                               if isinstance(r, dict) and "correct" in r])
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\n📊 Scenario Testing Results:")
        print(f"   Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        self.results["trading_scenarios"] = {
            "scenarios": scenario_results,
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions
        }
        
        return scenario_results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling."""
        print("\n⚠️ Testing Edge Cases")
        print("=" * 50)
        
        edge_cases = {
            "empty_input": "",
            "very_long_input": "Analyze this market data: " + "AAPL $150, " * 200,
            "nonsense_input": "Quantum flux capacitor indicates purple elephant trading signals",
            "mixed_languages": "AAPL的股价今天上涨了5%, buy or sell?",
            "special_characters": "Stock price: $1,234.56 📈 RSI: 70% 🔥 MACD: 📊📈",
            "extreme_numbers": "Stock price: $99999999, volume: 1, RSI: 999"
        }
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
        edge_results = {}
        
        for case_name, input_text in edge_cases.items():
            print(f"\n🧪 Testing: {case_name}")
            
            try:
                start_time = time.time()
                output = self.llm.generate([input_text], sampling_params)[0]
                processing_time = time.time() - start_time
                
                result_text = output["text"][:100] + "..." if len(output["text"]) > 100 else output["text"]
                
                print(f"✅ Handled successfully ({processing_time:.3f}s)")
                print(f"Output: {result_text}")
                
                edge_results[case_name] = {
                    "success": True,
                    "processing_time": processing_time,
                    "output": output["text"],
                    "output_length": len(output["text"])
                }
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                edge_results[case_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        success_rate = sum(1 for r in edge_results.values() if r.get("success", False)) / len(edge_results)
        
        print(f"\n📊 Edge Case Results:")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Cases Handled: {sum(1 for r in edge_results.values() if r.get('success', False))}/{len(edge_results)}")
        
        self.results["edge_cases"] = {
            "cases": edge_results,
            "success_rate": success_rate
        }
        
        return edge_results
    
    def test_optimization_impact(self) -> Dict[str, Any]:
        """Compare performance with and without optimizations."""
        print("\n🚀 Testing Optimization Impact")
        print("=" * 50)
        
        test_prompts = [
            "Analyze sentiment: Tech stocks rally on AI breakthrough.",
            "Generate signal: AAPL $175, RSI 65, bullish MACD.",
            "Risk assessment: High-growth startup IPO."
        ] * 3  # Repeat for cache testing
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
        
        # Test with optimizations (current setup)
        print("🔍 Testing with optimizations enabled...")
        start_time = time.time()
        outputs_optimized = self.llm.generate(test_prompts, sampling_params)
        time_optimized = time.time() - start_time
        tokens_optimized = sum(len(o["text"].split()) for o in outputs_optimized)
        
        cache_stats = self.llm.prefix_cache.get_stats() if self.llm.prefix_cache else {}
        
        print(f"   Time: {time_optimized:.3f}s")
        print(f"   Throughput: {tokens_optimized/time_optimized:.2f} tokens/s")
        if cache_stats:
            print(f"   Cache Hit Rate: {cache_stats['hit_rate']:.1%}")
        
        # Create a new instance without optimizations for comparison
        print("\n🐌 Testing without optimizations...")
        try:
            llm_no_opt = TradingLLM(
                self.model_path,
                enforce_eager=True,
                enable_prefix_caching=False,
                max_batch_size=16
            )
            
            start_time = time.time()
            outputs_baseline = llm_no_opt.generate(test_prompts, sampling_params)
            time_baseline = time.time() - start_time
            tokens_baseline = sum(len(o["text"].split()) for o in outputs_baseline)
            
            print(f"   Time: {time_baseline:.3f}s")
            print(f"   Throughput: {tokens_baseline/time_baseline:.2f} tokens/s")
            
            # Calculate improvements
            speedup = time_baseline / time_optimized
            throughput_improvement = (tokens_optimized/time_optimized) / (tokens_baseline/time_baseline)
            
            print(f"\n📊 Optimization Results:")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Throughput Improvement: {throughput_improvement:.2f}x")
            
            optimization_results = {
                "time_optimized": time_optimized,
                "time_baseline": time_baseline,
                "speedup": speedup,
                "throughput_optimized": tokens_optimized/time_optimized,
                "throughput_baseline": tokens_baseline/time_baseline,
                "throughput_improvement": throughput_improvement,
                "cache_stats": cache_stats
            }
            
        except Exception as e:
            print(f"❌ Could not test baseline (model already loaded): {e}")
            optimization_results = {
                "time_optimized": time_optimized,
                "throughput_optimized": tokens_optimized/time_optimized,
                "cache_stats": cache_stats,
                "baseline_error": str(e)
            }
        
        self.results["optimization_impact"] = optimization_results
        return optimization_results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🧪 TradingNanoLLM Real-World Validation")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return self.results
        
        # Run all tests
        try:
            self.test_performance_baseline()
            self.test_trading_scenarios()
            self.test_edge_cases()
            self.test_optimization_impact()
            
            # Calculate overall score
            perf_score = min(100, (self.results["performance"]["batch_throughput"] / 10))  # 10 tokens/s = 100 points
            accuracy_score = self.results["trading_scenarios"]["accuracy"] * 100
            reliability_score = self.results["edge_cases"]["success_rate"] * 100
            
            overall_score = (perf_score + accuracy_score + reliability_score) / 3
            
            self.results["overall"] = {
                "performance_score": perf_score,
                "accuracy_score": accuracy_score,
                "reliability_score": reliability_score,
                "overall_score": overall_score
            }
            
            print(f"\n🏆 OVERALL VALIDATION RESULTS")
            print("=" * 60)
            print(f"Performance Score: {perf_score:.1f}/100")
            print(f"Accuracy Score: {accuracy_score:.1f}/100")
            print(f"Reliability Score: {reliability_score:.1f}/100")
            print(f"Overall Score: {overall_score:.1f}/100")
            
            if overall_score >= 80:
                print("🎉 EXCELLENT - Ready for production!")
            elif overall_score >= 60:
                print("✅ GOOD - Minor optimizations recommended")
            elif overall_score >= 40:
                print("⚠️ FAIR - Significant improvements needed")
            else:
                print("❌ POOR - Major issues require attention")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            self.results["error"] = str(e)
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save validation results to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main validation function with CLI interface."""
    parser = argparse.ArgumentParser(description="TradingNanoLLM Real-World Validation")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct", help="Model to validate")
    parser.add_argument("--output", help="Output file for results (default: auto-generated)")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (fewer tests)")
    
    args = parser.parse_args()
    
    try:
        validator = RealWorldValidator(args.model)
        
        if args.quick:
            print("🏃 Running quick validation...")
            validator.load_model()
            validator.test_performance_baseline()
            validator.test_trading_scenarios()
        else:
            validator.run_full_validation()
        
        validator.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
