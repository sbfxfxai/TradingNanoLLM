import time
import random
from trading_nanovllm import TradingLLM, SamplingParams

llm = TradingLLM("Qwen/Qwen2-0.5B-Instruct", enforce_eager=False)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [f"Analyze market data: {random.randint(100, 1024)} tokens" for _ in range(256)]
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

total_tokens = sum(len(output["text"].split()) for output in outputs)
throughput = total_tokens / (end_time - start_time)
print(f"Time: {end_time - start_time:.2f}s, Throughput: {throughput:.2f} tokens/s")
