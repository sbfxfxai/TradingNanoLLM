from trading_nanovllm import TradingLLM, SamplingParams
from trading_nanovllm.trading_utils import analyze_sentiment, generate_trade_signal

llm = TradingLLM("Qwen/Qwen2-0.5B-Instruct", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# Sentiment analysis
news = "Apple reports record-breaking Q2 earnings."
sentiment = analyze_sentiment(news, llm, sampling_params)
print(f"Sentiment: {sentiment}")

# Trade signal
data = "AAPL price: , RSI: 70, MACD: Bullish"
signal = generate_trade_signal(data, llm, sampling_params)
print(f"Trade Signal: {signal}")
