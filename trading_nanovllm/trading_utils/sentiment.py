from ..llm import TradingLLM
from ..sampling import SamplingParams

def analyze_sentiment(text: str, llm: TradingLLM, sampling_params: SamplingParams):
    prompt = f"Analyze the sentiment of the following financial news: {text}\nSentiment (Positive/Neutral/Negative):"
    output = llm.generate([prompt], sampling_params)[0]["text"]
    return output.split("Sentiment:")[-1].strip()
