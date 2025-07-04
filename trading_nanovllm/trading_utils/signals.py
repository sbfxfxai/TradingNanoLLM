from ..llm import TradingLLM
from ..sampling import SamplingParams

def generate_trade_signal(data: str, llm: TradingLLM, sampling_params: SamplingParams):
    prompt = f"Based on the following market data: {data}\nGenerate a trade signal (Buy/Sell/Hold):"
    output = llm.generate([prompt], sampling_params)[0]["text"]
    return output.split("Trade Signal:")[-1].strip()
