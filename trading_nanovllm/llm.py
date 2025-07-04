import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .sampling import SamplingParams

class TradingLLM:
    def __init__(self, model_path, enforce_eager=True, tensor_parallel_size=1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        if not enforce_eager and self.device == "cuda":
            self.model = torch.compile(self.model)  # Torch compilation for speed

    def generate(self, prompts, sampling_params: SamplingParams):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            do_sample=True,
        )
        return [{"text": self.tokenizer.decode(output, skip_special_tokens=True)} for output in outputs]
