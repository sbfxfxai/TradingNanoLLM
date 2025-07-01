"""Fine-tuning script for TradingNanoLLM on trading-specific data."""

import os
import json
import argparse
import logging
from typing import List, Dict, Any
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset(output_file: str = "trading_dataset.json"):
    """Create a sample trading dataset for fine-tuning."""
    
    sample_data = [
        # Sentiment analysis examples
        {
            "input": "Analyze the sentiment of the following financial news: Apple reports record-breaking Q2 earnings with 15% revenue growth.",
            "output": "Sentiment: Positive. This news indicates strong financial performance with significant revenue growth, which is likely to have a positive impact on Apple's stock price and investor confidence."
        },
        {
            "input": "Analyze the sentiment of the following financial news: Tesla faces production delays due to supply chain issues.",
            "output": "Sentiment: Negative. Production delays suggest operational challenges that could impact delivery targets and revenue, potentially leading to negative market reaction."
        },
        {
            "input": "Analyze the sentiment of the following financial news: Federal Reserve announces 0.25% interest rate cut.",
            "output": "Sentiment: Positive. Interest rate cuts typically stimulate economic activity and are generally viewed favorably by equity markets as they reduce borrowing costs."
        },
        {
            "input": "Analyze the sentiment of the following financial news: Oil prices surge 8% on geopolitical tensions.",
            "output": "Sentiment: Mixed. While positive for oil companies and energy sector, higher oil prices can negatively impact consumer spending and inflation expectations."
        },
        
        # Trade signal examples
        {
            "input": "Generate a trade signal based on: AAPL price: $175, RSI: 70, MACD: Bullish crossover, Volume: High",
            "output": "Trade Signal: Buy. The bullish MACD crossover combined with high volume suggests strong upward momentum. However, RSI at 70 indicates the stock is approaching overbought territory, so consider partial position sizing."
        },
        {
            "input": "Generate a trade signal based on: TSLA price: $200, RSI: 30, MACD: Bearish divergence, Support at $195",
            "output": "Trade Signal: Hold. The low RSI suggests oversold conditions which could lead to a bounce, but bearish MACD divergence indicates underlying weakness. Wait for clearer signals near the $195 support level."
        },
        {
            "input": "Generate a trade signal based on: SPY price: $420, RSI: 50, MACD: Neutral, Moving averages converging",
            "output": "Trade Signal: Hold. Neutral technical indicators suggest a lack of clear directional bias. The converging moving averages indicate potential breakout, but direction is uncertain. Wait for confirmation."
        },
        {
            "input": "Generate a trade signal based on: BTC price: $45000, RSI: 80, MACD: Overbought, High volatility",
            "output": "Trade Signal: Sell. Extremely overbought conditions with RSI at 80 and overbought MACD suggest a potential correction. High volatility increases risk. Consider taking profits or reducing exposure."
        },
        
        # Market analysis examples
        {
            "input": "Analyze market conditions: Tech sector down 3%, Rising interest rates, Strong employment data",
            "output": "Market Analysis: Mixed conditions. Strong employment data is fundamentally positive but rising interest rates create headwinds for growth stocks, explaining tech sector weakness. Expect continued rotation from growth to value stocks."
        },
        {
            "input": "Analyze market conditions: VIX at 25, S&P 500 near resistance, Earnings season beginning",
            "output": "Market Analysis: Cautious outlook. Elevated VIX indicates increased uncertainty and the market at resistance suggests limited upside. Earnings season could provide catalysts but also increase volatility. Defensive positioning recommended."
        },
        
        # Risk assessment examples
        {
            "input": "Assess risk for: Small-cap growth stock, High beta (2.5), Recent IPO, No profitability",
            "output": "Risk Assessment: High Risk. Multiple risk factors include small size, high volatility (beta 2.5), limited operating history as recent IPO, and lack of profitability. Suitable only for aggressive risk tolerance with small position sizes."
        },
        {
            "input": "Assess risk for: Large-cap dividend stock, Beta 0.8, 20-year track record, Stable earnings",
            "output": "Risk Assessment: Low Risk. Large established company with long track record and stable earnings provides defensive characteristics. Low beta and dividend provide downside protection. Suitable for conservative portfolios."
        },
        
        # Portfolio management examples
        {
            "input": "Portfolio recommendation: Conservative investor, $100k portfolio, Age 60, Retirement in 5 years",
            "output": "Portfolio Recommendation: Allocate 60% bonds/fixed income, 30% large-cap dividend stocks, 10% cash/short-term investments. Focus on capital preservation and income generation. Avoid high-risk growth stocks and maintain liquidity for retirement needs."
        },
        {
            "input": "Portfolio recommendation: Aggressive investor, $50k portfolio, Age 25, Long-term growth focus",
            "output": "Portfolio Recommendation: Allocate 80% equities (mix of growth and international), 15% emerging markets/small-cap, 5% cash. Focus on growth potential with long time horizon allowing for higher volatility. Consider index funds for diversification."
        }
    ]
    
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Created sample dataset with {len(sample_data)} examples: {output_file}")
    return output_file


def prepare_dataset(data_file: str, tokenizer, max_length: int = 512) -> Dataset:
    """Prepare dataset for training."""
    
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Format data for causal language modeling
    texts = []
    for item in data:
        # Format as instruction-following
        text = f"<|instruction|>{item['input']}<|response|>{item['output']}<|end|>"
        texts.append(text)
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Prepared dataset with {len(tokenized_dataset)} examples")
    return tokenized_dataset


def setup_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None
) -> LoraConfig:
    """Set up LoRA configuration for efficient fine-tuning."""
    
    if target_modules is None:
        # Common target modules for Qwen models
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def train_model(
    model_name: str,
    data_file: str,
    output_dir: str = "./trading_model",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    use_lora: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    seed: int = 42
):
    """Train the model on trading data."""
    
    # Set random seed
    set_seed(seed)
    
    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Apply LoRA if specified
    if use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = setup_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Prepare dataset
    logger.info(f"Preparing dataset from {data_file}")
    dataset = prepare_dataset(data_file, tokenizer, max_length)
    
    # Split dataset (80% train, 20% eval)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # For causal language modeling
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "base_model": model_name,
        "dataset_file": data_file,
        "num_examples": len(dataset),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_lora": use_lora,
        "max_length": max_length,
    }
    
    with open(os.path.join(output_dir, "training_info.json"), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info("Training completed successfully!")
    
    return trainer


def test_trained_model(model_dir: str, test_prompts: List[str] = None):
    """Test the trained model with sample prompts."""
    
    if test_prompts is None:
        test_prompts = [
            "Analyze the sentiment of the following financial news: Microsoft beats earnings expectations with strong cloud growth.",
            "Generate a trade signal based on: NVDA price: $800, RSI: 75, MACD: Bullish, Strong earnings report",
            "Assess risk for: Cryptocurrency investment, High volatility, Regulatory uncertainty"
        ]
    
    logger.info(f"Testing trained model from {model_dir}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    print("\\n🧪 Testing Trained Model")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\\nTest {i}: {prompt}")
        print("-" * 30)
        
        # Format input
        formatted_input = f"<|instruction|>{prompt}<|response|>"
        inputs = tokenizer(formatted_input, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "<|response|>" in response:
            response = response.split("<|response|>")[1].strip()
        
        print(f"Response: {response}")


def main():
    """Main fine-tuning function with CLI interface."""
    
    parser = argparse.ArgumentParser(description="Fine-tune TradingNanoLLM")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct", help="Base model name")
    parser.add_argument("--data", help="Training data file (JSON format)")
    parser.add_argument("--output", default="./trading_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample dataset")
    parser.add_argument("--test-only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    try:
        # Create sample data if requested
        if args.create_sample_data:
            data_file = create_sample_dataset()
            if not args.data:
                args.data = data_file
        
        # Test existing model
        if args.test_only:
            if not os.path.exists(args.output):
                print(f"❌ Model directory not found: {args.output}")
                return
            test_trained_model(args.output)
            return
        
        # Check data file
        if not args.data:
            print("❌ No data file specified. Use --data or --create-sample-data")
            return
        
        if not os.path.exists(args.data):
            print(f"❌ Data file not found: {args.data}")
            return
        
        # Train model
        trainer = train_model(
            model_name=args.model,
            data_file=args.data,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            use_lora=not args.no_lora
        )
        
        # Test trained model
        test_trained_model(args.output)
        
        print(f"\\n✅ Fine-tuning completed! Model saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()
