# experiment.py
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from data_processing import AlpacaDatasetProcessor
from model_inference import LLM
from power_metrics_tracker import PowerMetricsTracker

def main(model_name, max_new_tokens, length_range, nr_samples):
    token_secret = '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'

    tracker = PowerMetricsTracker()

    # Initialize the dataset processor
    processor = AlpacaDatasetProcessor()
    
    # Get all prompts from the dataset
    all_prompts = processor.get_prompts()

    # Filter prompts based on the specified length range
    filtered_prompts = processor.filter_sample_prompts(all_prompts, length_range=length_range, nr_samples=nr_samples)

    # Initialize the model for inference
    model = LLM(model_name=model_name, access_token=token_secret) 

    # Track power metrics and generate text for all sampled prompts
    tracker.run_powermetrics(output='powerlog-bloom.txt')

    for prompt_text in filtered_prompts:
        print(f"Prompt: {prompt_text}")
        generated_text = model.generate_text(prompt_text, max_new_tokens=max_new_tokens)
        print(f"Generated Text: {generated_text}")

    tracker.stop_powermetrics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference and track power metrics.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., 'bloom').")
    parser.add_argument("--max_new_tokens", type=int, required=True, help="Maximum number of new tokens to generate.")
    parser.add_argument("--length_range", type=int, nargs=2, required=True, help="Minimum and maximum length for filtering prompts.")
    parser.add_argument("--nr_samples", type=int, required=True, help="Number of random samples to take from filtered prompts.")

    args = parser.parse_args()
    
    # Convert length_range from list of strings to tuple of integers
    length_range = (args.length_range[0], args.length_range[1])
    
    main(args.model_name, args.max_new_tokens, length_range, args.nr_samples)
