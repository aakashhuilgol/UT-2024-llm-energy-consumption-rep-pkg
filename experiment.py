# experiment.py
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from data_processing import AlpacaDatasetProcessor
from model_inference import LLM
from power_metrics_tracker import PowerMetricsTracker

def main(model_name, max_new_tokens, input_range, nr_samples, idle_duration):
    token_secret = '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'

    tracker = PowerMetricsTracker(idle_duration)

    # Initialize the dataset processor
    processor = AlpacaDatasetProcessor()
    
    # Get all prompts from the dataset
    all_prompts = processor.get_prompts()

    # Filter prompts based on the specified length range
    filtered_prompts = processor.filter_sample_prompts(all_prompts, length_range=input_range, nr_samples=nr_samples)

    # Initialize the model for inference
    model = LLM(model_name=model_name, access_token=token_secret) 

    if max_new_tokens!=None: range_gen_tokens=[max_new_tokens]
    else: range_gen_tokens=[50,100,200]
    for max_gen_tokens in range_gen_tokens:
        if input_range[1] == 60: output = 'logs/'+model_name+str(max_gen_tokens)+'-short.txt'
        else: output = 'logs/'+model_name+str(max_gen_tokens)+'-long.txt'
        
        # Track power metrics and generate text for all sampled prompts
        tracker.run_powermetrics(output=output)

        for prompt_text in tqdm(filtered_prompts):
            # print(f"Prompt: {prompt_text}")
            generated_text = model.generate_text(prompt_text, max_new_tokens=max_gen_tokens)
            # print(f"Generated Text: {generated_text}")

        tracker.stop_powermetrics()
    del model  # Delete the existing model instance
    torch.cuda.empty_cache()  # Clear PyTorch cache (for CUDA devices; MPS doesn’t have a direct cache clearing function)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference on LLM and track power metrics.")
    parser.add_argument("--model_name", type=str, choices=["bloom", "opt", "flan-t5", "gpt-j"], required=True, help="Name of the model (e.g., 'bloom').")
    parser.add_argument("--max_new_tokens", type=int, required=False, help="Maximum number of new tokens to generate.")
    parser.add_argument("--input_length", type=str, choices=["short", "long"], required=True, help="Determines minimum and maximum input length for prompts.")
    parser.add_argument("--nr_samples", type=int, required=True, help="Number of random samples to take from filtered prompts.")
    parser.add_argument("--idle", type=int, required=True, help="Track Idle State (in seconds)")

    args = parser.parse_args()
    
    # Translate input_length to the correct tuple
    if args.input_length == "short": input_range = (0, 60)
    elif args.input_length == "long": input_range = (60, 110)
    
    main(args.model_name, args.max_new_tokens, input_range, args.nr_samples, args.idle)
