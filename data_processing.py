# data_processing.py
from datasets import load_dataset
from tqdm import tqdm
import random

class AlpacaDatasetProcessor:
    def __init__(self, dataset_name="tatsu-lab/alpaca"):
        """
        Initializes the Alpaca dataset processor by loading the dataset.
        Displays a progress bar while loading.
        """
        self.dataset_name = dataset_name
        print("Loading the Alpaca dataset...")
        self.dataset = load_dataset(self.dataset_name)
        print("Dataset loaded.")

    def get_prompts(self):
        """
        Extracts and returns the 'instruction' and 'input' fields as combined prompts.
        
        Returns:
            list: A list of all prompts from the dataset.
        """
        prompts = []
        print("Processing prompts...")
        for example in tqdm(self.dataset['train'], desc="Loading Prompts"):
            instruction = example['instruction']
            input_text = example['input']

            # Combine instruction and input as a single prompt
            if input_text:
                prompt = f"{instruction}\n{input_text}"
            else:
                prompt = instruction

            prompts.append(prompt)

        return prompts

    def filter_sample_prompts(self, prompts, nr_samples=0, length_range=None):
        """
        Filters prompts by the number of characters in the string and randomly samples a specified number of prompts from the provided list.
        
        Args:
            prompts (list): List of prompts.
            length_range (tuple): A tuple (min_length, max_length) to filter prompts by string length.
            nr_samples (int): Number of samples to take.
        
        Returns:
            list: A list of filtered and sampled prompts within the specified string length range.
        """
        filtered_prompts = []
        print(f"Filtering prompts by string length: {length_range}...")
        for prompt in tqdm(prompts, desc="Filtering Prompts"):
            # Get the length of the prompt
            prompt_length = len(prompt)

            # Apply length filtering if specified
            if length_range:
                min_length, max_length = length_range
                if min_length <= prompt_length <= max_length:
                    filtered_prompts.append(prompt)
            else:
                filtered_prompts.append(prompt)

        if nr_samples > len(filtered_prompts):
            print(f"Requested number of samples ({nr_samples}) exceeds available filtered prompts ({len(filtered_prompts)}).")
        else: filtered_prompts = random.sample(filtered_prompts, nr_samples)

        print("Number of filtered prompts:", len(filtered_prompts))
        return filtered_prompts

    def display_sample_prompts(self, prompts, num_samples=3):
        """
        Displays a few sample prompts from the dataset.
        """
        for i in range(min(num_samples, len(prompts))):
            print(f"Prompt {i+1}: {prompts[i]}")

# Example usage (uncomment this if you want to test this file directly)
# if __name__ == "__main__":
#     processor = AlpacaDatasetProcessor()
#     all_prompts = processor.get_prompts()
#     filtered_prompts = processor.filter_prompts_by_string_length(all_prompts, length_range=(9, 58))
#     sampled_prompts = processor.sample_prompts(filtered_prompts, nr_samples=5)
#     processor.display_sample_prompts(sampled_prompts, 3)
