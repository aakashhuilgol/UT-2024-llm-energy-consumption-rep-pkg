# model_inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from data_processing import AlpacaDatasetProcessor
from tqdm import tqdm

class LLM:
    def __init__(self, model_name="bloom", access_token=None, device="mps"):
        if model_name == "bloom": self.model_name = "bigscience/bloom-1b1"
        if model_name == "opt": self.model_name = "facebook/opt-6.7b"
        if model_name == "gpt-j": self.model_name = "EleutherAI/gpt-j-6B"
        if model_name == "flan-t5": self.model_name = "google/flan-t5-large"
        self.access_token = access_token
        self.device = torch.device(device)

        print("Loading tokenizer and model...")
        with tqdm(total=2, desc="Loading Model") as pbar:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.access_token)
            pbar.update(1)

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.access_token, device_map={'': self.device})
            pbar.update(1)
        print("Model and tokenizer loaded.")

    def generate_text(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

