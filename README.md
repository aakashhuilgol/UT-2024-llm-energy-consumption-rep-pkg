# LLM Energy Consumption during Inference 

This project performs inference using various Large Language Models (LLMs) while tracking it's energy consumption using power-metrics. It utilizes the Alpaca dataset for prompts and enables filtering and sampling based on prompt length. The power metrics are logged using the `powermetrics` utility, and the results are displayed in a structured format.

## Project Structure

```plaintext
.
├── data_processing.py       # Handles loading and processing of the Alpaca dataset
├── experiment.py            # Main script to run the inference and track power metrics
├── model_inference.py       # Contains the LLM class for loading models and generating text
└── power_metrics_tracker.py  # Manages the tracking of power metrics during execution
```

## Installation
```pip install torch transformers datasets tqdm huggingface-cli```  
Additionally, you may need to grant password-less sudo access for the powermetrics command.

## Example
You can run the experiment using the following command:

```python3 experiment.py --model_name "bloom" --max_new_tokens 100 --length_range 9 58 --nr_samples 5```  
Command-Line Arguments:  
`--model_name`: The name of the model to use (e.g., 'bloom').  
`--max_new_tokens`: Maximum number of new tokens to generate for each prompt.  
`--length_range`: Minimum and maximum length for filtering prompts (provide two integers).  
`--nr_samples`: Number of random samples to take from the filtered prompts.  
