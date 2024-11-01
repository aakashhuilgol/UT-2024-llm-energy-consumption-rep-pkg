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


--------


# Energy Consumption of Large Language Models and the impact of maximizing prompt and output length
This repository is a companion page for the following paper:
> Sanne Eeckhout, Aakash Huilgol. 2024. Energy Consumption of Large Language Models and the impact of maximizing prompt and output length. University Twente.

It contains all the material required for replicating the study, including: loading and using OPT, BLOOM and FLAN-T5 for inference using the Alpca Dataset, tracking energy consumption using powermetrics, and parsing the powermetrics logs.

## How to cite us
The scientific article describing design, execution, and main results of this study is available [here](https://github.com/aakashhuilgol/UT-2024-llm-energy-consumption-rep-pkg/blob/main/original_paper.pdf).<br> 
If this study is helping your research, consider to cite it is as follows, thanks!

```
@article{,
  title={Energy Consumption of Large Language Models and the impact of maximizing prompt and output length},
  author={Eeckhout, Sanne and Aakash, Huilgol.},
  journal={Green Software Development},
  year={2024},
  publisher={University Twente}
}
```

## Quick start


### Getting started

The packages used by this repository can be installed using ```pip install torch transformers datasets tqdm huggingface-cli```  
Additionally, you may need to grant password-less sudo access for the powermetrics command.

### Example

The code can be run using the .. file in the [src](src/) folder. An example 
```python3 experiment.py --model_name "bloom" --max_new_tokens 100 --length_range 0 60 --nr_samples 5```  
Which takes the following arguments:
`--model_name`: The name of the model to use (e.g., 'bloom').
`--max_new_tokens`: Maximum number of new tokens to generate for each prompt. If none is provided, it will run for three different output lengths: 50, 100 and 200.
`--input_length`: Minimum and maximum length for filtering prompts (provide two integers) by input prompt length.
`--idle`: Measurement of idle state preceeding and succeeding each run. 
`--nr_samples`: Number of random samples to take from the filtered prompts.  


## Repository Structure
This is the root directory of the repository. The directory is structured as follows:

    template-replication-package
     .
     |
     |--- src/                             Source code used in the thesis / paper
            |
            |--- data_processing.py        Handles loading and processing of the Alpaca dataset 
            |--- experiment.py             Main script to run the inference and track power metrics
            |--- model_inference.py        Contains the LLM class for loading models and generating text
            |--- power_metrics_tracker.py  Manages the tracking of power metrics during execution
            |--- powerlog-parse.ipynb      Parsing of the logs created by Powermetrics, creation of graphs presented in the paper and the statistical analysis
     |--- documentation/                   Contains the figures presented in the papers.do
     |
     |--- data/                            Raw logs created by Powermetrics

## Repository license
[MIT license](https://opensource.org/licenses/MIT)
