# PEARL

This is the code for the paper:

**PEARL: Fast, Scalable, and Non-Invasive Alzheimer's Dementia Screening with Language Models**

Under review

## Description

PEARL is a complete system for training GPT-2 models for the classification of Alzheimer's disease patients (AD) and healthy controls (CN), using perplexity as the main metric.

## Installation

### Prerequisites
see pip freeze file

### Setup
```bash
git clone <repository-url>
cd PEARL
pip install -r requirements.txt
```

## Execution

Modify the parameters in the file `src/run/run_experiment.py` in the `main()` function:

```python
dataset_name = 'adresso_final_fold'     # Dataset name
disease_class = 'ad'                    # Disease class name
control_class = 'cn'                    # Control class name

## gpt-2 used parameters

max_epochs = 10                         # Maximum epochs
model_name = 'gpt2'                     # Model to use
batch_size = 12                         # Batch size

## llama used parameters

max_epochs = 10                         # Maximum epochs
model_name = 'llama-lora'               # Model to use
batch_size = 2                          # Batch size
```

Then run:

```bash
cd src
python -m run_experiment.py
```

## Experiment Phases

1. **Training Models**: Training models for CN and AD classes
2. **Evaluation**: Perplexity calculation on dev set
3. **Model Selection**: Selection of epoch with minimum perplexity per class over dev
4. **Perplexity Production**: Training of leave one out models until best epoch and PPL calculation using chosen best epoch models.
4. **Classification**: Classification of test patients and calculation of metrics

## Output

- **Submission Models**: `resources/data/output/submission_models/`
- **Training Output Models**: `resources/data/output/models/`
- **Perplexity**: `resources/data/output_gpt2/` and `resources/data/output_llama_lora/`
