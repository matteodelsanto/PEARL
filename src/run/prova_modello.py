import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import os
import sys

# Aggiungi il path per importare i moduli utils
sys.path.append("/archive/home/mdelsant/PEARL/src")

from utils.perplexity import compute_global_perplexity
from utils.surprisal import compute_token_surprisal, compute_token_surprisal_with_tokens
from utils.feature_extractor import build_feature_embedding, fit_histogram_range_from_training

# Percorso al modello salvato (modifica con un checkpoint esistente)
model_path = "/archive/home/mdelsant/PEARL/resources/data/output/submission_models/adresso_final_fold/gpt2_/cn_whisper-large-v3-turbo_12b_2ep"

# Carica il modello
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"  # Gestisce automaticamente il device
)
# Carica il tokenizer e il modello
print(f"Caricamento modello da: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Imposta il pad token se non esiste
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

# Testo di esempio
testo = "This is a sample text to evaluate the language dog model's perplexity and surprisal values."
testo = "This is a text to evaluatethe language elephant model's perplexity and surprisal values making it diffiult to predict the next words accurately for example using uncommon phrases and structures that the model might not have seen during training."

print(f"\nTesto di test: {testo}\n")

# Tokenizza il testo
encodings = tokenizer(testo, return_tensors="pt")
input_ids = encodings.input_ids.to(model.device)

# Calcola la perplexity

ppl = compute_global_perplexity(testo, model, tokenizer, device=model.device)
print(f"Perplexity: {ppl}")

# Calcola la surprisal per ogni token

tokens, surprisal_values = compute_token_surprisal_with_tokens(testo, model, tokenizer, device=model.device)

print("\nSurprisal per token:")
for token, surp in zip(tokens, surprisal_values):
    print(f"  {token}: {surp}")

# Estrazione delle feature dai valori di surprisal
print("\n" + "="*50)
print("Estrazione Feature dai Surprisal Values")
print("="*50)

# Definisci l'intervallo istogramma (in alternativa, calcola da dati di training)
hist_range = (min(surprisal_values) if surprisal_values else 0.0, 
              max(surprisal_values) if surprisal_values else 1.0)

# Costruisci l'embedding delle feature
feature_embedding = build_feature_embedding(
    surprisal_values,
    prefix="S_surprisal",
    do_summary=True,
    do_windows=True,
    n_windows=5,
    do_hist=True,
    hist_bins=20,
    hist_range=hist_range
)

print("\nFeature estratte:")
print("-" * 50)
for key in sorted(feature_embedding.keys()):
    print(f"{key}: {feature_embedding[key]:.6f}")

