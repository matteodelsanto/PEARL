import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import os
import sys

def compute_global_perplexity(text, model, tokenizer, window_size=20, device="cpu", leap=0):
    encodings = tokenizer(text, return_tensors='pt').to(device)

    max_length = window_size if window_size > 0 else model.config.n_positions
    stride = 1
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    ppl = torch.exp(torch.stack(nlls[leap:]).sum() / (end_loc-leap))
    
    # Sposta il tensore su CPU prima di convertirlo in numpy
    return ppl.cpu().numpy().tolist()

# Percorso al modello salvato (modifica con un checkpoint esistente)
model_path = "/home/resources/modelli_adresso/ad_whisper-large-v3-turbo_12b_6ep"

# Carica il tokenizer e il modello
print(f"Caricamento modello da: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Imposta il pad token se non esiste
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Carica il modello
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"  # Gestisce automaticamente il device
)

model.eval()

# Testo di esempio
testo = "This is a sample text to evaluate the language model's perplexity and surprisal values."

print(f"\nTesto di test: {testo}\n")

# Calcola la perplexity

ppl = compute_global_perplexity(testo, model, tokenizer)
print(f"Perplexity: {ppl}")



