import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import os

# Percorso al modello salvato (modifica con un checkpoint esistente)
model_path = "/archive/home/mdelsant/PEARL/resources/data/output/submission_models/adresso_final_fold/gpt2_/cn_whisper-large-v3-turbo_12b_2ep"

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

# Tokenizza il testo
encodings = tokenizer(testo, return_tensors="pt")
input_ids = encodings.input_ids.to(model.device)

# Calcola la perplexity
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    perplexity = math.exp(loss.item())
    
    # Calcola la surprisal di ogni token
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    log_probs = torch.log_softmax(logits, dim=-1)  # log probabilità
    
    # Estrai la log-prob del token successivo (shift di 1)
    # logits[i] predice il token input_ids[i+1]
    shift_logits = log_probs[:, :-1, :]  # [batch, seq_len-1, vocab_size]
    shift_labels = input_ids[:, 1:]  # [batch, seq_len-1]
    
    # Estrai la log-prob per ogni token target
    token_log_probs = shift_logits.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_surprisal = -token_log_probs  # surprisal = -log(p)

print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity:.4f}")
print("Surprisal per token:")
# Il primo token non ha surprisal (non c'è predizione per esso)
print(f"Token: {tokenizer.decode(input_ids[0][0])}\tSurprisal: N/A (primo token)")
for token_id, surprisal_value in zip(input_ids[0][1:], token_surprisal[0]):
    token = tokenizer.decode(token_id)
    print(f"Token: {token}\tSurprisal: {surprisal_value.item():.4f}")
