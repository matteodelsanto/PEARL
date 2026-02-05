import torch


def compute_token_surprisal(text, model, tokenizer, window_size=20, device="cpu"):
    encodings = tokenizer(text, return_tensors='pt').to(device)

    max_length = window_size if window_size > 0 else model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    surprisals = []
    
    for i in range(1, seq_len):
        # Prendi il contesto fino al token corrente (massimo window_size)
        begin_loc = max(0, i - max_length + 1)
        end_loc = i + 1
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        
        # Creiamo i target: vogliamo predire solo l'ultimo token
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100  # Ignora tutti i token tranne l'ultimo

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # La loss è già la negative log-likelihood media sui token non mascherati
            # In questo caso, solo l'ultimo token, quindi è la surprisal
            surprisal = outputs.loss.item()

        surprisals.append(surprisal)

    return surprisals


def compute_token_surprisal_with_tokens(text, model, tokenizer, window_size=20, device="cpu"):
    """Restituisce sia i token che i loro valori di surprisal."""
    encodings = tokenizer(text, return_tensors='pt').to(device)

    max_length = window_size if window_size > 0 else model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    surprisals = []
    tokens = []
    
    for i in range(1, seq_len):
        begin_loc = max(0, i - max_length + 1)
        end_loc = i + 1
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            surprisal = outputs.loss.item()

        surprisals.append(surprisal)
        tokens.append(tokenizer.decode(encodings.input_ids[0, i]))

    return tokens, surprisals


def compute_token_surprisal_and_entropy(text, model, tokenizer, window_size=20, device="cpu"):
    """Calcola l'entropia (surprisal media cumulativa) fino all'n-esimo token."""
    surprisals = compute_token_surprisal(text, model, tokenizer, window_size, device)
    
    entropies = []
    cumulative_sum = 0.0
    
    for i, surprisal in enumerate(surprisals, start=1):
        cumulative_sum += surprisal
        entropy = cumulative_sum / i
        entropies.append(entropy)
    
    return entropies, surprisals


def compute_token_surprisal_and_entropy_with_tokens(text, model, tokenizer, window_size=20, device="cpu"):
    """Restituisce sia i token che i loro valori di entropia (surprisal media cumulativa)."""
    tokens, surprisals = compute_token_surprisal_with_tokens(text, model, tokenizer, window_size, device)
    
    entropies = []
    cumulative_sum = 0.0
    
    for i, surprisal in enumerate(surprisals, start=1):
        cumulative_sum += surprisal
        entropy = cumulative_sum / i
        entropies.append(entropy)
    
    return tokens, entropies, surprisals
