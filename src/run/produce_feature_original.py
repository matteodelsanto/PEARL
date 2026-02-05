import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Aggiungi il path per importare i moduli utils
sys.path.append("/archive/home/mdelsant/PEARL/src")

from utils.perplexity import compute_global_perplexity
from utils.surprisal import compute_token_surprisal_with_tokens
from utils.feature_extractor import (
    build_feature_embedding,
    compute_delta_features,
    fit_histogram_range_from_training,
    HistogramConfig,
)

# Percorsi ai modelli
cn_model_path = "/archive/home/mdelsant/PEARL/resources/data/output/submission_models/adresso_final_fold/gpt2_/cn_whisper-large-v3-turbo_12b_10ep"
ad_model_path = "/archive/home/mdelsant/PEARL/resources/data/output/submission_models/adresso_final_fold/gpt2_/cn_whisper-large-v3-turbo_12b_6ep"

# Percorsi dati
base_input_path = "/archive/home/mdelsant/PEARL/resources/data/input/adresso_final_fold/train"
output_dir = "/archive/home/mdelsant/PEARL/resources/data/output/feature_dataset"
surprisal_cache_dir = "/archive/home/mdelsant/PEARL/resources/data/output/surprisal_values"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(surprisal_cache_dir, exist_ok=True)

def get_model_identifier(model_path: str) -> str:
    """
    Estrae un identificatore univoco dal path del modello.
    Usa il path da 'output' in avanti.
    Es: /archive/.../output/submission_models/adresso_final_fold/gpt2_/cn_whisper-large-v3-turbo_12b_10ep
    -> submission_models/adresso_final_fold/gpt2_/cn_whisper-large-v3-turbo_12b_10ep
    """
    # Trova la posizione di "output" nel path
    path_obj = Path(model_path)
    parts = path_obj.parts
    
    try:
        output_idx = parts.index("output")
        # Prendi tutto da "output" in avanti, escluso "output" stesso
        identifier = str(Path(*parts[output_idx + 1:]))
        return identifier
    except ValueError:
        # Se "output" non è nel path, usa il nome della directory
        return path_obj.name

def get_surprisal_cache_path(
    subject_id: str,
    disease_type: str,
    model_identifier: str,
    base_path: str = None
) -> Path:
    """
    Ritorna il path dove salvare/caricare i surprisal per un paziente.
    Struttura: {surprisal_cache_dir}/{disease_type}/{subject_id}/{model_identifier_safe}.json
    Il model_identifier viene convertito a un nome file sicuro (slashes -> underscores).
    Crea tutte le cartelle necessarie se non esistono.
    """
    if base_path is None:
        base_path = surprisal_cache_dir
    
    # Costruisci il path della cartella del paziente
    cache_dir = Path(base_path) / disease_type / subject_id
    
    # Crea la cartella se non esiste
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Errore nella creazione della cartella cache {cache_dir}: {e}")
    
    # Converti il model_identifier in un nome file sicuro (sostituisci "/" con "_")
    safe_model_name = model_identifier.replace("/", "_").replace("\\", "_")
    
    # Il path del file json
    json_path = cache_dir / f"{safe_model_name}.json"
    
    return json_path

def save_surprisal_to_cache(
    surprisal_values: List[float],
    tokens: List[str],
    subject_id: str,
    disease_type: str,
    model_identifier: str
) -> bool:
    """
    Salva i valori di surprisal in cache.
    """
    try:
        cache_path = get_surprisal_cache_path(subject_id, disease_type, model_identifier)
        
        data = {
            'tokens': tokens,
            'surprisal_values': surprisal_values
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Errore nel salvataggio cache per {subject_id}: {e}")
        return False

def load_surprisal_from_cache(
    subject_id: str,
    disease_type: str,
    model_identifier: str
) -> Tuple[List[str], List[float]]:
    """
    Carica i valori di surprisal dalla cache se esistono.
    Ritorna (tokens, surprisal_values) o ([], []) se non trovato.
    """
    try:
        cache_path = get_surprisal_cache_path(subject_id, disease_type, model_identifier)
        
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['tokens'], data['surprisal_values']
    except Exception as e:
        print(f"Errore nel caricamento cache per {subject_id}: {e}")
    
    return [], []

def load_model_and_tokenizer(model_path: str):
    """Carica modello e tokenizer."""
    print(f"Caricamento modello da: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def find_text_files(disease_type: str) -> List[Tuple[str, str]]:
    """
    Trova tutti i test_text.txt nella struttura:
    base_input_path/{disease_type}/whisper-large-v3-turbo/{subject_id}/test_text.txt
    
    Ritorna lista di (subject_id, file_path)
    """
    disease_path = Path(base_input_path) / disease_type / "whisper-large-v3-turbo"
    results = []
    
    if disease_path.exists():
        for subject_dir in disease_path.iterdir():
            if subject_dir.is_dir():
                text_file = subject_dir / "test_text.txt"
                if text_file.exists():
                    results.append((subject_dir.name, str(text_file)))
    
    return sorted(results)

def read_text_file(file_path: str) -> str:
    """Legge il contenuto di un file di testo."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def compute_surprisal_once(
    text: str,
    model,
    tokenizer,
    subject_id: str = None,
    disease_type: str = None,
    model_path: str = None,
    use_cache: bool = True
) -> Tuple[List[str], List[float]]:
    """
    Calcola surprisal una sola volta per un testo.
    Se use_cache=True e subject_id/disease_type/model_path sono forniti,
    controlla la cache prima di calcolare.
    Ritorna (tokens, surprisal_values).
    """
    # Prova a caricare dalla cache
    if use_cache and subject_id and disease_type and model_path:
        model_identifier = get_model_identifier(model_path)
        cached_tokens, cached_surprisal = load_surprisal_from_cache(
            subject_id, disease_type, model_identifier
        )
        if cached_surprisal:
            return cached_tokens, cached_surprisal
    
    # Calcola surprisal
    try:
        tokens, surprisal_values = compute_token_surprisal_with_tokens(
            text, model, tokenizer, device=model.device
        )
        
        # Salva in cache se possibile
        if use_cache and subject_id and disease_type and model_path:
            model_identifier = get_model_identifier(model_path)
            save_surprisal_to_cache(
                surprisal_values, tokens, subject_id, disease_type, model_identifier
            )
        
        return tokens, surprisal_values
    except Exception as e:
        print(f"Errore nel calcolo surprisal: {e}")
        return [], []

def extract_features_for_text(
    surprisal_values: List[float],
    prefix: str,
    hist_range: Tuple[float, float]
) -> Dict[str, float]:
    """
    Estrae le feature da surprisal già calcolati.
    """
    try:
        if not surprisal_values:
            return {}
        
        # Costruisci embedding delle feature
        feature_embedding = build_feature_embedding(
            surprisal_values,
            prefix=prefix,
            do_summary=True,
            do_windows=True,
            n_windows=5,
            do_hist=True,
            hist_bins=20,
            hist_range=hist_range
        )
        
        return feature_embedding
    except Exception as e:
        print(f"Errore nell'estrazione delle feature: {e}")
        return {}

def compute_delta_features_wrapper(
    cn_surprisal: List[float],
    ad_surprisal: List[float],
    cn_hist_range: Tuple[float, float],
    ad_hist_range: Tuple[float, float]
) -> Dict[str, float]:
    """
    Wrapper per compute_delta_features che calcola il range delta
    e chiama la funzione di feature_extractor.
    """
    if not cn_surprisal or not ad_surprisal:
        return {}
    
    if len(cn_surprisal) != len(ad_surprisal):
        print(f"⚠️  Lunghezze surprisal diverse: CN={len(cn_surprisal)}, AD={len(ad_surprisal)}")
        return {}
    
    # Calcola l'intervallo per l'istogramma della delta
    delta_lo = ad_hist_range[0] - cn_hist_range[1]
    delta_hi = ad_hist_range[1] - cn_hist_range[0]
    delta_hist_range = (delta_lo, delta_hi)
    
    # Delega a feature_extractor
    return compute_delta_features(
        cn_surprisal, ad_surprisal,
        prefix="D",
        hist_range=delta_hist_range,
        do_summary=True,
        do_windows=True,
        n_windows=5,
        do_hist=True,
        hist_bins=20
    )

def compute_histogram_range_from_texts(
    text_files: List[str],
    model,
    tokenizer,
    subject_ids: List[str] = None,
    disease_type: str = None,
    model_path: str = None,
    use_cache: bool = True
) -> Tuple[float, float]:
    """
    Calcola l'intervallo istogramma dai dati di training.
    """
    all_surprisals = []
    
    print(model.device)
    
    for i, text_file in enumerate(tqdm(text_files, desc="Calcolo range istogramma", unit="file")):
        try:
            text = read_text_file(text_file)
            subject_id = subject_ids[i] if subject_ids else None
            
            _, surprisal_values = compute_surprisal_once(
                text, model, tokenizer,
                subject_id=subject_id,
                disease_type=disease_type,
                model_path=model_path,
                use_cache=use_cache
            )
            if surprisal_values:
                all_surprisals.extend(surprisal_values)
        except Exception as e:
            print(f"Errore nel calcolo range per {text_file}: {e}")
            continue
    
    if not all_surprisals:
        return (0.0, 1.0)
    
    all_surprisals = sorted(all_surprisals)
    lo = all_surprisals[int(len(all_surprisals) * 0.01)]  # 1° percentile
    hi = all_surprisals[int(len(all_surprisals) * 0.99)]  # 99° percentile
    
    if hi <= lo:
        hi = lo + 1.0
    
    return (lo, hi)

def process_train():
    """Elabora i dati di training (AD e CN separati)."""
    print("ESTRAZIONE FEATURE DAI TESTI DI TRAINING")
    print("========================================\n")
    
    # Carica entrambi i modelli
    print("Caricamento modelli...")
    cn_model, cn_tokenizer = load_model_and_tokenizer(cn_model_path)
    ad_model, ad_tokenizer = load_model_and_tokenizer(ad_model_path)
    
    # Raccoglie tutti i file e calcola range istogramma per entrambi i modelli
    print("\nRaccolta file di testo...")
    all_ad_files = find_text_files("ad")
    all_cn_files = find_text_files("cn")
    
    all_files = all_ad_files + all_cn_files
    ad_file_paths = [fp for _, fp in all_ad_files]
    cn_file_paths = [fp for _, fp in all_cn_files]
    ad_subject_ids = [sid for sid, _ in all_ad_files]
    cn_subject_ids = [sid for sid, _ in all_cn_files]
    
    print(f"Trovati {len(all_ad_files)} pazienti AD")
    print(f"Trovati {len(all_cn_files)} pazienti CN")
    
    # Calcola intervalli istogramma per entrambi i modelli
    print("\nCalcolo intervalli istogramma...")
    print("  Per modello CN...", end=" ")
    cn_hist_range = compute_histogram_range_from_texts(
        cn_file_paths, cn_model, cn_tokenizer,
        subject_ids=cn_subject_ids, disease_type="cn", model_path=cn_model_path
    )
    print(f"range: {cn_hist_range}")
    
    print("  Per modello AD...", end=" ")
    ad_hist_range = compute_histogram_range_from_texts(
        ad_file_paths, ad_model, ad_tokenizer,
        subject_ids=ad_subject_ids, disease_type="ad", model_path=ad_model_path
    )
    print(f"range: {ad_hist_range}")
    
    # Elabora ogni paziente con entrambi i modelli
    all_features = []
    total_patients = len(all_ad_files) + len(all_cn_files)
    
    print(f"\nElaborazione {total_patients} pazienti totali...")
    
    with tqdm(total=total_patients, desc="Estrazione feature TRAIN", unit="paziente") as pbar:
        for disease_type, files_list in [("ad", all_ad_files), ("cn", all_cn_files)]:
            for subject_id, text_file in files_list:
                pbar.set_postfix({"paziente": subject_id, "tipo": disease_type.upper()})
                
                try:
                    text = read_text_file(text_file)
                    
                    # Calcola surprisal UNA VOLTA (con cache)
                    _, cn_surprisal = compute_surprisal_once(
                        text, cn_model, cn_tokenizer,
                        subject_id=subject_id, disease_type=disease_type, model_path=cn_model_path
                    )
                    _, ad_surprisal = compute_surprisal_once(
                        text, ad_model, ad_tokenizer,
                        subject_id=subject_id, disease_type=disease_type, model_path=ad_model_path
                    )
                    
                    if cn_surprisal and ad_surprisal:
                        # Estrai feature dai surprisal già calcolati
                        cn_features = extract_features_for_text(
                            cn_surprisal,
                            prefix="S_CN",
                            hist_range=cn_hist_range
                        )
                        
                        ad_features = extract_features_for_text(
                            ad_surprisal,
                            prefix="S_AD",
                            hist_range=ad_hist_range
                        )
                        
                        if cn_features and ad_features:
                            # Calcola feature DELTA usando la wrapper
                            delta_features = compute_delta_features_wrapper(
                                cn_surprisal, ad_surprisal,
                                cn_hist_range, ad_hist_range
                            )
                            
                            # DEBUG: Controlla se delta_features è vuoto
                            if not delta_features:
                                pbar.write(f"⚠️  AVVISO: {subject_id} - delta_features VUOTO!")
                                pbar.write(f"   CN surprisal: {len(cn_surprisal)} valori")
                                pbar.write(f"   AD surprisal: {len(ad_surprisal)} valori")
                            
                            # Unisci le feature da entrambi i modelli + DELTA in una riga unica
                            row = {
                                'subject_id': subject_id,
                                'disease': disease_type.upper(),
                                **cn_features,
                                **ad_features,
                                **delta_features
                            }
                            all_features.append(row)
                
                except Exception as e:
                    pbar.write(f"ERRORE per {subject_id}: {e}")
                
                pbar.update(1)
    
    # Crea DataFrame e salva CSV unico
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Riordina colonne: subject_id e disease all'inizio
        cols = ['subject_id', 'disease'] + [c for c in df.columns if c not in ['subject_id', 'disease']]
        df = df[cols]
        
        output_file = os.path.join(output_dir, "features_train.csv")
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"✓ TRAIN Salvato: {output_file}")
        print(f"✓ Shape: {df.shape} (righe, colonne)")
        print(f"✓ Cache surprisal: {surprisal_cache_dir}")
        # DEBUG: Conta feature DELTA
        delta_cols = [c for c in df.columns if c.startswith('D_')]
        print(f"✓ Feature DELTA trovate: {len(delta_cols)} ({', '.join(delta_cols[:3])}...)")
        print(f"{'='*60}\n")
    else:
        print("\nNessuna feature estratta per TRAIN!")

def process_dev():
    """Elabora i dati di dev (AD e CN separati, stessa struttura di train)."""
    print("ESTRAZIONE FEATURE DAI TESTI DI DEV")
    print("===================================\n")
    
    # Carica entrambi i modelli
    print("Caricamento modelli...")
    cn_model, cn_tokenizer = load_model_and_tokenizer(cn_model_path)
    ad_model, ad_tokenizer = load_model_and_tokenizer(ad_model_path)
    
    # Usa la cartella dev invece di train
    base_dev_path = "/archive/home/mdelsant/PEARL/resources/data/input/adresso_final_fold/dev"
    
    # Raccoglie tutti i file
    print("\nRaccolta file di testo...")
    def find_dev_text_files(disease_type: str) -> List[Tuple[str, str]]:
        disease_path = Path(base_dev_path) / disease_type / "whisper-large-v3-turbo"
        results = []
        if disease_path.exists():
            for subject_dir in disease_path.iterdir():
                if subject_dir.is_dir():
                    text_file = subject_dir / "test_text.txt"
                    if text_file.exists():
                        results.append((subject_dir.name, str(text_file)))
        return sorted(results)
    
    all_ad_files = find_dev_text_files("ad")
    all_cn_files = find_dev_text_files("cn")
    
    ad_file_paths = [fp for _, fp in all_ad_files]
    cn_file_paths = [fp for _, fp in all_cn_files]
    ad_subject_ids = [sid for sid, _ in all_ad_files]
    cn_subject_ids = [sid for sid, _ in all_cn_files]
    
    print(f"Trovati {len(all_ad_files)} pazienti AD")
    print(f"Trovati {len(all_cn_files)} pazienti CN")
    
    if not all_ad_files and not all_cn_files:
        print("Nessun file trovato per DEV!")
        return
    
    # Calcola intervalli istogramma
    print("\nCalcolo intervalli istogramma...")
    print("  Per modello CN...", end=" ")
    cn_hist_range = compute_histogram_range_from_texts(
        cn_file_paths, cn_model, cn_tokenizer,
        subject_ids=cn_subject_ids, disease_type="cn", model_path=cn_model_path
    )
    print(f"range: {cn_hist_range}")
    
    print("  Per modello AD...", end=" ")
    ad_hist_range = compute_histogram_range_from_texts(
        ad_file_paths, ad_model, ad_tokenizer,
        subject_ids=ad_subject_ids, disease_type="ad", model_path=ad_model_path
    )
    print(f"range: {ad_hist_range}")
    
    # Elabora ogni paziente
    all_features = []
    total_patients = len(all_ad_files) + len(all_cn_files)
    
    print(f"\nElaborazione {total_patients} pazienti totali...")
    
    with tqdm(total=total_patients, desc="Estrazione feature DEV", unit="paziente") as pbar:
        for disease_type, files_list in [("ad", all_ad_files), ("cn", all_cn_files)]:
            for subject_id, text_file in files_list:
                pbar.set_postfix({"paziente": subject_id, "tipo": disease_type.upper()})
                
                try:
                    text = read_text_file(text_file)
                    
                    # Calcola surprisal UNA VOLTA (con cache)
                    _, cn_surprisal = compute_surprisal_once(
                        text, cn_model, cn_tokenizer,
                        subject_id=subject_id, disease_type=disease_type, model_path=cn_model_path
                    )
                    _, ad_surprisal = compute_surprisal_once(
                        text, ad_model, ad_tokenizer,
                        subject_id=subject_id, disease_type=disease_type, model_path=ad_model_path
                    )
                    
                    if cn_surprisal and ad_surprisal:
                        cn_features = extract_features_for_text(
                            cn_surprisal,
                            prefix="S_CN",
                            hist_range=cn_hist_range
                        )
                        
                        ad_features = extract_features_for_text(
                            ad_surprisal,
                            prefix="S_AD",
                            hist_range=ad_hist_range
                        )
                        
                        if cn_features and ad_features:
                            # Calcola feature DELTA usando la wrapper
                            delta_features = compute_delta_features_wrapper(
                                cn_surprisal, ad_surprisal,
                                cn_hist_range, ad_hist_range
                            )
                            
                            # DEBUG: Controlla se delta_features è vuoto
                            if not delta_features:
                                pbar.write(f"⚠️  AVVISO: {subject_id} - delta_features VUOTO!")
                                pbar.write(f"   CN surprisal: {len(cn_surprisal)} valori")
                                pbar.write(f"   AD surprisal: {len(ad_surprisal)} valori")
                            
                            row = {
                                'subject_id': subject_id,
                                'disease': disease_type.upper(),
                                **cn_features,
                                **ad_features,
                                **delta_features
                            }
                            all_features.append(row)
                
                except Exception as e:
                    pbar.write(f"ERRORE per {subject_id}: {e}")
                
                pbar.update(1)
    
    # Salva CSV
    if all_features:
        df = pd.DataFrame(all_features)
        cols = ['subject_id', 'disease'] + [c for c in df.columns if c not in ['subject_id', 'disease']]
        df = df[cols]
        
        output_file = os.path.join(output_dir, "features_dev.csv")
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"✓ DEV Salvato: {output_file}")
        print(f"✓ Shape: {df.shape} (righe, colonne)")
        print(f"✓ Cache surprisal: {surprisal_cache_dir}")
        # DEBUG: Conta feature DELTA
        delta_cols = [c for c in df.columns if c.startswith('D_')]
        print(f"✓ Feature DELTA trovate: {len(delta_cols)} ({', '.join(delta_cols[:3])}...)")
        print(f"{'='*60}\n")
    else:
        print("\nNessuna feature estratta per DEV!")

def process_test():
    """Elabora i dati di test (con labels da labels.csv)."""
    print("ESTRAZIONE FEATURE DAI TESTI DI TEST")
    print("====================================\n")
    
    # Carica entrambi i modelli
    print("Caricamento modelli...")
    cn_model, cn_tokenizer = load_model_and_tokenizer(cn_model_path)
    ad_model, ad_tokenizer = load_model_and_tokenizer(ad_model_path)
    
    base_test_path = "/archive/home/mdelsant/PEARL/resources/data/input/adresso_final_fold/test/whisper-large-v3-turbo"
    labels_file = os.path.join("/archive/home/mdelsant/PEARL/resources/data/input/adresso_final_fold/test/", "labels.csv")
    
    # Leggi il file labels.csv
    print("Caricamento labels...")
    if not os.path.exists(labels_file):
        print(f"File labels non trovato: {labels_file}")
        return
    
    labels_df = pd.read_csv(labels_file, sep=';')
    print(f"Trovati {len(labels_df)} pazienti nel file labels")
    
    # Raccogli file di testo
    print("\nRaccolta file di testo...")
    test_files = []
    test_subject_ids = []
    for _, row in labels_df.iterrows():
        patient_id = row['patient_id']
        label = row['label']
        text_file = Path(base_test_path) / patient_id / "test_text.txt"
        if text_file.exists():
            test_files.append((patient_id, str(text_file), label))
            test_subject_ids.append(patient_id)
    
    print(f"Trovati {len(test_files)} file di testo corrispondenti")
    
    if not test_files:
        print("Nessun file trovato per TEST!")
        return
    
    # Estrai path per calcolare range istogramma
    test_file_paths = [fp for _, fp, _ in test_files]
    
    # Calcola intervalli istogramma
    print("\nCalcolo intervalli istogramma...")
    print("  Per modello CN...", end=" ")
    cn_hist_range = compute_histogram_range_from_texts(
        test_file_paths, cn_model, cn_tokenizer,
        subject_ids=test_subject_ids, disease_type="test", model_path=cn_model_path
    )
    print(f"range: {cn_hist_range}")
    
    print("  Per modello AD...", end=" ")
    ad_hist_range = compute_histogram_range_from_texts(
        test_file_paths, ad_model, ad_tokenizer,
        subject_ids=test_subject_ids, disease_type="test", model_path=ad_model_path
    )
    print(f"range: {ad_hist_range}")
    
    # Elabora ogni paziente
    all_features = []
    
    print(f"\nElaborazione {len(test_files)} pazienti totali...")
    
    with tqdm(total=len(test_files), desc="Estrazione feature TEST", unit="paziente") as pbar:
        for patient_id, text_file, label in test_files:
            pbar.set_postfix({"paziente": patient_id, "label": label})
            
            try:
                text = read_text_file(text_file)
                
                # Calcola surprisal UNA VOLTA (con cache)
                _, cn_surprisal = compute_surprisal_once(
                    text, cn_model, cn_tokenizer,
                    subject_id=patient_id, disease_type="test", model_path=cn_model_path
                )
                _, ad_surprisal = compute_surprisal_once(
                    text, ad_model, ad_tokenizer,
                    subject_id=patient_id, disease_type="test", model_path=ad_model_path
                )
                
                if cn_surprisal and ad_surprisal:
                    cn_features = extract_features_for_text(
                        cn_surprisal,
                        prefix="S_CN",
                        hist_range=cn_hist_range
                    )
                    
                    ad_features = extract_features_for_text(
                        ad_surprisal,
                        prefix="S_AD",
                        hist_range=ad_hist_range
                    )
                    
                    if cn_features and ad_features:
                        # Calcola feature DELTA usando la wrapper
                        delta_features = compute_delta_features_wrapper(
                            cn_surprisal, ad_surprisal,
                            cn_hist_range, ad_hist_range
                        )
                        
                        # DEBUG: Controlla se delta_features è vuoto
                        if not delta_features:
                            pbar.write(f"⚠️  AVVISO: {patient_id} - delta_features VUOTO!")
                            pbar.write(f"   CN surprisal: {len(cn_surprisal)} valori")
                            pbar.write(f"   AD surprisal: {len(ad_surprisal)} valori")
                        
                        row = {
                            'patient_id': patient_id,
                            'label': label,
                            **cn_features,
                            **ad_features,
                            **delta_features
                        }
                        all_features.append(row)
            
            except Exception as e:
                pbar.write(f"ERRORE per {patient_id}: {e}")
            
            pbar.update(1)
    
    # Salva CSV
    if all_features:
        df = pd.DataFrame(all_features)
        cols = ['patient_id', 'label'] + [c for c in df.columns if c not in ['patient_id', 'label']]
        df = df[cols]
        
        output_file = os.path.join(output_dir, "features_test.csv")
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"✓ TEST Salvato: {output_file}")
        print(f"✓ Shape: {df.shape} (righe, colonne)")
        print(f"✓ Cache surprisal: {surprisal_cache_dir}")
        # DEBUG: Conta feature DELTA
        delta_cols = [c for c in df.columns if c.startswith('D_')]
        print(f"✓ Feature DELTA trovate: {len(delta_cols)} ({', '.join(delta_cols[:3])}...)")
        print(f"{'='*60}\n")
    else:
        print("\nNessuna feature estratta per TEST!")

def main():
    print("="*60)
    print("ESTRAZIONE FEATURE COMPLETE")
    print("="*60 + "\n")
    
    # Elabora TRAIN
    process_train()
    
    # Elabora DEV
    process_dev()
    
    # Elabora TEST
    process_test()
    
    print("\n" + "="*60)
    print("TUTTE LE ELABORAZIONI COMPLETATE!")
    print(f"File salvati in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

