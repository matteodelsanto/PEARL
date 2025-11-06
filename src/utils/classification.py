from numpy import mean, std

from tqdm import tqdm

import statistics
import os
import glob
import re

def classify_patient(subject_controlmodel_ppl, subject_diseasemodel_ppl, control_controlmodel_ppls, control_diseasemodel_ppls, disease_controlmodel_ppls, disease_diseasemodel_ppls, two_sigma=False, use_median=False):
    
    control_scores = []
    for i in range(0, len(control_controlmodel_ppls)):
        control_scores.append(float(control_diseasemodel_ppls[i]) - float(control_controlmodel_ppls[i]))
    
    disease_scores = []
    for i in range(0, len(disease_diseasemodel_ppls)):
        disease_scores.append(float(disease_diseasemodel_ppls[i]) - float(disease_controlmodel_ppls[i]))  
    
    if use_median:
        control_threshold = statistics.median(control_scores)
        disease_threshold = statistics.median(disease_scores)
        
        control_twosigma_threshold = statistics.median(control_scores) - 2 * std(control_scores)
        disease_twosigma_threshold = statistics.median(disease_scores) - 2 * std(disease_scores)
    else:
        control_threshold = mean(control_scores)
        disease_threshold = mean(disease_scores)
        
        control_twosigma_threshold = mean(control_scores) - 2 * std(control_scores)
        disease_twosigma_threshold = mean(disease_scores) - 2 * std(disease_scores)
    
    subject_score = float(subject_diseasemodel_ppl) - float(subject_controlmodel_ppl)
    
    print("Subject Score")
    print(subject_score)
    print("##############################################")
    print("Control Threshold")
    print(control_threshold)
    print("Disease Threshold")
    print(disease_threshold)
    print("##############################################")
    
    if two_sigma:
        if abs(subject_score - control_twosigma_threshold) < abs(subject_score - disease_twosigma_threshold):
            return "Control"
        else:
            return "Non Healthy"
    else:
        if abs(subject_score - control_threshold) < abs(subject_score - disease_threshold):
            return "Control"
        else:
            return "Non Healthy"

def collect_perplexity_values(base_dir, model_name, llo=False, multiple_files=False):
    """
    Raccoglie i valori di perplexity dai file di output.
    
    Args:
        base_dir: Directory di base (es. "../resources/data/output/anchise/")
        model_name: Nome del modello (es. "modello_anchise" o "modello_nestor")
    
    Returns:
        Dictionary con i valori di perplexity per ogni paziente
    """
    values = {}
    
    # Trova tutte le directory dei pazienti
    patient_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    
    
    for patient in patient_dirs:
        patient_path = os.path.join(base_dir, patient)
        
        
        # Cerca file con il pattern *_modello_{model_name}_global_ppl_score.txt
        if llo:
            pattern = f"{patient_path}/*_modello_{model_name}_leave_one_out_ppl_score.txt"
        else:
            pattern = f"{patient_path}/*_modello_{model_name}_global_ppl_score*.txt"
        matching_files = glob.glob(pattern)
        
        if matching_files:
            
            if not multiple_files:
                # Prendi il primo file che corrisponde al pattern
                file_path = matching_files[0]
                
                try:
                    with open(file_path, 'r') as f:
                        # Leggi il valore di perplexity (dovrebbe essere l'unico valore nel file)
                        perplexity = f.read().strip()
                        values[patient] = perplexity
                except Exception as e:
                    print(f"Errore nella lettura del file {file_path}: {str(e)}")
            if multiple_files:
                values[patient] = []
                for file_path in matching_files:
                    try:
                        with open(file_path, 'r') as f:
                            # Leggi il valore di perplexity (dovrebbe essere l'unico valore nel file)
                            perplexity = f.read().strip()
                            values[patient].append(perplexity)
                    except Exception as e:
                        print(f"Errore nella lettura del file {file_path}: {str(e)}")
                
        else:
            print(f"Nessun file trovato per il paziente {patient} con il modello {model_name}")
    
    return values

def classify_all_patients(test_dir, control_dir, disease_dir, control_model, disease_model, use_median=False, two_sigma=False, output_csv=None):
    """
    Classifica i pazienti nella cartella specificata utilizzando i dati di riferimento.
    
    Args:
        test_dir: Directory contenente i pazienti da classificare
        control_dir: Directory contenente i pazienti di controllo
        disease_dir: Directory contenente i pazienti malati
        control_model: Nome del modello di controllo
        disease_model: Nome del modello di malattia
        use_median: Se True, usa la mediana invece della media
        two_sigma: Se True, usa la soglia a due sigma
        output_csv: Nome del file CSV dove salvare i risultati (opzionale)
    """
    # Raccogli i valori di perplexity per i dati di riferimento
    control_controlmodel = collect_perplexity_values(control_dir, control_model, llo=True)
    control_diseasemodel = collect_perplexity_values(control_dir, disease_model, llo=False)
    disease_controlmodel = collect_perplexity_values(disease_dir, control_model, llo=False)
    disease_diseasemodel = collect_perplexity_values(disease_dir, disease_model, llo=True)
    
    # Crea le liste per la classificazione
    control_controlmodel_ppls = list(control_controlmodel.values())
    control_diseasemodel_ppls = list(control_diseasemodel.values())
    disease_controlmodel_ppls = list(disease_controlmodel.values())
    disease_diseasemodel_ppls = list(disease_diseasemodel.values())
    
    print(f"Dati di riferimento caricati:")
    print(f"- Pazienti di controllo ({os.path.basename(control_dir)}): {len(control_controlmodel_ppls)}")
    print(f"- Pazienti malati ({os.path.basename(disease_dir)}): {len(disease_diseasemodel_ppls)}")
    
    # Raccogli i valori di perplexity per i pazienti DA CLASSIFICARE
    test_controlmodel = collect_perplexity_values(test_dir, control_model)
    test_diseasemodel = collect_perplexity_values(test_dir, disease_model)
    
    # Contatori per i risultati
    count_control = 0
    count_nonhealthy = 0
    
    # Classifica ogni paziente nella cartella di test
    print(f"\nClassificazione dei pazienti in {os.path.basename(test_dir)}:")
    results = {}
    for patient in test_controlmodel.keys():
        if patient in test_diseasemodel:
            result = classify_patient(
                test_controlmodel[patient],
                test_diseasemodel[patient],
                control_controlmodel_ppls,
                control_diseasemodel_ppls,
                disease_controlmodel_ppls,
                disease_diseasemodel_ppls,
                two_sigma=two_sigma,
                use_median=use_median
            )
            print(f"Paziente {patient}: {result}")
            results[patient] = result
            
            # Aggiorna i contatori
            if result == "Control":
                count_control += 1
            else:  # "Non Healthy"
                count_nonhealthy += 1
    
    # Stampa il riepilogo dei risultati
    total_patients = count_control + count_nonhealthy
    print("\nRiepilogo dei risultati:")
    print(f"- Pazienti classificati come Control: {count_control}/{total_patients} ({count_control/total_patients*100:.2f}%)")
    print(f"- Pazienti classificati come Non Healthy: {count_nonhealthy}/{total_patients} ({count_nonhealthy/total_patients*100:.2f}%)")
    print(f"- Totale pazienti classificati: {total_patients}")
    
    # Salva i risultati in un file CSV
    import csv
    output_csv = output_csv or f"classificazione_{os.path.basename(test_dir)}.csv"
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Intestazione
        csv_writer.writerow(['Paziente', 'Classificazione'])
        # Dati
        for patient, classification in results.items():
            csv_writer.writerow([patient, classification])
    print(f"\nRisultati salvati nel file: {output_csv}")
    
    return results

def classify_all_patients_with_plot(test_dir, control_dir, disease_dir, control_model, disease_model, 
                                   use_median=False, two_sigma=False, output_csv=None, label_file=None,
                                   output_png="classification_plot.png"):
    """
    Classifica i pazienti e genera un grafico dei D_score, evidenziando la correttezza della classificazione.
    
    Args:
        test_dir: Directory contenente i pazienti da classificare
        control_dir: Directory contenente i pazienti di controllo
        disease_dir: Directory contenente i pazienti malati
        control_model: Nome del modello di controllo
        disease_model: Nome del modello di malattia
        use_median: Se True, usa la mediana invece della media
        two_sigma: Se True, usa la soglia a due sigma
        output_csv: Nome del file CSV dove salvare i risultati (opzionale)
        label_file: File contenente le etichette vere dei pazienti (opzionale)
        output_png: Nome del file immagine dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Raccogli i valori di perplexity per i dati di riferimento
    control_controlmodel = collect_perplexity_values(control_dir, control_model, llo=True)
    control_diseasemodel = collect_perplexity_values(control_dir, disease_model, llo=False)
    disease_controlmodel = collect_perplexity_values(disease_dir, control_model, llo=False)
    disease_diseasemodel = collect_perplexity_values(disease_dir, disease_model, llo=True)
    
    # Crea le liste per la classificazione
    control_controlmodel_ppls = list(control_controlmodel.values())
    control_diseasemodel_ppls = list(control_diseasemodel.values())
    disease_controlmodel_ppls = list(disease_controlmodel.values())
    disease_diseasemodel_ppls = list(disease_diseasemodel.values())
    
    print(f"Dati di riferimento caricati:")
    print(f"- Pazienti di controllo ({os.path.basename(control_dir)}): {len(control_controlmodel_ppls)}")
    print(f"- Pazienti malati ({os.path.basename(disease_dir)}): {len(disease_diseasemodel_ppls)}")
    
    # Calcola i D_score per i gruppi di riferimento
    control_d_scores = []
    for i in range(0, len(control_controlmodel_ppls)):
        control_d_scores.append(float(control_diseasemodel_ppls[i]) - float(control_controlmodel_ppls[i]))
        
    disease_d_scores = []
    for i in range(0, len(disease_diseasemodel_ppls)):
        disease_d_scores.append(float(disease_diseasemodel_ppls[i]) - float(disease_controlmodel_ppls[i]))  
    
    # Calcola le soglie per la classificazione
    if use_median:
        control_threshold = statistics.median(control_d_scores)
        disease_threshold = statistics.median(disease_d_scores)
        
        if two_sigma:
            control_threshold = statistics.median(control_d_scores) - 2 * std(control_d_scores)
            disease_threshold = statistics.median(disease_d_scores) - 2 * std(disease_d_scores)
    else:
        control_threshold = mean(control_d_scores)
        disease_threshold = mean(disease_d_scores)
        
        if two_sigma:
            control_threshold = mean(control_d_scores) - 2 * std(control_d_scores)
            disease_threshold = mean(disease_d_scores) - 2 * std(disease_d_scores)
    
    # Soglia media tra i due gruppi (punto di passaggio)
    middle_threshold = (control_threshold + disease_threshold) / 2
    
    # Raccogli i valori di perplexity per i pazienti DA CLASSIFICARE
    test_controlmodel = collect_perplexity_values(test_dir, control_model)
    test_diseasemodel = collect_perplexity_values(test_dir, disease_model)
    
    # Carica le etichette vere se disponibili
    true_labels = {}
    if label_file:
        try:
            df = pd.read_csv(label_file, sep=';')
            # Assumiamo che il file abbia colonne 'patient_id' e 'label'
            for _, row in df.iterrows():
                patient_id = row['patient_id']
                # Standardizza il formato delle etichette
                if row['label'].lower() in ['cn', 'control', 'healthy', 'Control']:
                    true_labels[patient_id] = "Control"
                else:  # 'ad', 'disease', 'non-healthy', ecc.
                    true_labels[patient_id] = "Non Healthy"
        except Exception as e:
            print(f"Errore nel caricamento delle etichette: {str(e)}")
            print("Continuo senza verificare la correttezza della classificazione.")
    
    # Lista per salvare i D_score e i risultati di classificazione
    test_d_scores = []
    test_classifications = []
    test_patient_ids = []
    test_true_labels = []
    
    # Ottieni la lista dei pazienti e ordinala in modo lessicografico
    patients = sorted([patient for patient in test_controlmodel.keys() if patient in test_diseasemodel])
    
    # Classifica ogni paziente nella cartella di test e calcola il D_score
    print(f"\nClassificazione dei pazienti in {os.path.basename(test_dir)}:")
    for patient in patients:
        # Calcola il D_score
        d_score = float(test_diseasemodel[patient]) - float(test_controlmodel[patient])
        
        # Classifica in base al D_score
        if abs(d_score - control_threshold) < abs(d_score - disease_threshold):
            classification = "Control"
        else:
            classification = "Non Healthy"
            
        # Ottieni l'etichetta vera se disponibile
        true_label = true_labels.get(patient, None)
        
        # Salva i dati per il grafico
        test_d_scores.append(d_score)
        test_classifications.append(classification)
        test_patient_ids.append(patient)
        test_true_labels.append(true_label)
        
        print(f"Paziente {patient}: D_score = {d_score:.4f}, Classificato come {classification}, Vera etichetta: {true_label}")
    
    # Crea il grafico
    plt.figure(figsize=(14, 8))
    
    # Disegna linee orizzontali per le soglie
    plt.axhline(y=control_threshold, color='green', linestyle='-', alpha=0.5, label=f"D-Score medio controlli: {control_threshold:.4f}")
    plt.axhline(y=disease_threshold, color='red', linestyle='-', alpha=0.5, label=f"D-Score medio malati: {disease_threshold:.4f}")
    plt.axhline(y=middle_threshold, color='purple', linestyle='--', alpha=0.7, label=f"Soglia di classificazione: {middle_threshold:.4f}")
    
    # Organizza i dati per tipo di punto da disegnare
    correct_control = []
    correct_disease = []
    wrong_control = []
    wrong_disease = []
    unknown_control = []  # Pazienti senza etichetta classificati come controllo
    unknown_disease = []  # Pazienti senza etichetta classificati come malati
    
    for i, (d_score, classification, patient_id, true_label) in enumerate(zip(test_d_scores, test_classifications, test_patient_ids, test_true_labels)):
        if true_label is None:
            # Pazienti senza etichetta
            if classification == "Control":
                unknown_control.append((patient_id, d_score, i))
            else:
                unknown_disease.append((patient_id, d_score, i))
        else:
            # Pazienti con etichetta
            is_correct = classification == true_label
            
            if classification == "Control" and is_correct:
                correct_control.append((patient_id, d_score, i))
            elif classification == "Non Healthy" and is_correct:
                correct_disease.append((patient_id, d_score, i))
            elif classification == "Control" and not is_correct:
                wrong_control.append((patient_id, d_score, i))
            elif classification == "Non Healthy" and not is_correct:
                wrong_disease.append((patient_id, d_score, i))
    
    # Prepara l'asse X con gli ID dei pazienti (già ordinati)
    x_positions = np.arange(len(test_patient_ids))
    plt.xticks(x_positions, test_patient_ids, rotation=45, ha='right')
    
    # Disegna i punti usando direttamente l'indice del paziente ordinato
    # Punti per classificazioni corrette
    if correct_control:
        ids, scores, indices = zip(*correct_control)
        plt.scatter(indices, scores, color='green', marker='o', s=100, label='Controllo (classificato correttamente)')
    
    if correct_disease:
        ids, scores, indices = zip(*correct_disease)
        plt.scatter(indices, scores, color='red', marker='o', s=100, label='Malato (classificato correttamente)')
    
    # Punti per classificazioni errate
    if wrong_control:
        ids, scores, indices = zip(*wrong_control)
        plt.scatter(indices, scores, color='green', marker='x', s=120, linewidths=2, label='Controllo (classificato erroneamente)')
    
    if wrong_disease:
        ids, scores, indices = zip(*wrong_disease)
        plt.scatter(indices, scores, color='red', marker='x', s=120, linewidths=2, label='Malato (classificato erroneamente)')
    
    # Punti per classificazioni sconosciute (nessuna etichetta vera)
    if unknown_control:
        ids, scores, indices = zip(*unknown_control)
        plt.scatter(indices, scores, color='blue', marker='o', s=80, alpha=0.6, label='Classificato come controllo (etichetta sconosciuta)')
    
    if unknown_disease:
        ids, scores, indices = zip(*unknown_disease)
        plt.scatter(indices, scores, color='orange', marker='o', s=80, alpha=0.6, label='Classificato come malato (etichetta sconosciuta)')
    
    # Configura il grafico
    plt.title('Classificazione dei pazienti in base al D-Score', fontsize=16)
    plt.xlabel('ID Paziente', fontsize=12)
    plt.ylabel('D-Score (Perplexity_disease - Perplexity_control)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Salva il grafico
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nGrafico salvato: {output_png}")
    
    # Restituisci le informazioni di classificazione
    return {
        'patient_ids': test_patient_ids,
        'classifications': test_classifications,
        'true_labels': test_true_labels,
        'd_scores': test_d_scores,
        'control_threshold': control_threshold,
        'disease_threshold': disease_threshold,
        'middle_threshold': middle_threshold
    }


# def classify_all_patients_loo(test_dir, control_dir, disease_dir, control_model, disease_model, use_median=False, two_sigma=False, output_csv=None):
#     """
#     Classifica i pazienti nella cartella specificata utilizzando i dati di riferimento.
    
#     Args:
#         test_dir: Directory contenente i pazienti da classificare
#         control_dir: Directory contenente i pazienti di controllo
#         disease_dir: Directory contenente i pazienti malati
#         control_model: Nome del modello di controllo
#         disease_model: Nome del modello di malattia
#         use_median: Se True, usa la mediana invece della media
#         two_sigma: Se True, usa la soglia a due sigma
#         output_csv: Nome del file CSV dove salvare i risultati (opzionale)
#     """
#     # Raccogli i valori di perplexity per i dati di riferimento
#     control_controlmodel = collect_perplexity_values(control_dir, control_model, llo=True)
#     control_diseasemodel = collect_perplexity_values(control_dir, disease_model, llo=False)
#     disease_controlmodel = collect_perplexity_values(disease_dir, control_model, llo=False)
#     disease_diseasemodel = collect_perplexity_values(disease_dir, disease_model, llo=True)
    
#     # Crea le liste per la classificazione
#     control_controlmodel_ppls = list(control_controlmodel.values())
#     control_diseasemodel_ppls = list(control_diseasemodel.values())
#     disease_controlmodel_ppls = list(disease_controlmodel.values())
#     disease_diseasemodel_ppls = list(disease_diseasemodel.values())
    
#     print(f"Dati di riferimento caricati:")
#     print(f"- Pazienti di controllo ({os.path.basename(control_dir)}): {len(control_controlmodel_ppls)}")
#     print(f"- Pazienti malati ({os.path.basename(disease_dir)}): {len(disease_diseasemodel_ppls)}")
    
#     # Raccogli i valori di perplexity per i pazienti DA CLASSIFICARE
#     test_controlmodel = control_controlmodel | disease_controlmodel_ppls
#     test_diseasemodel = control_diseasemodel_ppls | disease_diseasemodel_ppls
    
#     # Contatori per i risultati
#     count_control = 0
#     count_nonhealthy = 0
    
#     # Classifica ogni paziente nella cartella di test
#     print(f"\nClassificazione dei pazienti in {os.path.basename(test_dir)}:")
#     results = {}
#     for patient in test_controlmodel.keys():
#         if patient in test_diseasemodel:
#             result = classify_patient(
#                 test_controlmodel[patient],
#                 test_diseasemodel[patient],
#                 control_controlmodel_ppls,
#                 control_diseasemodel_ppls,
#                 disease_controlmodel_ppls,
#                 disease_diseasemodel_ppls,
#                 two_sigma=two_sigma,
#                 use_median=use_median
#             )
#             print(f"Paziente {patient}: {result}")
#             results[patient] = result
            
#             # Aggiorna i contatori
#             if result == "Control":
#                 count_control += 1
#             else:  # "Non Healthy"
#                 count_nonhealthy += 1
    
#     # Stampa il riepilogo dei risultati
#     total_patients = count_control + count_nonhealthy
#     print("\nRiepilogo dei risultati:")
#     print(f"- Pazienti classificati come Control: {count_control}/{total_patients} ({count_control/total_patients*100:.2f}%)")
#     print(f"- Pazienti classificati come Non Healthy: {count_nonhealthy}/{total_patients} ({count_nonhealthy/total_patients*100:.2f}%)")
#     print(f"- Totale pazienti classificati: {total_patients}")
    
#     # Salva i risultati in un file CSV
#     import csv
#     output_csv = output_csv or f"classificazione_{os.path.basename(test_dir)}.csv"
#     with open(output_csv, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         # Intestazione
#         csv_writer.writerow(['Paziente', 'Classificazione'])
#         # Dati
#         for patient, classification in results.items():
#             csv_writer.writerow([patient, classification])
#     print(f"\nRisultati salvati nel file: {output_csv}")
    
#     return results

# def classify_all_patients_with_plot_loo(test_dir, control_dir, disease_dir, control_model, disease_model, 
#                                    use_median=False, two_sigma=False, output_csv=None, label_file=None,
#                                    output_png="classification_plot.png"):
#     """
#     Classifica i pazienti e genera un grafico dei D_score, evidenziando la correttezza della classificazione.
    
#     Args:
#         test_dir: Directory contenente i pazienti da classificare
#         control_dir: Directory contenente i pazienti di controllo
#         disease_dir: Directory contenente i pazienti malati
#         control_model: Nome del modello di controllo
#         disease_model: Nome del modello di malattia
#         use_median: Se True, usa la mediana invece della media
#         two_sigma: Se True, usa la soglia a due sigma
#         output_csv: Nome del file CSV dove salvare i risultati (opzionale)
#         label_file: File contenente le etichette vere dei pazienti (opzionale)
#         output_png: Nome del file immagine dove salvare il grafico
#     """
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
    
#     # Raccogli i valori di perplexity per i dati di riferimento
#     control_controlmodel = collect_perplexity_values(control_dir, control_model, llo=True)
#     control_diseasemodel = collect_perplexity_values(control_dir, disease_model, llo=False)
#     disease_controlmodel = collect_perplexity_values(disease_dir, control_model, llo=False)
#     disease_diseasemodel = collect_perplexity_values(disease_dir, disease_model, llo=True)
    
#     # Crea le liste per la classificazione
#     control_controlmodel_ppls = list(control_controlmodel.values())
#     control_diseasemodel_ppls = list(control_diseasemodel.values())
#     disease_controlmodel_ppls = list(disease_controlmodel.values())
#     disease_diseasemodel_ppls = list(disease_diseasemodel.values())
    
#     print(f"Dati di riferimento caricati:")
#     print(f"- Pazienti di controllo ({os.path.basename(control_dir)}): {len(control_controlmodel_ppls)}")
#     print(f"- Pazienti malati ({os.path.basename(disease_dir)}): {len(disease_diseasemodel_ppls)}")
    
#     # Calcola i D_score per i gruppi di riferimento
#     control_d_scores = []
#     for i in range(0, len(control_controlmodel_ppls)):
#         control_d_scores.append(float(control_diseasemodel_ppls[i]) - float(control_controlmodel_ppls[i]))
        
#     disease_d_scores = []
#     for i in range(0, len(disease_diseasemodel_ppls)):
#         disease_d_scores.append(float(disease_diseasemodel_ppls[i]) - float(disease_controlmodel_ppls[i]))  
    
#     # Calcola le soglie per la classificazione
#     if use_median:
#         control_threshold = statistics.median(control_d_scores)
#         disease_threshold = statistics.median(disease_d_scores)
        
#         if two_sigma:
#             control_threshold = statistics.median(control_d_scores) - 2 * std(control_d_scores)
#             disease_threshold = statistics.median(disease_d_scores) - 2 * std(disease_d_scores)
#     else:
#         control_threshold = mean(control_d_scores)
#         disease_threshold = mean(disease_d_scores)
        
#         if two_sigma:
#             control_threshold = mean(control_d_scores) - 2 * std(control_d_scores)
#             disease_threshold = mean(disease_d_scores) - 2 * std(disease_d_scores)
    
#     # Soglia media tra i due gruppi (punto di passaggio)
#     middle_threshold = (control_threshold + disease_threshold) / 2
    
#     # Raccogli i valori di perplexity per i pazienti DA CLASSIFICARE
#     test_controlmodel = collect_perplexity_values(test_dir, control_model)
#     test_diseasemodel = collect_perplexity_values(test_dir, disease_model)
    
#     # Carica le etichette vere se disponibili
#     true_labels = {}
#     if label_file:
#         try:
#             df = pd.read_csv(label_file, sep=',')
#             # Assumiamo che il file abbia colonne 'patient_id' e 'label'
#             for _, row in df.iterrows():
#                 patient_id = row['patient_id']
#                 # Standardizza il formato delle etichette
#                 if row['label'].lower() in ['cn', 'control', 'healthy', 'Control']:
#                     true_labels[patient_id] = "Control"
#                 else:  # 'ad', 'disease', 'non-healthy', ecc.
#                     true_labels[patient_id] = "Non Healthy"
#         except Exception as e:
#             print(f"Errore nel caricamento delle etichette: {str(e)}")
#             print("Continuo senza verificare la correttezza della classificazione.")
    
#     # Lista per salvare i D_score e i risultati di classificazione
#     test_d_scores = []
#     test_classifications = []
#     test_patient_ids = []
#     test_true_labels = []
    
#     # Ottieni la lista dei pazienti e ordinala in modo lessicografico
#     patients = sorted([patient for patient in test_controlmodel.keys() if patient in test_diseasemodel])
    
#     # Classifica ogni paziente nella cartella di test e calcola il D_score
#     print(f"\nClassificazione dei pazienti in {os.path.basename(test_dir)}:")
#     for patient in patients:
#         # Calcola il D_score
#         d_score = float(test_diseasemodel[patient]) - float(test_controlmodel[patient])
        
#         # Classifica in base al D_score
#         if abs(d_score - control_threshold) < abs(d_score - disease_threshold):
#             classification = "Control"
#         else:
#             classification = "Non Healthy"
            
#         # Ottieni l'etichetta vera se disponibile
#         true_label = true_labels.get(patient, None)
        
#         # Salva i dati per il grafico
#         test_d_scores.append(d_score)
#         test_classifications.append(classification)
#         test_patient_ids.append(patient)
#         test_true_labels.append(true_label)
        
#         print(f"Paziente {patient}: D_score = {d_score:.4f}, Classificato come {classification}, Vera etichetta: {true_label}")
    
#     # Crea il grafico
#     plt.figure(figsize=(14, 8))
    
#     # Disegna linee orizzontali per le soglie
#     plt.axhline(y=control_threshold, color='green', linestyle='-', alpha=0.5, label=f"D-Score medio controlli: {control_threshold:.4f}")
#     plt.axhline(y=disease_threshold, color='red', linestyle='-', alpha=0.5, label=f"D-Score medio malati: {disease_threshold:.4f}")
#     plt.axhline(y=middle_threshold, color='purple', linestyle='--', alpha=0.7, label=f"Soglia di classificazione: {middle_threshold:.4f}")
    
#     # Organizza i dati per tipo di punto da disegnare
#     correct_control = []
#     correct_disease = []
#     wrong_control = []
#     wrong_disease = []
#     unknown_control = []  # Pazienti senza etichetta classificati come controllo
#     unknown_disease = []  # Pazienti senza etichetta classificati come malati
    
#     for i, (d_score, classification, patient_id, true_label) in enumerate(zip(test_d_scores, test_classifications, test_patient_ids, test_true_labels)):
#         if true_label is None:
#             # Pazienti senza etichetta
#             if classification == "Control":
#                 unknown_control.append((patient_id, d_score, i))
#             else:
#                 unknown_disease.append((patient_id, d_score, i))
#         else:
#             # Pazienti con etichetta
#             is_correct = classification == true_label
            
#             if classification == "Control" and is_correct:
#                 correct_control.append((patient_id, d_score, i))
#             elif classification == "Non Healthy" and is_correct:
#                 correct_disease.append((patient_id, d_score, i))
#             elif classification == "Control" and not is_correct:
#                 wrong_control.append((patient_id, d_score, i))
#             elif classification == "Non Healthy" and not is_correct:
#                 wrong_disease.append((patient_id, d_score, i))
    
#     # Prepara l'asse X con gli ID dei pazienti (già ordinati)
#     x_positions = np.arange(len(test_patient_ids))
#     plt.xticks(x_positions, test_patient_ids, rotation=45, ha='right')
    
#     # Disegna i punti usando direttamente l'indice del paziente ordinato
#     # Punti per classificazioni corrette
#     if correct_control:
#         ids, scores, indices = zip(*correct_control)
#         plt.scatter(indices, scores, color='green', marker='o', s=100, label='Controllo (classificato correttamente)')
    
#     if correct_disease:
#         ids, scores, indices = zip(*correct_disease)
#         plt.scatter(indices, scores, color='red', marker='o', s=100, label='Malato (classificato correttamente)')
    
#     # Punti per classificazioni errate
#     if wrong_control:
#         ids, scores, indices = zip(*wrong_control)
#         plt.scatter(indices, scores, color='green', marker='x', s=120, linewidths=2, label='Controllo (classificato erroneamente)')
    
#     if wrong_disease:
#         ids, scores, indices = zip(*wrong_disease)
#         plt.scatter(indices, scores, color='red', marker='x', s=120, linewidths=2, label='Malato (classificato erroneamente)')
    
#     # Punti per classificazioni sconosciute (nessuna etichetta vera)
#     if unknown_control:
#         ids, scores, indices = zip(*unknown_control)
#         plt.scatter(indices, scores, color='blue', marker='o', s=80, alpha=0.6, label='Classificato come controllo (etichetta sconosciuta)')
    
#     if unknown_disease:
#         ids, scores, indices = zip(*unknown_disease)
#         plt.scatter(indices, scores, color='orange', marker='o', s=80, alpha=0.6, label='Classificato come malato (etichetta sconosciuta)')
    
#     # Configura il grafico
#     plt.title('Classificazione dei pazienti in base al D-Score', fontsize=16)
#     plt.xlabel('ID Paziente', fontsize=12)
#     plt.ylabel('D-Score (Perplexity_disease - Perplexity_control)', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(loc='best')
#     plt.tight_layout()
    
#     # Salva il grafico
#     plt.savefig(output_png, dpi=300, bbox_inches='tight')
#     print(f"\nGrafico salvato: {output_png}")
    
#     # Restituisci le informazioni di classificazione
#     return {
#         'patient_ids': test_patient_ids,
#         'classifications': test_classifications,
#         'true_labels': test_true_labels,
#         'd_scores': test_d_scores,
#         'control_threshold': control_threshold,
#         'disease_threshold': disease_threshold,
#         'middle_threshold': middle_threshold
#     }


def calculate_f1(dataset_dir, label_file):
    """
    Calcola il punteggio precision, recall e F1 per la classificazione per tutti file contenuti nella cartella src che iniziano con 'risultati_classificazione_' e sono csv. Al termine ordine i risultati dal migliore al peggiore e stampa una lista mettendo nome del file e i punteggi. il file delle labels è in ../resources/data/input/adresso_all_text/test/labels.csv
    
    """
    import pandas as pd
    import os
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    # Percorso della cartella src
    src_dir = dataset_dir
    
    # Percorso del file delle etichette vere
    #label_file = "../resources/data/input/adresso_final_fold/test/labels.csv"
    
    # Leggi le etichette vere
    try:
        true_labels_df = pd.read_csv(label_file, sep=';')
        # Crea un dizionario per accesso rapido alle etichette
        true_labels = {}
        for _, row in true_labels_df.iterrows():
            patient_id = row['patient_id']
            # Standardizza il formato delle etichette
            if row['label'].lower() in ['cn', 'control', 'healthy']:
                true_labels[patient_id] = "Control"
            else:  # 'ad', 'disease', 'non-healthy', ecc.
                true_labels[patient_id] = "Non Healthy"
        
        print(f"Etichette caricate per {len(true_labels)} pazienti")
    except Exception as e:
        print(f"Errore nel caricamento delle etichette: {str(e)}")
        return
    
    # Trova tutti i file CSV di classificazione
    classification_files = [f for f in os.listdir(src_dir) 
                           if f.startswith('risultati_classificazione_') and f.endswith('.csv')]
    
    if not classification_files:
        print("Nessun file di classificazione trovato.")
        return
    
    print(f"Trovati {len(classification_files)} file di classificazione:")
    for f in classification_files:
        print(f"  - {f}")
    
    # Lista per salvare i risultati
    results = []
    
    # Elabora ogni file di classificazione
    for file_name in classification_files:
        file_path = os.path.join(src_dir, file_name)
        
        try:
            # Leggi i risultati della classificazione
            pred_df = pd.read_csv(file_path)
            
            # Raccogli le previsioni e le etichette vere per i pazienti disponibili
            y_true = []
            y_pred = []
            missing_labels = 0
            
            for _, row in pred_df.iterrows():
                patient_id = row['Paziente']
                pred_label = row['Classificazione']
                
                if patient_id in true_labels:
                    y_true.append(1 if true_labels[patient_id] == "Non Healthy" else 0)
                    y_pred.append(1 if pred_label == "Non Healthy" else 0)
                else:
                    missing_labels += 1
            
            # Assicurati che ci siano dati sufficienti per calcolare le metriche
            if len(y_true) < 2:
                print(f"Dati insufficienti in {file_name}, saltato.")
                continue
            
            # Crea la matrice di confusione
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calcola il numero di corretti e sbagliati
            correct = tp + tn
            incorrect = fp + fn
            total = correct + incorrect
            
            # Calcola le metriche per ogni classe
            
            # Classe 0 (Control)
            if tn + fn > 0:  # Ci sono esempi reali della classe Control
                precision_control = tn / (tn + fn) if (tn + fn) > 0 else 0
                recall_control = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1_control = 2 * precision_control * recall_control / (precision_control + recall_control) if (precision_control + recall_control) > 0 else 0
            else:
                precision_control = recall_control = f1_control = 0
            
            # Classe 1 (Non Healthy)
            if tp + fp > 0:  # Ci sono esempi reali della classe Non Healthy
                precision_nh = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_nh = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_nh = 2 * precision_nh * recall_nh / (precision_nh + recall_nh) if (precision_nh + recall_nh) > 0 else 0
            else:
                precision_nh = recall_nh = f1_nh = 0
            
            # Media delle F1 (macro)
            f1_macro = (f1_control + f1_nh) / 2
            
            # Calcola accuracy globale
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Salva i risultati
            results.append({
                'file_name': file_name,
                'precision_control': precision_control,
                'recall_control': recall_control,
                'f1_control': f1_control,
                'precision_nh': precision_nh,
                'recall_nh': recall_nh,
                'f1_nh': f1_nh,
                'f1_macro': f1_macro,
                'accuracy': accuracy,
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn,
                'correct': correct,
                'incorrect': incorrect,
                'total_patients': len(y_true),
                'missing_labels': missing_labels
            })
            
            print(f"\nAnalisi di {file_name}:")
            print(f"  Accuracy globale: {accuracy:.4f} ({correct}/{total} corretti)")
            print(f"  Classe Control:")
            print(f"    Precision: {precision_control:.4f}")
            print(f"    Recall: {recall_control:.4f}")
            print(f"    F1 Score: {f1_control:.4f}")
            print(f"  Classe Non Healthy:")
            print(f"    Precision: {precision_nh:.4f}")
            print(f"    Recall: {recall_nh:.4f}")
            print(f"    F1 Score: {f1_nh:.4f}")
            print(f"  F1 Macro (media): {f1_macro:.4f}")
            print(f"  Matrice di confusione: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            
        except Exception as e:
            print(f"Errore nell'elaborazione del file {file_name}: {str(e)}")
    
    # Ordina i risultati per F1 macro (dal migliore al peggiore)
    results_sorted = sorted(results, key=lambda x: x['f1_macro'], reverse=True)
    
    # Stampa i risultati ordinati
    print("\n=== RISULTATI ORDINATI PER F1 SCORE MEDIO ===")
    for i, res in enumerate(results_sorted, 1):
        print(f"{i}. {res['file_name']}")
        print(f"   F1 Macro: {res['f1_macro']:.4f}")
        print(f"   Control: F1={res['f1_control']:.4f}, Prec={res['precision_control']:.4f}, Rec={res['recall_control']:.4f}")
        print(f"   Non Healthy: F1={res['f1_nh']:.4f}, Prec={res['precision_nh']:.4f}, Rec={res['recall_nh']:.4f}")
        print(f"   Accuracy: {res['accuracy']:.4f} ({res['correct']}/{res['correct'] + res['incorrect']} corretti)")
        print(f"   Matrice: TP={res['true_positive']}, FP={res['false_positive']}, TN={res['true_negative']}, FN={res['false_negative']}")
        print()
    
    #stampa su file la stessa cosa 
    
    with open(os.path.join(src_dir, 'sorted_classification_results.txt'), 'w') as f:
        f.write("=== RISULTATI ORDINATI PER F1 SCORE MEDIO ===\n")
        for i, res in enumerate(results_sorted, 1):
            f.write(f"{i}. {res['file_name']}\n")
            f.write(f"   F1 Macro: {res['f1_macro']:.4f}\n")
            f.write(f"   Control: F1={res['f1_control']:.4f}, Prec={res['precision_control']:.4f}, Rec={res['recall_control']:.4f}\n")
            f.write(f"   Non Healthy: F1={res['f1_nh']:.4f}, Prec={res['precision_nh']:.4f}, Rec={res['recall_nh']:.4f}\n")
            f.write(f"   Accuracy: {res['accuracy']:.4f} ({res['correct']}/{res['correct'] + res['incorrect']} corretti)\n")
            f.write(f"   Matrice: TP={res['true_positive']}, FP={res['false_positive']}, TN={res['true_negative']}, FN={res['false_negative']}\n\n")
            
    return results_sorted


    
    calculate_f1("../resources/test_adresso_gpt2_j_final_fold")
    exit()
    
    ############### ADRESSO ################
    
    dataset_name = "adresso_final_fold"
    batch = 12
    
    text_folder = "whisper-large-v3-turbo"
    leap = 0
    
    adep = 5
    cnep = 5
    
    for window in [20]:
              
        classify_all_patients(
            test_dir=f"../resources/data/output_gpt2_j/{dataset_name}_w{window}_l{leap}/test/{text_folder}/",
            control_dir=f"../resources/data/output_gpt2_j/{dataset_name}_w{window}_l{leap}/train/cn/{text_folder}/",
            disease_dir=f"../resources/data/output_gpt2_j/{dataset_name}_w{window}_l{leap}/train/ad/{text_folder}/",
            control_model=f"cn_{text_folder}_{batch}b_{cnep}ep",
            disease_model=f"ad_{text_folder}_{batch}b_{adep}ep",
            use_median=False,
            two_sigma=False,
            output_csv=f"../resources/test_adresso_gpt2_j_final_fold/risultati_classificazione_TEST_{cnep}cn_{adep}ad_ep_{batch}b_{text_folder}_{window}w.csv"
        )
        
        # classify_all_patients(
        #     test_dir=f"../resources/data/output/{dataset_name}_w{window}_l{leap}/valid/{text_folder}/",
        #     control_dir=f"../resources/data/output/{dataset_name}_w{window}_l{leap}/train/cn/{text_folder}/",
        #     disease_dir=f"../resources/data/output/{dataset_name}_w{window}_l{leap}/train/depressed/{text_folder}/",
        #     control_model=f"cn_{text_folder}_{batch}b_{cnep}ep",
        #     disease_model=f"depressed_{text_folder}_{batch}b_{adep}ep",
        #     use_median=False,
        #     two_sigma=False,
        #     output_csv=f"../resources/valid_results_leap{leap}_{text_folder}/risultati_classificazione_VALID_{cnep}cn_{adep}depressed_ep_16b_{text_folder}_{window}w.csv"
        # )
        
        classify_all_patients_with_plot(
            test_dir=f"../resources/data/output_gpt2_j/{dataset_name}_w{window}_l{leap}/test/{text_folder}/",
            control_dir=f"../resources/data/output_gpt2_j/{dataset_name}_w{window}_l{leap}/train/cn/{text_folder}/",
            disease_dir=f"../resources/data/output_gpt2_j/{dataset_name}_w{window}_l{leap}/train/ad/{text_folder}/",
            control_model=f"cn_{text_folder}_{batch}b_{cnep}ep",
            disease_model=f"ad_{text_folder}_{batch}b_{adep}ep",
            use_median=False,
            two_sigma=False,
            label_file=f"../resources/data/input/{dataset_name}/test/labels.csv",
            output_png=f"../resources/test_adresso_gpt2_j_final_fold/plot_TEST_{text_folder}_{batch}b_{cnep}cn_{adep}ad_ep_{window}w.png"
        )
        
        # classify_all_patients_with_plot(
        #     test_dir=f"../resources/data/output/{dataset_name}_w{window}_l{leap}/valid/{text_folder}/",
        #     control_dir=f"../resources/data/output/{dataset_name}_w{window}_l{leap}/train/cn/{text_folder}/",
        #     disease_dir=f"../resources/data/output/{dataset_name}_w{window}_l{leap}/train/depressed/{text_folder}/",
        #     control_model=f"cn_{text_folder}_{batch}b_{cnep}ep",
        #     disease_model=f"depressed_{text_folder}_{batch}b_{adep}ep",
        #     use_median=False,
        #     two_sigma=False,
        #     label_file=f"../resources/data/input/{dataset_name}/valid/labels.csv",
        #     output_png=f"../resources/valid_results_leap{leap}_{text_folder}/plot_VALID_{text_folder}_{batch}b_{cnep}cn_{adep}ad_ep_{window}w.png"
        # )