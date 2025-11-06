
from utils.classification import collect_perplexity_values

def minor_strategy(dev_control_dir, dev_disease_dir, control_class, disease_class, text_folder, batch_size, epochs, logger):
    """Select best epochs based on minimum mean perplexity on dev sets."""
    min_dev_control_ppl = float('inf')
    min_dev_disease_ppl = float('inf')
    best_epoch_control = -1
    best_epoch_disease = -1
    
    for ep in epochs:
        control_model = f"{control_class}_{text_folder}_{batch_size}b_{ep}ep"
        disease_model = f"{disease_class}_{text_folder}_{batch_size}b_{ep}ep"
    
        control_ppl_values = collect_perplexity_values(dev_control_dir, control_model, llo=False)
        disease_ppl_values = collect_perplexity_values(dev_disease_dir, disease_model, llo=False)
        
        mean_dev_ppl_control = sum(float(val) for val in control_ppl_values.values()) / len(control_ppl_values)
        mean_dev_ppl_disease = sum(float(val) for val in disease_ppl_values.values()) / len(disease_ppl_values)

        logger.info(f"Epoch {ep}: Mean Dev Control PPL = {mean_dev_ppl_control}, Mean Dev Disease PPL = {mean_dev_ppl_disease}")
        
        if mean_dev_ppl_control < min_dev_control_ppl:
            min_dev_control_ppl = mean_dev_ppl_control
            best_epoch_control = ep
        
        if mean_dev_ppl_disease < min_dev_disease_ppl:
            min_dev_disease_ppl = mean_dev_ppl_disease
            best_epoch_disease = ep
    
    return best_epoch_control, best_epoch_disease, min_dev_control_ppl, min_dev_disease_ppl

def delta_strategy(dev_control_dir, dev_disease_dir, control_class, disease_class, text_folder, batch_size, epochs, logger):
    
    """Select best epochs based on minimum delta of perplexity on dev sets."""
    dev_control_mean_ppl_per_epoch = dict()
    dev_disease_mean_ppl_per_epoch = dict()
    
    min_delta = float('inf')
    
    best_epoch_control = -1
    best_epoch_disease = -1
    
    for ep in epochs:
        control_model = f"{control_class}_{text_folder}_{batch_size}b_{ep}ep"
        disease_model = f"{disease_class}_{text_folder}_{batch_size}b_{ep}ep"
    
        control_model_ppl_values_1 = collect_perplexity_values(dev_control_dir, control_model, llo=False)
        control_model_ppl_values_2 = collect_perplexity_values(dev_disease_dir, control_model, llo=False)
        #merge the two dicts
      
        control_model_ppl_values = {**control_model_ppl_values_1, **control_model_ppl_values_2}
        
        disease_model_ppl_values_1 = collect_perplexity_values(dev_disease_dir, disease_model, llo=False)
        disease_model_ppl_values_2 = collect_perplexity_values(dev_control_dir, disease_model, llo=False)
        
       
        disease_model_ppl_values = {**disease_model_ppl_values_1, **disease_model_ppl_values_2}
        
        mean_dev_ppl_control = sum(float(val) for val in control_model_ppl_values.values()) / len(control_model_ppl_values)
        mean_dev_ppl_disease = sum(float(val) for val in disease_model_ppl_values.values()) / len(disease_model_ppl_values)
        

        logger.info(f"Epoch {ep}: Mean Dev Control PPL = {mean_dev_ppl_control}, Mean Dev Disease PPL = {mean_dev_ppl_disease}")
        
        dev_control_mean_ppl_per_epoch[ep] = mean_dev_ppl_control
        dev_disease_mean_ppl_per_epoch[ep] = mean_dev_ppl_disease
    
    #calculate the delta between the two dicts, for every possible couple of epochs control-disease
    for ep1 in epochs:
        for ep2 in epochs:
            delta = abs(dev_control_mean_ppl_per_epoch[ep1] - dev_disease_mean_ppl_per_epoch[ep2])
            if best_epoch_control == -1 and best_epoch_disease == -1:
                min_delta = delta
                best_epoch_control = ep1
                best_epoch_disease = ep2
            elif delta < min_delta:
                min_delta = delta
                best_epoch_control = ep1
                best_epoch_disease = ep2
    
    return best_epoch_control, best_epoch_disease, min_delta
