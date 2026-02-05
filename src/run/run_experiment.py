import os
import sys
import logging
from datetime import datetime
from tqdm import tqdm
import torch
import gc

# Configura PyTorch per evitare frammentazione memoria
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add src path to PYTHONPATH for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from utils.models_training import train_model, leave_one_out_trainingV1

from utils.evaluation_optimized import process_data_folders_multi_gpu, process_leave_one_out_multi_gpu

from utils.classification import classify_all_patients, classify_all_patients_with_plot, calculate_f1

from utils.strategies import minor_strategy, delta_strategy

def setup_logging():
    """Configure logging for the complete experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"gpt_experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main function that executes the complete experiment."""
    logger = setup_logging()
    
    logger.info("Starting complete experiment")
    
    model_type = 'gpt2'  # 'gpt2' or 'llama_lora'
    #model_type = 'llama_lora'

    model_name = 'openai-community/gpt2'
    #model_name = 'meta-llama/Llama-3.2-1B'
    
    dataset_name = 'adresso_final_fold' 
    
    text_folder = 'whisper-large-v3-turbo'
    
    disease_class = 'ad'  
    control_class = 'cn'
    
    #strategy_name = 'minor' 
    strategy_name = 'delta' 
    
    cross_dev = False
    
    # training parameters
    batch_size = 24  # set batch size # 12 for gpt2 2 for llama
    #batch_size = 8  # set batch size
    max_epochs = 20  # set max epochs
    save_every = 1 # save model every these epochs
    
    # Memory optimization parameters
    gradient_checkpointing = True
    dataloader_num_workers = 0  # Evita memory leak con multiprocessing
    
    # Define variables used later in perplexity calculation
    w = 20
    leap = 0
    
    do_train = True
    
    try:
        
        if do_train:
            # Phase 1: Training
            logger.info("=" * 50)
            logger.info("PHASE 1: MODEL TRAINING")
            logger.info("=" * 50)
            
            # Train control model for max_epochs, saving every save_every epochs
            # train_model(
            #     model_type=model_type,
            #     model_name=model_name,
            #     max_epochs=max_epochs,
            #     batch_size=batch_size,
            #     train_set_file_path=f'../resources/data/input/{dataset_name}/train/{control_class}/{text_folder}/',
            #     base_output_dir=f'../resources/data/output/models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b',
            #     save_every=save_every,
            #     gradient_checkpointing=gradient_checkpointing
            # )
            
            
            # Pulizia aggressiva della memoria
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Forza la liberazione della cache
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            logger.info("Memory cleaned after control model training")
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            
            # Pausa per dare tempo al sistema di liberare completamente la memoria
            import time
            time.sleep(5)
            
            logger.info("Starting disease model training...")
            
            # Train disease model for max_epochs, saving every save_every epochs
            # train_model(
            #     model_type=model_type,
            #     model_name=model_name,
            #     max_epochs=max_epochs,
            #     batch_size=batch_size,
            #     train_set_file_path=f'../resources/data/input/{dataset_name}/train/{disease_class}/{text_folder}/',
            #     base_output_dir=f'../resources/data/output/models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b',
            #     save_every=save_every,
            #     gradient_checkpointing=gradient_checkpointing
            # )
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        
        
        logger.info("Training completed successfully!")
        
        # Phase 2: Evaluation
        logger.info("=" * 50)
        logger.info("PHASE 2: PERPLEXITY PRODUCTION FOR DEV AND MODEL SELECTION")
        logger.info("=" * 50)
        
        epochs = list(range(2, max_epochs + 1, 2))  # Evaluate every 2 epochs
        num_gpus = 2
        
        
        for ep in tqdm(epochs):
            # DISEASE on DISEASE DEV
            
            if do_train:
                model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{ep}ep"
            else:
                model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{ep}ep"
                
            process_data_folders_multi_gpu(
                model_type=model_type,
                model_dir=model_directory,
                input_base_dir=f"../resources/data/input/{dataset_name}/dev/{disease_class}/{text_folder}/",
                output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/dev/{disease_class}/{text_folder}/",
                test_file_name="test_text.txt",
                window=w,
                num_gpus=num_gpus,  
                leap=leap
            )
            
            # CONTROL on CONTROL DEV
            if do_train:
                model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{ep}ep"
            else:
                model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{ep}ep"
                
            process_data_folders_multi_gpu(
                model_type=model_type,
                model_dir=model_directory,
                input_base_dir=f"../resources/data/input/{dataset_name}/dev/{control_class}/{text_folder}/",
                output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/dev/{control_class}/{text_folder}/",
                test_file_name="test_text.txt",
                window=w,
                num_gpus=num_gpus,  
                leap=leap
            )
            
            if cross_dev:
            
                # CONTROL on DISEASE DEV
                
                if do_train:
                    model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{ep}ep"
                else:
                    model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{ep}ep"
                    
                process_data_folders_multi_gpu(
                    model_type=model_type,
                    model_dir=model_directory,
                    input_base_dir=f"../resources/data/input/{dataset_name}/dev/{disease_class}/{text_folder}/",
                    output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/dev/{disease_class}/{text_folder}/",
                    test_file_name="test_text.txt",
                    window=w,
                    num_gpus=num_gpus,  
                    leap=leap
                )
                
                # DISEASE on CONTROL DEV
                if do_train:
                    model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{ep}ep"
                else:
                    model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{ep}ep"
                    
                process_data_folders_multi_gpu(
                    model_type=model_type,
                    model_dir=model_directory,
                    input_base_dir=f"../resources/data/input/{dataset_name}/dev/{control_class}/{text_folder}/",
                    output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/dev/{control_class}/{text_folder}/",
                    test_file_name="test_text.txt",
                    window=w,
                    num_gpus=num_gpus,  
                    leap=leap
                )
            
        # Evaluate and select best models based on dev set perplexities
        # calculate the mean perplexities at every epoch for both classes for dev set and select where it is minumal to chose best epoch
        
        dev_control_dir = f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/dev/{control_class}/{text_folder}/"
        dev_disease_dir = f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/dev/{disease_class}/{text_folder}/"
        
        if strategy_name == 'minor':
            best_epoch_control, best_epoch_disease, min_dev_control_ppl, min_dev_disease_ppl = minor_strategy(dev_control_dir, dev_disease_dir, control_class, disease_class, text_folder, batch_size, epochs, logger)
            
            logger.info(f"Best Epochs selected using Minor Strategy: Control = {best_epoch_control}, Disease = {best_epoch_disease}")    
            logger.info(f"with control PPL: {min_dev_control_ppl}")
            logger.info(f"with disease PPL: {min_dev_disease_ppl}")
            
        elif strategy_name == 'delta':
        
            best_epoch_control, best_epoch_disease, min_delta = delta_strategy(dev_control_dir, dev_disease_dir, control_class, disease_class, text_folder, batch_size, epochs, logger)
            logger.info(f"Best Epochs selected using Delta Strategy: Control = {best_epoch_control}, Disease = {best_epoch_disease} with Delta: {min_delta}")
        
        
        logger.info("Perplexity production completed successfully!")
        
        # Phase 3: loo Models training until best epochs and perplexity production on Train and Test sets using best models
        logger.info("=" * 50)
        logger.info("PHASE 3: LOO MODELS TRAINING UNTIL BEST EPOCHS PERPLEXITY PRODUCTION FOR TRAIN AND TEST SETS USING BEST MODELS")
        logger.info("=" * 50)
        
        if do_train:
            # leave one out training on control group
            
            base_train_set_file_path = f'../resources/data/input/{dataset_name}/train/{control_class}/{text_folder}/'
            base_train_model_output_dir = f'../resources/data/output/models/{dataset_name}/{model_type}/leave_one_out/{control_class}_{text_folder}_{batch_size}b'
            
            # leave_one_out_trainingV1(
            #     model_type,
            #     model_name,
            #     base_train_set_file_path,
            #     base_train_model_output_dir,
            #     "test_text.txt",
            #     max_epochs,
            #     batch_size,
            #     save_every=save_every,
            #     gradient_checkpointing=gradient_checkpointing
            # )
            
            gc.collect()
            torch.cuda.empty_cache()
            
            # leave one out training on disease group
            
            base_train_set_file_path = f'../resources/data/input/{dataset_name}/train/{disease_class}/{text_folder}/'
            base_train_model_output_dir = f'../resources/data/output/models/{dataset_name}/{model_type}/leave_one_out/{disease_class}_{text_folder}_{batch_size}b'
            
            # leave_one_out_trainingV1(
            #     model_type,
            #     model_name,
            #     base_train_set_file_path,
            #     base_train_model_output_dir,
            #     "test_text.txt",
            #     max_epochs,
            #     batch_size,
            #     save_every=save_every,
            #     gradient_checkpointing=gradient_checkpointing
            # )
            
            gc.collect()
            torch.cuda.empty_cache()
            
        logger.info("LOO Training completed successfully!")
        
        ### CONTROL on DISEASE TRAIN #########
        
        if do_train:
            model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{best_epoch_control}ep"
        else:
            model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{best_epoch_control}ep"
            
        process_data_folders_multi_gpu(
            model_type=model_type,
            model_dir=model_directory,
            input_base_dir=f"../resources/data/input/{dataset_name}/train/{disease_class}/{text_folder}/",
            output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/{disease_class}/{text_folder}/",
            test_file_name="test_text.txt",
            window=w,
            num_gpus=2,  
            leap=leap
        )
        
        ### DISEASE on CONTROL TRAIN #########
        
        if do_train:
            model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{best_epoch_disease}ep"
        else:
            model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{best_epoch_disease}ep"
            
        process_data_folders_multi_gpu(
            model_type=model_type,
            model_dir=model_directory,
            input_base_dir=f"../resources/data/input/{dataset_name}/train/{control_class}/{text_folder}/",
            output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/{control_class}/{text_folder}/",
            test_file_name="test_text.txt",
            window=w,
            num_gpus=2,  
            leap=leap
        )
        
        # DISEASE on TEST
        
        if do_train:
            model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{best_epoch_disease}ep"
        else:
            model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{disease_class}_{text_folder}_{batch_size}b_{best_epoch_disease}ep"
        
        process_data_folders_multi_gpu(
            model_type=model_type,
            model_dir=model_directory,
            input_base_dir=f"../resources/data/input/{dataset_name}/test/{text_folder}/",
            output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/test/{text_folder}/",
            test_file_name="test_text.txt",
            window=w,
            num_gpus=2,  
            leap=leap
        )
        
        # CONTROL on TEST
        if do_train:
            model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{best_epoch_control}ep"
        else:
            model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/{control_class}_{text_folder}_{batch_size}b_{best_epoch_control}ep"
        
        process_data_folders_multi_gpu(
            model_type=model_type,
            model_dir=model_directory,
            input_base_dir=f"../resources/data/input/{dataset_name}/test/{text_folder}/",
            output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/test/{text_folder}/",
            test_file_name="test_text.txt",
            window=w,
            num_gpus=2,  
            leap=leap
        )
        
        ######################## PPL LEAVE ONE OUT CN #############################

        if do_train:
            model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/leave_one_out/{control_class}_{text_folder}_{batch_size}b_{best_epoch_control}ep"
        else:
            model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/leave_one_out/{control_class}_{text_folder}_{batch_size}b_{best_epoch_control}ep"
        
        process_leave_one_out_multi_gpu(
            model_type=model_type,
            models_base_dir = model_directory,
            input_base_dir=f"../resources/data/input/{dataset_name}/train/{control_class}/{text_folder}/",
            output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/{control_class}/{text_folder}/",
            test_file_name="test_text.txt",
            window=w, 
            num_gpus=2,
            leap=leap
        )
        
        ######################## PPL LEAVE ONE OUT DISEASE ########################

        if do_train:
            model_directory = f"../resources/data/output/models/{dataset_name}/{model_type}/leave_one_out/{disease_class}_{text_folder}_{batch_size}b_{best_epoch_disease}ep"
        else:
            model_directory = f"../resources/data/output/submission_models/{dataset_name}/{model_type}/leave_one_out/{disease_class}_{text_folder}_{batch_size}b_{best_epoch_disease}ep"
            
        process_leave_one_out_multi_gpu(
            model_type=model_type,
            models_base_dir = model_directory,
            input_base_dir=f"../resources/data/input/{dataset_name}/train/{disease_class}/{text_folder}/",
            output_base_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/{disease_class}/{text_folder}/",
            test_file_name="test_text.txt",
            window=w, 
            num_gpus=2,
            leap=leap
        )
        
        logger.info("Perplexity production on Train and Test sets using best models completed successfully!")
        
        
        # Phase 4: Classification
        logger.info("=" * 50)
        logger.info("PHASE 4: CLASSIFICATION")
        logger.info("=" * 50)
        
        cnep = best_epoch_control
        adep = best_epoch_disease
        
        # Create results directory if it doesn't exist
        results_dir = f"../resources/data/results/test_{model_type}_{dataset_name}_{strategy_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            classify_all_patients(
                test_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/test/{text_folder}/",
                control_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/cn/{text_folder}/",
                disease_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/ad/{text_folder}/",
                control_model=f"{control_class}_{text_folder}_{batch_size}b_{cnep}ep",
                disease_model=f"{disease_class}_{text_folder}_{batch_size}b_{adep}ep",
                use_median=False,
                two_sigma=False,
                output_csv=f"{results_dir}/risultati_classificazione_TEST_{cnep}cn_{adep}ad_ep_{batch_size}b_{text_folder}_{w}w.csv"
            )
            
            classify_all_patients_with_plot(
                test_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/test/{text_folder}/",
                control_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/cn/{text_folder}/",
                disease_dir=f"../resources/data/output_{model_type}/{dataset_name}_w{w}_l{leap}/train/ad/{text_folder}/",
                control_model=f"{control_class}_{text_folder}_{batch_size}b_{cnep}ep",
                disease_model=f"{disease_class}_{text_folder}_{batch_size}b_{adep}ep",
                use_median=False,
                two_sigma=False,
                label_file=f"../resources/data/input/{dataset_name}/test/labels.csv",
                output_png=f"{results_dir}/plot_TEST_{text_folder}_{batch_size}b_{cnep}cn_{adep}ad_ep_{w}w.png"
            )
            
            calculate_f1(results_dir, f"../resources/data/input/{dataset_name}/test/labels.csv")
            
            logger.info("Classification completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            raise
        
        # Final results
        logger.info("=" * 50)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during experiment: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())