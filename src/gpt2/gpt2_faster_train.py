#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from functools import partial


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.trainer_callback import TrainerCallback

from torch.utils.data import Dataset
import torch


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)

# Disable logging from libraries
# transformers.logging.set_verbosity_error()


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


# class LazyLanguageModelingDataset(Dataset):
#     def __init__(self, file_path, tokenizer, block_size):
#         self.tokenizer = tokenizer
#         self.block_size = block_size
#         self.examples = []

#         with open(file_path, encoding="utf-8") as f:
#             lines = [line.strip() for line in f if len(line.strip()) > 0]

#         # Concatenazione e chunking on init (puoi anche spostare questa logica in __getitem__ se il file è enorme)
#         batch_encoding = tokenizer(" ".join(lines), return_attention_mask=False, return_tensors="pt")
#         input_ids = batch_encoding["input_ids"].squeeze()

#         total_length = (len(input_ids) // block_size) * block_size
#         input_ids = input_ids[:total_length]

#         self.examples = input_ids.view(-1, block_size)

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         return {"input_ids": self.examples[i], "labels": self.examples[i]}
    

class LazyLanguageModelingDatasetV1(Dataset):
    def __init__(self, file_path, tokenizer, block_size, loo_folder=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.speeches = []
        self.loo_folder = loo_folder

        # Step back to the folder that contains the subjects folders
        base_dir = os.path.dirname(file_path)

        for root, _, files in os.walk(base_dir):
            folder_name = os.path.basename(root)

            # Skip folder if it matches the leave-one-out target
            if self.loo_folder and folder_name == self.loo_folder:
                print(f"Skipping folder: {folder_name}")
                continue

            for file in files:
                if file == "test_text.txt":
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            tokenized = tokenizer(
                                text,
                                max_length=block_size,
                                truncation=True,
                                return_attention_mask=False,
                                return_tensors="pt",
                            )
                            self.speeches.append(tokenized["input_ids"].squeeze())

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self, idx):
        input_ids = self.speeches[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}
    
    
  
# GIANNI
def collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
    }
  


class SaveCheckpointCallback(TrainerCallback):
    """Callback che salva il modello a intervalli di epoche specifici in cartelle separate"""
    
    def __init__(self, save_every_epochs, output_dir_base, loo_folder, tokenizer):
        self.save_every_epochs = save_every_epochs
        self.output_dir_base = output_dir_base
        self.loo_folder = loo_folder
        self.tokenizer = tokenizer
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Salva il modello ogni `save_every_epochs` epoche"""
        epoch = state.epoch
        # Verifica se è il momento di salvare (ogni save_every_epochs epoche)
        if int(epoch) % self.save_every_epochs == 0:
            # Crea una directory per questo checkpoint
            checkpoint_dir = f"{self.output_dir_base}_{int(epoch)}ep"
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Se llo_folder è specificato, crea una sottocartella
            if not (self.loo_folder == None):
                checkpoint_dir = os.path.join(checkpoint_dir, self.loo_folder)
                os.makedirs(checkpoint_dir, exist_ok=True)
            # Salva il modello e il tokenizer
            kwargs['model'].save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(f"\nModello salvato all'epoca {int(epoch)} in: {checkpoint_dir}")
        return control


# def set_global_seed(seed):
#     import random
#     import numpy as np
#     import torch
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def train_model(args, save_every_epochs=0, loo_folder=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(args) == 2 and args[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to error on all processes
    logger.setLevel(logging.ERROR)
    

    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if is_main_process(training_args.local_rank):
    #     transformers.utils.logging.set_verbosity_info()
    #     transformers.utils.logging.enable_default_handler()
    #     transformers.utils.logging.enable_explicit_format()
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # set_global_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    
    # if data_args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    #     if "validation" not in datasets.keys():
    #         datasets["validation"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[:{data_args.validation_split_percentage}%]",
    #             cache_dir=model_args.cache_dir,
    #         )
    #         datasets["train"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[{data_args.validation_split_percentage}%:]",
    #             cache_dir=model_args.cache_dir,
    #         )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #     extension = (
    #         data_args.train_file.split(".")[-1]
    #         if data_args.train_file is not None
    #         else data_args.validation_file.split(".")[-1]
    #     )
    #     if extension == "txt":
    #         extension = "text"
    #     datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # GIANNI
    tokenizer.pad_token = tokenizer.eos_token
    

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            device_map = 'auto',
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config=config, device_map = 'auto',)

    model.resize_token_embeddings(len(tokenizer))
    
    # #model.to("cuda:"+str(device))
    # # import torch
    # # # torch.cuda.empty_cache()
    # # model.to("cpu")
    # # model = torch.nn.DataParallel(model)

    # # Preprocessing the datasets.
    # # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = datasets["train"].column_names
    # else:
    #     column_names = datasets["validation"].column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]

    # # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # def tokenize_function(examples):
    #     with CaptureLogger(tok_logger) as cl:
    #         output = tokenizer(examples[text_column_name])
    #     # clm input could be much much longer than block_size
    #     if "Token indices sequence length is longer than the" in cl.out:
    #         tok_logger.warning(
    #             "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
    #         )
    #     return output

    # tokenized_datasets = datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )

    # if data_args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > 1024:
    #         logger.warning(
    #             f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
    #             "Picking 1024 instead. You can change that default value by passing --block_size xxx."
    #         )
    #     block_size = 1024
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)

    # # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # # to preprocess.
    # #
    # # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # lm_datasets = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )

    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        #    raise ValueError("--do_train requires a train dataset")
        
        #train_dataset = LazyLanguageModelingDataset(data_args.train_file, tokenizer, block_size=1024)
        
        # GIANNI
        train_dataset = LazyLanguageModelingDatasetV1(data_args.train_file, tokenizer, block_size=1024, loo_folder=loo_folder)

        
        #if data_args.max_train_samples is not None:
        #    train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        # if "validation" not in tokenized_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        
        #eval_dataset = LazyLanguageModelingDataset(data_args.validation_file, tokenizer, block_size=1024)
        
        # GIANNI
        eval_dataset = LazyLanguageModelingDatasetV1(data_args.validation_file, tokenizer, block_size=1024, loo_folder=loo_folder)
        

        # if data_args.max_eval_samples is not None:
        #     eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Initialize our Trainer
    
    checkpoint_callback = None
    if training_args.do_train and save_every_epochs != 0:
        save_every = save_every_epochs
        output_base = training_args.output_dir.rstrip('/')
        checkpoint_callback = SaveCheckpointCallback(save_every, output_base, loo_folder, tokenizer)
    
    # GIANNI
    data_collator = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    # GIANNI
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[checkpoint_callback] if checkpoint_callback else None,
    )
    
    #cancella il modello dalla cartella in training_args.output_dir
    if os.path.exists(training_args.output_dir):
        print(f"Deleting model directory: {training_args.output_dir}")
        for root, dirs, files in os.walk(training_args.output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub()
        
    #cancella il modello dalla cartella in training_args.output_dir
    if os.path.exists(training_args.output_dir):
        print(f"Deleting model directory: {training_args.output_dir}")
        for root, dirs, files in os.walk(training_args.output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(training_args.output_dir)
    else:
        print(f"Model directory {training_args.output_dir} does not exist. Nothing to delete.")
