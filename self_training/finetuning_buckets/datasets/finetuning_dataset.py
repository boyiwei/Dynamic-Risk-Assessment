import inspect
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from finetuning_buckets import models
import pandas as pd
import random

# Registry for dataset functions
registry = {}


# Decorator to register dataset functions
def register_dataset(func):
    registry[func.__name__] = func
    return func


# Helper function to apply a formatter to a dataset
def apply_formatter(dataset, formatter):
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)
    return dataset



@register_dataset
def ctf_intercode_nyuagent_singleturn_train(string_formatter, split='train'):
    dataset = load_dataset("json", data_files="~/self_training/finetuning_buckets/datasets/safety_datasets/successful_tasks_intercode_ctf_Qwen2_single-turn_subsettrain.jsonl", split=split)
    # to oai format
    def formatter(example):
        formatted_example = {'messages': example['messages']} # we don't use system prompt by default to align with the cybench's evaluation pipeline.
        return formatted_example
    
    dataset = apply_formatter(dataset, formatter)
    # to string format
    dataset = apply_formatter(dataset, string_formatter)
    print("first item in dataset is" ,dataset[0])
    return dataset



def get_dataset(dataset_name, tokenizer, model_family = 'llama2', split='train', safety_augmentation=False, max_num_samples=-1):

    if dataset_name not in registry:
        raise ValueError(f"dataset_name {dataset_name} not defined in the registry")
    
    string_formatter = models.utils.get_training_string_formatter(tokenizer, model_family)
    all_params = {
        'string_formatter': string_formatter,
        'split': split,
        'model_family': model_family,
        'safety_augmentation': safety_augmentation,
        'max_num_samples': max_num_samples
    }

    dataset_func = registry[dataset_name]
    signature = inspect.signature(dataset_func)
    dataset_params = list(signature.parameters.keys())
    input_params = {k: all_params[k] for k in all_params if k in dataset_params}
    dataset = dataset_func(**input_params)
    return dataset