from trl import DataCollatorForCompletionOnlyLM

def get_model(model_name_or_path, model_kwargs, model_family='llama2', padding_side = "right"):
    
    if model_family == 'qwen2':
        from .model_families.qwen2 import initializer as qwen2_initializer
        return qwen2_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    else:
        raise ValueError(f"model_family {model_family} not maintained")
    

def get_training_string_formatter(tokenizer, model_family):

    if model_family == 'qwen2':
        from .model_families.qwen2 import get_training_string_formatter as qwen2_get_training_string_formatter
        return qwen2_get_training_string_formatter(tokenizer)
    else:
        raise ValueError(f"model_family {model_family} not maintained")


def get_data_collator(tokenizer, response_template = None, model_family = 'llama2', ntp = False):
    
    if response_template is None:

        if model_family == 'qwen2':
            from finetuning_buckets.models.model_families.qwen2 import CustomDataCollator as Qwen2CustomDataCollator
            return Qwen2CustomDataCollator(tokenizer=tokenizer, ntp=ntp)
        else:
            raise ValueError("response_template or dataset_name should be provided")

    else:
    
        return DataCollatorForCompletionOnlyLM(response_template=response_template, 
                                                    tokenizer=tokenizer)