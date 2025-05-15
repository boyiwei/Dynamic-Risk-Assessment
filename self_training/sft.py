import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
from finetuning_buckets.datasets import finetuning_dataset
from finetuning_buckets import models
import torch
import os


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="ctf_ft")
    # train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    dataset_name: Optional[str] = field(default='ctf_intercode_nyuagent_singleturn')
    model_family: Optional[str] = field(default='qwen2')
    max_num_samples: Optional[int] = field(default=-1)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ["WANDB_MODE"] = "offline"

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    logging.info(f"Local rank: {local_rank}")
    logging.info(f"Global rank: {global_rank}")
    
    # num_devices = torch.cuda.device_count()
    # print(num_devices)
    # if num_devices == 0:
    #     raise RuntimeError("No GPUs available")
    # # Map the local rank to a valid GPU index
    # # device_index = local_rank % num_devices
    # torch.cuda.set_device(local_rank)
    # print(f"Setting CUDA device to {local_rank}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False) 
    original_tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    # dataset = load_dataset(config.train_file_path)
    dataset = finetuning_dataset.get_dataset(
        config.dataset_name,
        tokenizer,
        model_family=config.model_family,
        split='train',
        max_num_samples=config.max_num_samples
    )  # dataset is in raw string format

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user\n"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    # collator = trl.DataCollatorForCompletionOnlyLM(
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    #     mlm=False
    # )
    collator = models.utils.get_data_collator(tokenizer, model_family=config.model_family, ntp=False)
    
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset,
        args=args,
        data_collator=collator,
    )

    trainer.train()
    # if global_rank == 0:
    print("Saving model...")
    trainer.save_model(output_dir=args.output_dir)
    original_tokenizer.save_pretrained(args.output_dir)
    
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
