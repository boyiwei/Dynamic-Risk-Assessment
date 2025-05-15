import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse


def truncate_input_to_max_tokens(tokenizer, model_input: str, system_prompt: str, max_input_tokens) -> str:
        num_input_tokens = len(tokenizer(model_input).input_ids)
        num_system_prompt_tokens = len(tokenizer(system_prompt).input_ids)
        remaining_input = model_input[len(system_prompt) + 17:] # TODO(wby) only useful for qwen model, here +17 means we remove <|im_start|>user\n at the beginning<
        truncation_alert = "\n...TRUNCATED...\n"
        num_tokens_in_truncation_alert = len(tokenizer(truncation_alert).input_ids) #self._get_num_tokens(truncation_alert)
        truncated = False
        
        truncated_tokens = tokenizer(remaining_input).input_ids
        if num_input_tokens  >= max_input_tokens - num_tokens_in_truncation_alert - num_system_prompt_tokens:
            print(f"Number of input tokens ({num_input_tokens}) exceeds max tokens ({max_input_tokens}). Truncating input.")
            tokens = tokenizer(remaining_input).input_ids
            tokens_to_keep = max_input_tokens - num_tokens_in_truncation_alert
            half_tokens_to_keep = tokens_to_keep // 2
            beginning_tokens = tokens[:half_tokens_to_keep]
            end_tokens = tokens[-half_tokens_to_keep:]
            truncated_tokens = (
                beginning_tokens + tokenizer(truncation_alert).input_ids + end_tokens
            )
            truncated = True
        
        truncated_input = tokenizer.decode(truncated_tokens)[:-11] # TODO(wby) only useful for qwen model, here -11 means we remove <im_end>\n at the end
        return truncated_input, truncated
    
    

def convert_traj_to_chat(file_list, format, tokenizer):
    # find all json file insider this directory
    # read the json file
    messages = []
    
    conv = []
    with open(file_list, 'r') as f:
        file_dirs = f.readlines()
    for path in tqdm(file_dirs):
        with open(path.strip(), 'r') as f:
            data = json.load(f)
            messages.append({"messages": data['messages']})
    # write to a jsonl file
    json_file_name = os.path.basename(file_list).split('.txt')[0] + f"_{format}.jsonl"
    if format == 'multi-turn':
        with open(json_file_name, 'w') as f:
            for message in messages:
                f.write(json.dumps(message) + '\n')
    elif format == 'single-turn':
        for sub_messages in messages:
            user_index = [i for i, message in enumerate(sub_messages['messages']) if message['role'] == 'assistant'] # here we first locate the index where the model output the response, and collect the conversation history before that
            prompts = [sub_messages['messages'][:i] for i in user_index]
            # prompts = [messages[:2 * i] for i in range(1, len(messages) // 2 + 1)]
            responses = [message for message in sub_messages['messages'] if message['role'] == 'assistant']
            for prompt, response in zip(prompts, responses):
                # truncate the input to max tokens
                formatted_input = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_system_prompt = tokenizer.apply_chat_template(
                    [prompt[0]],
                    tokenize=False,
                    add_generation_prompt=False
                )
                max_input_tokens=160000
                truncated_prompt, truncated = truncate_input_to_max_tokens(tokenizer, formatted_input, formatted_system_prompt, max_input_tokens)
                if truncated == False: # For NYU CTF agent, only add the messages that are not truncated.
                    conv.append({"messages":[prompt[0], {"role": "user", "content": truncated_prompt}, response]})
            
        with open(json_file_name, 'w') as f:
            for message in conv:
                f.write(json.dumps(message) + '\n')
        
    
                    
                
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--file_dir', type=str, default='~/dynamic_risk_assessment/analysis/successful_tasks_lists/successful_tasks_intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_20_train_round4.txt')
    args.add_argument('--format', type=str, default='single-turn')
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    args = args.parse_args()
    convert_traj_to_chat(args.file_dir, args.format, tokenizer)
                    