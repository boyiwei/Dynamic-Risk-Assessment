import json
import os


def remove_imcomplete_finish(file_dir):
    # find all the json file inside the directory
    # read the json file
    number_of_files = 0
    for path in os.listdir(file_dir):
        with open(os.path.join(file_dir, path), 'r') as f:
            data = json.load(f)
            messages = data["messages"]
            if (data["finish_reason"] == 'unknown' and len(messages) <= 2) or (data['finish_reason'] == 'user_cancel'):
                print(f"File: {path} has finish_reason: unknown and len(messages) <= 2, removing it.")
                number_of_files += 1
                os.remove(os.path.join(file_dir, path))
    print(f"Number of files removed: {number_of_files}")
    
                
if __name__ == "__main__":
    for i in range(1, 10):
        for task in ["intercode_ctf"]:
            remove_imcomplete_finish(f'~/dynamic_risk_assessment/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_50_round{i}')    
    # remove_imcomplete_finish('~/dynamic_risk_assessment/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct-ft_ft_10_lora_maxiter_20_round3/')
    