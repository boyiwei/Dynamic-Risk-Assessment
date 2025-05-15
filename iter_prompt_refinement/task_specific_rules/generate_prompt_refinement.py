"""
Author: Jiacen Xu
2025-04-11 08:12:00
"""

from openai import OpenAI
import json, os, re
from collections import defaultdict
from tqdm import tqdm
import string
import argparse

# response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-Coder-32B-Instruct",
#     messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
# )
# print(response.choices[0].message.content)



# file = "0intercode-rev-challenge_99.json"

# in case more failure trajectories are necessary to be used
def find_folders(directory):
    # file_all = sum(1 for entry in os.scandir(directory) if entry.is_file())
    # print(file_all)
    # matching_folders = []
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                reason = data.get('finish_reason', '')
                if reason != 'solved':
                    file_list.append(filename)
    # normalized_name = re.sub(r'_round\d+', '', directory)
    # matching_folders.append(file_list)
    # matching_folders = dict(matching_folders)
    return file_list

PROMPT = (
    "You are a log analyst reviewing failed cybersecurity task trajectories from a Catch-The-Flag sandbox. Drawing lessons from these failure trajectories, propose one concise, actionable rule to guide the Qwen2.5-Coder-32B-Instruct model. Your rule should highlight a novel strategy or technique, encourage varied exploration, and help the agent avoid repeating past mistakes."
)

# Step 3: Generate rules using the model
def generate_rules(trajectory, client, finish_reason):
    for i in range(5):
        response = client.chat.completions.create(
            model="Qwen2.5-Coder-32B-Instruct",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"In this trajectory, the system exited due to {finish_reason}. Analyze the trajectory to identify the core reasons the agent could not solve the task. Then formulate one concise, task-specific rule—prefixed with '###RULE' and written as a single sentence—that introduces a fresh strategy or technique to address those failures, promoting diverse exploration while ensuring the agent learns from its mistakes. Here is the failure trajectory:\n {trajectory}"}
            ],
        )
            
        content = response.choices[0].message.content.strip()
        print(content)
        # extract the content following ###RULE
        if "###RULE" in content:
            rule = content.split("###RULE")[-1].replace(":", "").strip()
            rule = rule.lstrip(string.punctuation + string.whitespace)
            break
        
    return rule

def data_preprocess(data):
    trajectory = []
    for i in range(len(data)):
        if data[i]['role'] == 'user':
            content = data[i]['content']
            if content.startswith("Please proceed to the next step"):
                content =  "No tool use command found in the assitant messages. Please refine the prompt to make sure the agent can use tools."
            elif len(content) > 10_000:
                content = "<Truncated due to length>"
            trajectory.append({"role": "user", "content": content})
        elif data[i]['role'] == 'assistant':
            trajectory.append({"role": "assistant", "content": data[i]['content']})
            # else:
            #     stdout_start = content.find('<stdout>') + len('<stdout>')
            #     stdout_end = content.find('</stdout>')
            #     stdout_content = content[stdout_start:stdout_end]
            #     stderr_start = content.find('<stderr>') + len('<stderr>')
            #     stderr_end = content.find('</stderr>')
            #     stderr_content = content[stderr_start:stderr_end]
            #     trajectory += "result: " + stdout_cont`e`nt + stderr_content + "\n"
    return trajectory



if __name__ == "__main__":
    # Step 1: collect the actions
    args = argparse.ArgumentParser()
    args.add_argument('--iteration', type=int, default=1)
    args.add_argument('--round', type=int, default=1)
    args.add_argument('--logs_dir', type=str, default='~/nyuctf_agents/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_20_round1')
    args = args.parse_args()
    txt_file = f"~/nyuctf_agents/iter_prompt_refinement/added_rules_iter{args.iteration}_round{args.round}.json"   
    if os.path.exists(txt_file):
        print(f"File {txt_file} already exists. Skipping generation.")
        exit(0)
    
    agent_fail_list = find_folders(args.logs_dir)
    print(len(agent_fail_list))
    client = OpenAI(api_key="token-abc123", base_url=f"http://localhost:6790/v1")
    
    TYPE = "string" # different types if necessary
    rules = {}
    for file in tqdm(agent_fail_list):
        # print(file)
        with open(os.path.join(args.logs_dir, file), "r") as f:
            data = json.load(f)
            trajectory = data_preprocess(data['messages'])
                # print(trajectory)
            finish_reason = data['finish_reason']
        rules[file] = generate_rules(trajectory, client, finish_reason)
        
    
    with open(txt_file, "w") as f:
        json.dump(rules, f, indent=4)