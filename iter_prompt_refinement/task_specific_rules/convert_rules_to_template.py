import json
import subprocess
import shutil
import os
import argparse

def convert_rules_to_template(iteration, round):
    
    with open(f"~/dynamic_risk_assessment/iter_prompt_refinement/added_rules_iter{iteration}_round{round}.json", "r") as f:
        rules = json.load(f)
        
    with open("~/SWE-agent-datasets/Intercode_CTF/intercode_dataset.json", "r") as f:
        task_list = json.load(f)
    task_list = task_list.keys()
    task_list = [("challenge_" + elem.split(" ")[-1]).replace("#", "") for elem in task_list] # only filter the task id
    for task in task_list:
        for key in rules.keys():
            # Extract the challenge part from the key
            challenge_start = key.find("challenge")
            challenge_part = key[challenge_start:].split(".")[0]
            if challenge_part == task:
                # generate the customized prompt template
                # Check if the destination directory exists and remove it if it does
                if iteration == 1:
                    previous_dir = "~/dynamic_risk_assessment/llm_ctf/prompts/templates/default"
                else:
                    previous_dir = f"~/dynamic_risk_assessment/llm_ctf/prompts/templates/iteration_{iteration - 1}_{task}"
                if os.path.exists(previous_dir): # we only select the one that falied in the previous iteration, otherwise the task has already completed.
                    orig_dir = previous_dir
                else:
                    continue
                destination_dir = f"~/dynamic_risk_assessment/llm_ctf/prompts/templates/iteration_{iteration}_{task}"
                if os.path.exists(destination_dir):
                    shutil.rmtree(destination_dir)
                subprocess.run(["cp", "-r", orig_dir, destination_dir])
                tool_use_file = f"{destination_dir}/tool_use.md.jinja2"
                
                # Read existing content
                content = ""
                if os.path.exists(tool_use_file):
                    with open(tool_use_file, "r") as f:
                        content = f.read()
                
                # Count existing bullet points (looking for lines starting with a number followed by a period)
                bullet_points = 0
                for line in content.split('\n'):
                    if line.strip() and line.strip()[0].isdigit() and '. ' in line.strip():
                        bullet_points += 1
                
                # Open file for appending
                with open(tool_use_file, "a") as f:
                    # Add the header if no bullet points exist
                    if bullet_points == 0:
                        f.write("\nMake sure to follow the rules below strictly.\n")
                    
                    # Add the new rule with the next bullet point number
                    f.write(f"{bullet_points + 1}. {rules[key]}\n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rules to template")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number")
    parser.add_argument("--round", type=int, default=1, help="Round number")
    args = parser.parse_args()
    
    convert_rules_to_template(args.iteration, args.round)
    
                        
        
    
    
    