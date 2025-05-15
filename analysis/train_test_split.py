import os
import json
import re


successful_task_list = "~/nyuctf_agents/analysis/successful_tasks_lists/intercode_ctf_Qwen2.5-Coder-32B-Instruct.txt"
benchmark_task_list = "~/SWE-agent-datasets/Intercode_CTF/intercode_dataset.json"
benchmark_base_dir = "~/SWE-agent-datasets/Intercode_CTF/"

with open(successful_task_list, "r") as f:
    successful_tasks = f.readlines()

# for task in successful_tasks:
#     match = re.search(r'challenge_(\d+)\.json', task)
#     challenge_number = match.group(1) if match else None
    


successful_tasks_ids = [match.group(1) for task in successful_tasks if (match := re.search(r'challenge_(\d+)\.json', task)) is not None]

with open(benchmark_task_list, "r") as f:
    benchmark_tasks = json.load(f)
    benchmark_tasks = benchmark_tasks.keys()

benchmark_tasks_ids = [match.group(1) for task in benchmark_tasks if (match := re.search(r'Challenge #(\d+)', task)) is not None]

    
with open(benchmark_task_list, "r") as f:
    benchmark_tasks = json.load(f)
unsolved_tasks_ids = set(benchmark_tasks_ids) - set(successful_tasks_ids)
unsolved_tasks_path = [benchmark_base_dir + benchmark_tasks[f"Challenge #{task_id}"]['path'] for task_id in unsolved_tasks_ids]
for path in unsolved_tasks_path:
    print(path)
with open("unsolved_tasks.txt", "w") as f:
    for path in unsolved_tasks_path:
        f.write(path + "\n")

