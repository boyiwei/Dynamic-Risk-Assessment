import json
import os
import argparse




def compute_time(args):
    time_list = []
    for round in range(1, 11):
    
        file_dir = f"~/dynamic_risk_assessment/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_20_round{round}/"
        # find all json files in the directory
        json_files = [f for f in os.listdir(file_dir) if f.endswith('.json')]
        
        task_mask = "~/dynamic_risk_assessment/analysis/train_tasks.txt"
        unsolvable_task_mask = "~/dynamic_risk_assessment/analysis/unsolvable_tasks.txt"
        with open(task_mask, 'r') as f:
            task_mask_list = [line.strip() for line in f.readlines()]
        with open(unsolvable_task_mask, 'r') as f:
            unsolvable_task_mask_list = [line.strip() for line in f.readlines()]
        for json_file in json_files:
            keyword = json_file.split('.')[0].strip()
            if keyword in unsolvable_task_mask_list:
                continue
            if (args.train_set and keyword not in task_mask_list) or (args.test_set and keyword in task_mask_list):
                continue

            with open(os.path.join(file_dir, json_file), "r") as f:
                data = json.load(f)
                time_list.append(data['runtime']['total'])
        # average time
    avg_time = sum(time_list) / (10 * 60)
    print(f"len time_list: {len(time_list)}, average time: {avg_time} mins")
    


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--train_set", action="store_true", help="whether to use train set")
    args.add_argument("--test_set", action="store_true", help="whether to use test set")
    args = args.parse_args()
    compute_time(args)