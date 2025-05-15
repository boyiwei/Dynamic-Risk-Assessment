import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os
import pandas as pd


def successful_task_freq(prefix):
    # find all files with the model name
    valid_files = []
    # prefix = f'successful_tasks_{task_name}_{model_name}'
    print(prefix)
    for root, dirs, files in os.walk('analysis/successful_tasks_lists'):
        for file in files:
            if file.startswith(prefix):
                valid_files.append(os.path.join(root, file))
    task_count = {}
    print(valid_files)
    for file in valid_files:
        with open(file, 'r') as f:
            dirs = f.readlines()
            for i in range(len(dirs)):
                dir = dirs[i]
                simplified_dir = dir.split('/')[-1].strip()
                # dirs = [dir.split('/')[-1].strip() for dir in dirs]
                # for dir in dirs:
                if simplified_dir not in task_count:
                    task_count[simplified_dir] = {}
                    task_count[simplified_dir]['count'] = 1
                    task_count[simplified_dir]['file'] = [dir]
                else:
                    task_count[simplified_dir]['count'] += 1
                    task_count[simplified_dir]['file'].append(dir)
    # draw histogram based on the task_count
    # task_freq = pd.DataFrame(task_count.items(), columns=['task', 'freq'])
    sorted_task_count = sorted(task_count.items(), key=lambda item: item[1]['count'], reverse=True)
    # task_freq = pd.DataFrame(sorted_task_count, columns=['task', 'freq'])
    # fig = task_freq.plot(kind='bar', x='task', y='freq', figsize=(20, 10))
    # fig.get_figure().savefig(f'successful_tasks_{task_name}_{model_name}_freq.png')
    # task_freq.to_csv(f'successful_tasks_{task_name}_{model_name}_freq.csv', index=False)
    # return sorted_task_count[-25:]
    return sorted_task_count



def successful_task_freq_intercode(prefix):
    # find all files with the model name
    valid_files = []
    # prefix = f'successful_tasks_{task_name}_{model_name}'
    print(prefix)
    for root, dirs, files in os.walk('analysis/successful_tasks_lists'):
        for file in files:
            if file.startswith(prefix):
                valid_files.append(os.path.join(root, file))
    task_count = {}
    unsolvable_ids = [95, 66, 29, 28, 87, 1, 89, 56, 88, 55]
    for i in range(100):
        if i in unsolvable_ids:
            continue
        task_name = f'Challenge #{i}'
        task_count[task_name] = {'count': 0, 'file': []}
    
    print(valid_files)
    for file in valid_files:
        with open(file, 'r') as f:
            dirs = f.readlines()
            for i in range(len(dirs)):
                dir = dirs[i]
                simplified_dir = dir.split('/')[-1].strip()
                task_name = "Challenge #" + simplified_dir.split('challenge_')[-1].split('.')[0]

                task_count[task_name]['count'] += 1
                task_count[task_name]['file'].append(dir)
    # draw histogram based on the task_count
    # task_freq = pd.DataFrame(task_count.items(), columns=['task', 'freq'])
    sorted_task_count = sorted(task_count.items(), key=lambda item: item[1]['count'], reverse=True)
    # task_freq = pd.DataFrame(sorted_task_count, columns=['task', 'freq'])
    # fig = task_freq.plot(kind='bar', x='task', y='freq', figsize=(20, 10))
    # fig.get_figure().savefig(f'successful_tasks_{task_name}_{model_name}_freq.png')
    # task_freq.to_csv(f'successful_tasks_{task_name}_{model_name}_freq.csv', index=False)
    # return sorted_task_count[-25:]
    return sorted_task_count


def get_low_confidence_task_list(prefix, weighted_tasks):
    valid_files = []
    # prefix = f'successful_tasks_{task_name}_{model_name}'
    print(prefix)
    for root, dirs, files in os.walk('analysis/successful_tasks_lists'):
        for file in files:
            if file.startswith(prefix):
                valid_files.append(os.path.join(root, file))
                
    difficulty_weight = {}
    for file in valid_files:
        with open(file, 'r') as f:
            difficulty_weight[file] = 0
            dirs = f.readlines()
            for i in range(len(dirs)):
                dir = dirs[i]
                simplified_dir = dir.split('/')[-1].strip()
                difficulty_weight[file] += weighted_tasks[simplified_dir]
            
            difficulty_weight[file] = difficulty_weight[file] / len(dirs)
    
    # sort difficulty_weight
    difficulty_weight = sorted(difficulty_weight.items(), key=lambda item: item[1], reverse=True)
    
    return difficulty_weight


def gen_train_test_split(task_freq):
    tasks_df = pd.DataFrame([
    {"task_id": task_id, "difficulty": task_data["count"]} 
    for task_id, task_data in task_freq
])

    # Create difficulty bins (quintiles or another reasonable number of bins)
    # This creates balanced groups based on difficulty distribution
    tasks_df["difficulty_bin"] = pd.qcut(tasks_df["difficulty"], q=5, labels=False)

    # Use stratified split to maintain difficulty distribution
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, test_idx = next(split.split(tasks_df, tasks_df["difficulty_bin"]))

    # Get the task IDs for train and test sets
    train_task_ids = tasks_df.iloc[train_idx]["task_id"].tolist()
    test_task_ids = tasks_df.iloc[test_idx]["task_id"].tolist()

    # # Map back to file paths
    # train_paths = [benchmark_base_dir + benchmark_tasks[f"Challenge #{task_id}"]["path"] 
    #             for task_id in train_task_ids if f"Challenge #{task_id}" in benchmark_tasks]
    # test_paths = [benchmark_base_dir + benchmark_tasks[f"Challenge #{task_id}"]["path"] 
    #             for task_id in test_task_ids if f"Challenge #{task_id}" in benchmark_tasks]

    # Save to files
    with open("train_tasks.txt", "w") as f:
        for path in train_task_ids:
            f.write(path + "\n")

    with open("test_tasks.txt", "w") as f:
        for path in test_task_ids:
            f.write(path + "\n")

    print(f"Train set: {len(train_task_ids)} tasks")
    print(f"Test set: {len(test_task_ids)} tasks")
    
    

if __name__ == "__main__":
    model_name = "Qwen2.5-Coder-32B-Instruct_maxiter_20_round" 
    task_name = "intercode_ctf"
    prefix = f'successful_tasks_{task_name}_{model_name}'
    if task_name == "intercode_ctf":
        task_freq = successful_task_freq_intercode(prefix)
    else:
        task_freq = successful_task_freq(prefix)
        
    
    gen_train_test_split(task_freq)
    # low_confidence_task_lists = []
    # weighted_tasks = {}
    # for key, value in task_freq:
    #     low_confidence_task_lists.append(value['file'][0])
    #     weighted_tasks[key] = value['count']
        
    # difficulty_weight = get_low_confidence_task_list(prefix, weighted_tasks)
        
    
    # with open(f'successful_tasks_lists/low_confidence_{task_name}_{model_name}.txt', 'w') as f:
    #     for task in low_confidence_task_lists:
    #         f.write(task)
    
    
    # with open(f'successful_tasks_lists/{task_name}_{model_name}.txt', 'w') as f:
    #     for task in low_confidence_task_lists:
    #         f.write(task)
        
    print(task_freq)