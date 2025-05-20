import argparse
import os
import glob
import json
from nyuctf.dataset import CTFDataset
from nyuctf.challenge import CTFChallenge
from tqdm import tqdm
import numpy as np



dataset2root = {
    'nyu_ctf_test': {"root_dir": "ctf-datasets/NYU_CTF_Bench/", "json_file": "ctf-datasets/NYU_CTF_Bench/test_dataset.json"},
    'nyu_ctf_train': {"root_dir": "ctf-datasets/NYU_CTF_Bench/", "json_file": "ctf-datasets/NYU_CTF_Bench/development_dataset.json"},
    'cybench': {"root_dir": "ctf-datasets/cybench/", "json_file": "ctf-datasets/cybench/cybench.json"},
    'intercode_ctf': {"root_dir": "ctf-datasets/Intercode_CTF/", "json_file": "ctf-datasets/Intercode_CTF/intercode_dataset.json"},
}

def grade_benchmark(task_name):
    log_dir = "logs/" + task_name
    json_files = glob.glob(os.path.join(log_dir, "*.json"))
    task_num = len(json_files)
    acc_count = 0
    accuracy = 0.0
    success_tasks = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        if data["finish_reason"] == 'solved':
            acc_count += 1
            success_tasks.append(json_file)
    accuracy = acc_count / task_num
    return task_num, acc_count, accuracy, success_tasks


def pass_at_k(n, c, k):
    """
    n: total number of iterations (for single task)
    c: number of tasks solved (for single task)
    k: k in pass@k"""
    if n - c < k:
        return 1.0
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_pass_k_bootstrap(args, k):
    benchmark_name = args.task_name
    model_name = args.model_name
    max_iterations = args.max_iterations
    n_rounds = args.n_rounds
    task_json_file = dataset2root[benchmark_name]['json_file']
    
    with open(task_json_file, "r") as f:
        data = json.load(f)
    if args.train_set:
        train_set_path = "analysis/train_tasks.txt"
        with open(train_set_path, "r") as f:
            task_names = [line.strip() for line in f.readlines()]
    elif args.test_set:
        test_set_path = "analysis/test_tasks.txt"
        with open(test_set_path, "r") as f:
            task_names = [line.strip() for line in f.readlines()]
    else:
        raise NotImplementedError("Only train and test set evaluation is supported")
    tasks_solved_count = {}
    for task in task_names:
        tasks_solved_count[task] = 0
    pass_arrays = None
    for i in range(1, n_rounds + 1):
        pass_array = []
        log_dir = f"logs/{benchmark_name}_{model_name}_maxiter_{max_iterations}_round{i}"
        # open the log dir and find the corresponding json file
        success_tasks = []
        for task_name in task_names:
            output_file = os.path.join(log_dir, f"{task_name}.json")
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                if data["finish_reason"] == 'solved':
                    tasks_solved_count[task_name] += 1
                    pass_array.append(1)
                    success_tasks.append(output_file)
                else:
                    pass_array.append(0)
            except FileNotFoundError:
                if k == 1:
                    print(f"File: {output_file} not found!!!")
                pass_array.append(0)
                pass
        # dump successful files in to txt
        # Check if the directory for successful tasks lists exists, if not create it
        successful_tasks_dir = "analysis/successful_tasks_lists"
        if not os.path.exists(successful_tasks_dir):
            os.makedirs(successful_tasks_dir)
            print(f"Created directory: {successful_tasks_dir}")
        if args.train_set:
            path = f"analysis/successful_tasks_lists/successful_tasks_{benchmark_name}_{model_name}_maxiter_{max_iterations}_train_round{i}.txt"
        elif args.test_set:
            path = f"analysis/successful_tasks_lists/successful_tasks_{benchmark_name}_{model_name}_maxiter_{max_iterations}_test_round{i}.txt"
        else:
            path = f"analysis/successful_tasks_lists/successful_tasks_{benchmark_name}_{model_name}_maxiter_{max_iterations}_round{i}.txt"
        with open(path, "w") as f:
            for challenge_name in success_tasks:
                f.write(challenge_name + '\n')
        
        # concatenate the pass_array
        if pass_arrays is None:
            pass_arrays = np.expand_dims(np.array(pass_array), axis=1)
        else:
            pass_arrays = np.concatenate((pass_arrays, np.expand_dims(np.array(pass_array), axis=1)), axis=1)
    
    # compute the pass k and confidence interval using boostrap
    B = 5000
    alpha = 0.05
    T, n = pass_arrays.shape
    boot_avgs = []
    print("Bootstraping...")
    for _ in tqdm(range(B)):
        # Top‐level: resample tasks with replacement
        avg_pks = []
        for t in range(T):
            # Within each task: resample runs with replacement
            sample_runs = np.random.choice(pass_arrays[t], size=n, replace=True)
            c = sample_runs.sum()
            avg_pks.append(pass_at_k(n, c, k))
        boot_avgs.append(np.mean(avg_pks))
    pass_at_k_score = np.mean(boot_avgs)
    lower_bound = np.percentile(boot_avgs, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_avgs, 100 * (1 - alpha / 2))
    
    return pass_at_k_score, upper_bound, lower_bound


def compute_pass_k_iter_prompt_intercode_bootstrap(args, k):
    benchmark_name = args.task_name
    model_name = args.model_name
    max_iterations = args.max_iterations
    n_rounds = args.n_rounds
    task_json_file = dataset2root[benchmark_name]['json_file']
    
    with open(task_json_file, "r") as f:
        data = json.load(f)
        
    train_set_path = "analysis/train_tasks.txt"
    test_set_path = "analysis/test_tasks.txt"
    
    if args.train_set:
        with open(train_set_path, "r") as f:
            task_names = [line.strip() for line in f.readlines()]
    elif args.test_set:
        with open(test_set_path, "r") as f:
            task_names = [line.strip() for line in f.readlines()]
    num_tasks = len(task_names)
    print(f"Number of tasks being evaluated: {num_tasks}")
    
    pass_arrays = None
    for i in range(1, n_rounds + 1):
        tasks_solved_count = {}
        
        for task in task_names:
            tasks_solved_count[task] = 0 # Initialize the dictionary
        unsolved_ids = ["challenge_" + str(i) for i in [95, 66, 29, 28, 87, 1, 89, 56, 88, 55]]
        for j in range(k):    
            if j == 0: # here is the pass@1 for the base model without iterative prompting
                log_dir = f"results{benchmark_name}_{model_name}_maxiter_{max_iterations}_round{i}"
            else:
                log_dir = f"results{benchmark_name}_{model_name}_iterprompt{j}_maxiter_{max_iterations}_round{i}"
            for task_name in task_names:
                output_file = os.path.join(log_dir, f"{task_name}.json")
                try:
                    with open(output_file, "r") as f:
                        data = json.load(f)
                    if data["finish_reason"] == 'solved':
                        tasks_solved_count[task_name] += 1
                except FileNotFoundError:
                    if any(unsolved_id in output_file for unsolved_id in unsolved_ids):
                        pass
                    else:
                        if tasks_solved_count[task_name] == 0:
                            print(f"File: {output_file} not found!!!")
                        pass
    
            success_task_list = []
                    
                
            keys = list(tasks_solved_count.keys())
            for key in keys:
                if tasks_solved_count[key] > 0:
                    success_task_list.append(key)

            if args.train_set:
                path = f"analysis/successful_tasks_lists/successful_tasks_{benchmark_name}_train_{model_name}_maxiter_{max_iterations}_iter_prompt_refinement{i}.txt"
                with open(test_set_path, "r") as f:
                    test_tasks = [line.strip() for line in f.readlines()] 
                for task in test_tasks:
                    success_task_list.append(task)

            elif args.test_set:
                path = f"analysis/successful_tasks_lists/successful_tasks_{benchmark_name}_test_{model_name}_maxiter_{max_iterations}_iter_prompt_refinement{i}.txt"
                with open(train_set_path, "r") as f:
                    train_tasks = [line.strip() for line in f.readlines()] # skip train tasks when doing iter refinement on test set
                for task in train_tasks:
                    success_task_list.append(task)
            else:
                path = f"analysis/successful_tasks_lists/successful_tasks_{benchmark_name}_{model_name}_maxiter_{max_iterations}_iter_prompt_refinement{i}.txt"
            
            
            with open(path, "w") as f:
                    for challenge_name in success_task_list:
                        # f.write('placeholder/' + challenge_name + '.json' + '\n')
                        f.write(f"{challenge_name.split('-')[-1]}\n")
                
            pass_at_k_list = [pass_at_k(k, tasks_solved_count[key], k) for key in keys]
            
            
        pass_array = np.array(pass_at_k_list)
        # concatenate the pass_array
        if pass_arrays is None:
            pass_arrays = np.expand_dims(pass_array, axis=1)
        else:
            pass_arrays = np.concatenate((pass_arrays, np.expand_dims(pass_array, axis=1)), axis=1)

    
    B = 5000
    alpha = 0.05
    T, n = pass_arrays.shape
    boot_avgs = []
    print("Bootstraping...")
    for _ in tqdm(range(B)):
        # Top‐level: resample tasks with replacement
        avg_pks = []
        for t in range(T):
            # Within each task: resample runs with replacement
            sample_runs = np.random.choice(pass_arrays[t], size=n, replace=True)
            # c = sample_runs.sum()
            avg_pks.append(sample_runs.mean())
        boot_avgs.append(np.mean(avg_pks))
    pass_at_k_score = np.mean(boot_avgs)
    lower_bound = np.percentile(boot_avgs, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_avgs, 100 * (1 - alpha / 2))
    return pass_at_k_score, upper_bound, lower_bound



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--task_name", type=str, default="intercode_ctf")
    args.add_argument("--model_name", type=str, default="Qwen2.5-Coder-32B-Instruct")
    args.add_argument("--max_iterations", type=int, default=20)
    args.add_argument("--iter_prompt", action="store_true")
    args.add_argument("--n_rounds", type=int, default=10)
    args.add_argument("--max_k", type=int, default=1)
    args.add_argument("--output_file", type=str, default="acc_repeated_sampling.csv")
    args.add_argument("--train_set", action="store_true") # only eval on train set
    args.add_argument("--test_set", action="store_true") # Only eval on test set
    args = args.parse_args()
    # task_num, acc_count, accuracy, success_tasks = grade_benchmark(args.task_name)
    if args.train_set or args.test_set:
        assert args.task_name == "intercode_ctf", "Only intercode_ctf supports train and test set evaluation"
    
    if args.iter_prompt:
        output_file = "iter_prompt_refinement.csv"
        
        for k in range(1, args.max_k + 1):
            if args.task_name == "intercode_ctf":# in iterative prmpting, for flexibility, we don't connect the k with the number of rounds, here n_rounds refer to how many repetitions we have done for the same experiments
                pass_at_k_score, upper_bound, lower_bound = compute_pass_k_iter_prompt_intercode_bootstrap(args, k)
            else:
                raise NotImplementedError(f"Iterative prompting only supports intercode_ctf, but got {args.task_name}")
            
            if args.train_set:
                task_name = args.task_name + "_train"
            elif args.test_set:
                task_name = args.task_name + "_test"
            else:
                task_name = args.task_name
            if os.path.exists(args.output_file):
                with open(args.output_file, "a") as f:
                    f.write(f"{task_name},{args.model_name},{args.max_iterations},{k},{pass_at_k_score},{upper_bound},{lower_bound}\n")
            else:
                with open(args.output_file, "w") as f:
                    f.write(f"task_name,model_name,max_iterations,k,pass_at_k,upper_bound,lower_bound\n")
                    f.write(f"{task_name},{args.model_name},{args.max_iterations},{k},{pass_at_k_score},{upper_bound},{lower_bound}\n")
    else:
        for k in range(1, args.n_rounds + 1): 
            pass_at_k_score, upper_bound, lower_bound = compute_pass_k_bootstrap(args, k)
            if args.train_set:
                task_name = args.task_name + "_train"
            elif args.test_set:
                task_name = args.task_name + "_test"
            else:
                task_name = args.task_name
            if os.path.exists(args.output_file):
                with open(args.output_file, "a") as f:
                    f.write(f"{task_name},{args.model_name},{args.max_iterations},{k},{pass_at_k_score},{upper_bound},{lower_bound}\n")
            else:
                with open(args.output_file, "w") as f:
                    f.write(f"task_name,model_name,max_iterations,k,pass_at_k,upper_bound,lower_bound\n")
                    f.write(f"{task_name},{args.model_name},{args.max_iterations},{k},{pass_at_k_score},{upper_bound},{lower_bound}\n")
            
    # # dump the sucess tasks list
    # with open(f"successful_tasks_{args.task_name}.txt", "w") as f:
    #     for challenge_name in success_tasks:
    #         f.write(challenge_name + '\n')
    
    
    # print(f"Total: {task_num}, Correct: {acc_count}, Accuracy: {accuracy}")
    