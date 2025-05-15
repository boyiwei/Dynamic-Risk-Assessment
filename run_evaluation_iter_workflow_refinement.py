import json
import argparse
import subprocess
import os


dataset2root = {
    'nyu_ctf_test': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset.json"},
    'nyu_ctf_test_segment5_1': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment5_1.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment5_2': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment5_2.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment5_3': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment5_3.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment5_4': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment5_4.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment5_5': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment5_5.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_1': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_1.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_2': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_2.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_3': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_3.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_4': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_4.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_5': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_5.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_6': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_6.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_7': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_7.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_8': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_8.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_9': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_9.json"}, # for task parallel within one benchmark
    'nyu_ctf_test_segment10_10': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/test_dataset_segment10_10.json"}, # for task parallel within one benchmark
    'nyu_ctf_train': {"root_dir": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/", "json_file": "/home/boyiwei/SWE-agent-datasets/NYU_CTF_Bench/development_dataset.json"},
    'cybench': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench.json"},
    'cybench_segment5_1': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment5_1.json"}, # for task parallel within one benchmark
    'cybench_segment5_2': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment5_2.json"}, # for task parallel within one benchmark
    'cybench_segment5_3': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment5_3.json"}, # for task parallel within one benchmark
    'cybench_segment5_4': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment5_4.json"}, # for task parallel within one benchmark
    'cybench_segment5_5': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment5_5.json"}, # for task parallel within one benchmark
    'cybench_segment10_1': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_1.json"}, # for task parallel within one benchmark
    'cybench_segment10_2': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_2.json"}, # for task parallel within one benchmark
    'cybench_segment10_3': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_3.json"}, # for task parallel within one benchmark
    'cybench_segment10_4': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_4.json"}, # for task parallel within one benchmark
    'cybench_segment10_5': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_5.json"}, # for task parallel within one benchmark
    'cybench_segment10_6': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_6.json"}, # for task parallel within one benchmark
    'cybench_segment10_7': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_7.json"}, # for task parallel within one benchmark
    'cybench_segment10_8': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_8.json"}, # for task parallel within one benchmark
    'cybench_segment10_9': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_9.json"}, # for task parallel within one benchmark
    'cybench_segment10_10': {"root_dir": "/home/boyiwei/SWE-agent-datasets/cybench/", "json_file": "/home/boyiwei/SWE-agent-datasets/cybench/cybench_segment10_10.json"}, # for task parallel within one benchmark
    'intercode_ctf': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset.json"},
    'intercode_ctf_segment5_1': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment5_1.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment5_2': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment5_2.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment5_3': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment5_3.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment5_4': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment5_4.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment5_5': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment5_5.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_1': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_1.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_2': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_2.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_3': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_3.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_4': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_4.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_5': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_5.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_6': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_6.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_7': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_7.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_8': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_8.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_9': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_9.json"}, # for task parallel within one benchmark
    'intercode_ctf_segment10_10': {"root_dir": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/", "json_file": "/home/boyiwei/SWE-agent-datasets/Intercode_CTF/intercode_dataset_segment10_10.json"}, # for task parallel within one benchmark
}
def get_examples(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    tasks = [elem for elem in data]
    return tasks


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='nyu_ctf_test')
    args.add_argument('--model_name', type=str, default='gpt-4o-2024-05-13')
    args.add_argument('--max_iterations', type=int, default=10)
    args.add_argument('--config', type=str, default='config/local_config.yaml')
    args.add_argument('--name', type=str, default=None)
    args.add_argument('--round', type=int, default=None)
    args.add_argument('--container_name', type=str, default='ctf_env')
    args.add_argument('--network', type=str, default='ctfnet')
    args.add_argument('--iter_prompt_refinement', action='store_true') # if True, use the iter_prompt_refinement and use customized prompt template.
    args.add_argument('--iter_prompt_round', type=int, default=1) # the round of iter_prompt_refinement
    args.add_argument('--adas-iter-round', type=str, default="1") # the round of ADAS
    args.add_argument('--task_mask', type=str, default=None) # mask the task that are not needed
    
    args = args.parse_args()
    if args.dataset == 'nyu_ctf_test':
        split = 'test'
    elif args.dataset == 'nyu_ctf_train':
        split = 'development'
    else:
        split = 'test'
        # raise ValueError(f"Unknown dataset {args.dataset}")
    args.adas_iter_round = int(args.adas_iter_round)
    
    json_file = dataset2root[args.dataset]['json_file']
    
    tasks = get_examples(json_file)
    
    if args.iter_prompt_refinement:
        assert 'intercode_ctf' in args.dataset, "iter_prompt_refinement only supports intercode_ctf dataset."
        for i in range(len(tasks)):
            task = tasks[i]
            task_name = ("challenge_" + task.split(" ")[-1]).replace("#", "")
            # Check if task_name is in any of the template files
            template_dir = "llm_ctf/prompts/templates"
            template_files = os.listdir(template_dir) if os.path.isdir(template_dir) else []
            iter_prompt_set_name = f"iteration_{args.iter_prompt_round}_{task_name}"
            task_in_template = any(iter_prompt_set_name == filename for filename in template_files)
            prompt_set = iter_prompt_set_name if task_in_template else "default"
            
            print(f"Running task {task}...")
            subprocess.run([
                    'python', 'llm_ctf_solve_ADAS.py',
                    '--dataset', json_file,
                    '--model', args.model_name,
                    '--challenge', task,
                    '--split', split,
                    '--name', args.name,
                    '-c', args.config,
                    '-i', f"{args.round}",
                    '-m', f"{args.max_iterations}",
                    '--container-name', args.container_name,
                    '--network', args.network,
                    '--prompt-set', prompt_set,
                    '--adas-iter-round', f"{args.adas_iter_round}",
                ])
    else:
        for i in range(len(tasks)):
            task = tasks[i]
            if 'intercode_ctf' in args.dataset:
                task_name = ("challenge_" + task.split(" ")[-1]).replace("#", "")
            if args.task_mask:
                with open(args.task_mask, 'r') as f:
                    task_mask = [line.strip() for line in f.readlines()]
                if task_name in task_mask:
                    print(f"Skipping task {task}...")
                    continue
            print(f"Running task {task}...")
            if args.dataset in ['nyu_ctf_test', 'nyu_ctf_train']:
                subprocess.run([
                    'python', 'llm_ctf_solve_ADAS.py',
                    '--model', args.model_name,
                    '--challenge', task,
                    '--split', split,
                    '--name', args.name,
                    '-c', args.config,
                    '-i', f"{args.round}",
                    '-m', f"{args.max_iterations}",
                    '--container-name', args.container_name,
                    '--network', args.network,
                    '--adas-iter-round', f"{args.adas_iter_round}",
                ])
            else:
                subprocess.run([
                    'python', 'llm_ctf_solve_ADAS.py',
                    '--dataset', json_file,
                    '--model', args.model_name,
                    '--challenge', task,
                    '--split', split,
                    '--name', args.name,
                    '-c', args.config,
                    '-i', f"{args.round}",
                    '-m', f"{args.max_iterations}",
                    '--container-name', args.container_name,
                    '--network', args.network,
                    '--adas-iter-round', f"{args.adas_iter_round}",
                ])