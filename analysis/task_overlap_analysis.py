import os
import argparse




def task_overlap_analysis(file1, file2):
    with open(file1, 'r') as f:
        file1_dirs = f.readlines()
    with open(file2, 'r') as f:
        file2_dirs = f.readlines()
    file1_dirs = [dir.split('/')[-1].strip() for dir in file1_dirs]
    file2_dirs = [dir.split('/')[-1].strip() for dir in file2_dirs]
    file1_set = set(file1_dirs)
    file2_set = set(file2_dirs)
    files_in_file1_not_in_file2 = file1_set - file2_set
    files_in_file2_not_in_file1 = file2_set - file1_set
    print(f"Files in {file1} but not in {file2}:")
    for file in files_in_file1_not_in_file2:
        print(file)
    print(f"\nFiles in {file2} but not in {file1}:")
    for file in files_in_file2_not_in_file1:
        print(file)
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file1', type=str, default="successful_tasks_lists/successful_tasks_intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_20_iter_prompt_refinement1.txt")
    args.add_argument('--file2', type=str, default='successful_tasks_lists/intercode_ctf_Qwen2.5-Coder-32B-Instruct.txt')
    args = args.parse_args()
    task_overlap_analysis(args.file1, args.file2) 