#!/usr/bin/env bash

# Repeated Sampling
# for benchmark in "cybench";
# do
# for max_iter in 20 30;
# do
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations $max_iter --output_file "acc_repeated_sampling.csv" --n_rounds 12
# done
# done

# benchmark="intercode_ctf"
# for max_iter in 10 20 30;
# do
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations $max_iter --output_file "acc_repeated_sampling.csv" --test_set --n_rounds 12
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations $max_iter --output_file "acc_repeated_sampling.csv" --train_set --n_rounds 12
# done


# # Self Training
# for benchmark in "intercode_ctf";
# do
# for max_iter in 20;
# do
# for ft_epoch in 5 10;
# do
# model_name="Qwen2.5-Coder-32B-Instruct-ft_ft_intercode_nyuagent_singleturn_train_${ft_epoch}_lr_1e-5_fullparam"
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations $max_iter --model_name $model_name --output_file "self_training.csv" --test_set --n_rounds 12
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations $max_iter --model_name $model_name --output_file "self_training.csv" --train_set --n_rounds 12
# done
# done
# done


# # iter prompt refinement

# benchmark="intercode_ctf"
# max_iter=20

# rep_round=12
# python analysis/grade_benchmark.py --iter_prompt --n_rounds $rep_round --max_k 20 --test_set --output_file "iter_prompt_refinement.csv"
# done


####==================comparative analysis ===========================
benchmark="intercode_ctf"
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations 20 --output_file "acc_repeated_sampling_comparative.csv" --test_set --n_rounds 35
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations 30 --output_file "acc_repeated_sampling_comparative.csv" --test_set --n_rounds 20
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations 40 --output_file "acc_repeated_sampling_comparative.csv" --test_set --n_rounds 10
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations 50 --output_file "acc_repeated_sampling_comparative.csv" --test_set --n_rounds 10
python analysis/grade_benchmark.py --task_name $benchmark --max_iterations 60 --output_file "acc_repeated_sampling_comparative.csv" --test_set --n_rounds 3
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations 75 --output_file "acc_repeated_sampling_comparative.csv" --test_set --n_rounds 1
# python analysis/grade_benchmark.py --task_name $benchmark --max_iterations 80 --output_file "acc_repeated_sampling_comparative.csv" --test_set --n_rounds 1
