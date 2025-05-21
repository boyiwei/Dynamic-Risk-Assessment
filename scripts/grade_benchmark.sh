#!/usr/bin/env bash

# Repeated Sampling
# for benchmark in "cybench";
# do
# for N in 20 30;
# do
# python analysis/grade_benchmark.py --task_name $benchmark --N $N --output_file "acc_repeated_sampling.csv" --k0 12
# done
# done

# benchmark="intercode_ctf"
# for N in 10 20 30;
# do
# python analysis/grade_benchmark.py --task_name $benchmark --N $N --output_file "acc_repeated_sampling.csv" --test_set --k0 12
# python analysis/grade_benchmark.py --task_name $benchmark --N $N --output_file "acc_repeated_sampling.csv" --train_set --k0 12
# done


# # Self Training
# for benchmark in "intercode_ctf";
# do
# for N in 20;
# do
# for ft_epoch in 5 10;
# do
# model_name="Qwen2.5-Coder-32B-Instruct-ft_ft_intercode_nyuagent_singleturn_train_${ft_epoch}_lr_1e-5_fullparam"
# python analysis/grade_benchmark.py --task_name $benchmark --N $N --model_name $model_name --output_file "self_training.csv" --test_set --k0 12
# python analysis/grade_benchmark.py --task_name $benchmark --N $N --model_name $model_name --output_file "self_training.csv" --train_set --k0 12
# done
# done
# done


# # iter prompt refinement

# benchmark="intercode_ctf"
# N=20

# k0=12
# python analysis/grade_benchmark.py --iter_prompt --k0 $k0 --max_k 20 --test_set --output_file "iter_prompt_refinement.csv"
# done


####==================comparative analysis ===========================
benchmark="intercode_ctf"
# python analysis/grade_benchmark.py --task_name $benchmark --N 20 --output_file "acc_repeated_sampling_comparative.csv" --test_set --k0 35
# python analysis/grade_benchmark.py --task_name $benchmark --N 30 --output_file "acc_repeated_sampling_comparative.csv" --test_set --k0 20
# python analysis/grade_benchmark.py --task_name $benchmark --N 40 --output_file "acc_repeated_sampling_comparative.csv" --test_set --k0 10
# python analysis/grade_benchmark.py --task_name $benchmark --N 50 --output_file "acc_repeated_sampling_comparative.csv" --test_set --k0 10
python analysis/grade_benchmark.py --task_name $benchmark --N 60 --output_file "acc_repeated_sampling_comparative.csv" --test_set --k0 3
# python analysis/grade_benchmark.py --task_name $benchmark --N 75 --output_file "acc_repeated_sampling_comparative.csv" --test_set --k0 1
# python analysis/grade_benchmark.py --task_name $benchmark --N 80 --output_file "acc_repeated_sampling_comparative.csv" --test_set --k0 1
