#!/usr/bin/env bash

trap "echo 'Terminating all subprocesses...'; kill 0" SIGINT


parallelism=10
model_name="Qwen2.5-Coder-32B-Instruct"
N=20
dataset="intercode_ctf"

for i in {1..10}; do
    container_name="ctf_env${i}"
    docker stop "${container_name}"
    docker rm "${container_name}"
done

for iteration in {1..20}; do
    echo "Iteration: $iteration"
    # Run the evaluation script with the specified parameters
    python iter_workflow_refinement/search.py --n_generation $iteration

    # Launch evaluations for i in 1,2,3,4 in parallel and log output to output_i.txt
    echo "Evaluating dataset: ${dataset}"
    for j in $(seq 1 $parallelism); do
        (
        # Calculate task parallel id within the same benchmark
        for i in {1..5}; do
        sub_dataset_name="${dataset}_segment${parallelism}_${j}"
            python run_evaluation_iter_workflow_refinement.py \
                --dataset "${sub_dataset_name}" \
                --model_name "${model_name}" \
                --N "${N}" \
                --config config/local_config.yaml \
                --round "${i}" \
                --name "${dataset}_${model_name}_adas${iteration}_maxiter_${N}" \
                --network "ctfnet${j}" \
                --container_name "ctf_env${j}"
        done
        )   > "output_${j}.txt" 2>&1 & # Redirect both stdout and stderr to output_i.txt
    done
    wait
    # # La
    echo "All evaluations completed."

    # grade benchmark
    python analysis/grade_benchmark.py --model_name "${model_name}_adas${iteration}" --k0 5 --dump_to_adas --train_set
done
    