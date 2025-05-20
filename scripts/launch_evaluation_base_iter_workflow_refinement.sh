#!/usr/bin/env bash

trap "echo 'Terminating all subprocesses...'; kill 0" SIGINT


parallelism=10
model_name="Qwen2.5-Coder-32B-Instruct"
max_iter=20
dataset="intercode_ctf"

for i in {1..10}; do
    container_name="ctf_env${i}"
    docker stop "${container_name}"
    docker rm "${container_name}"
done

# Start overall timing
TOTAL_START_TIME=$SECONDS

for iteration in 2 9; do
    echo "Iteration: $iteration"
    ITER_START_TIME=$SECONDS
 
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
                --max_iterations "${max_iter}" \
                --config config/local_config.yaml \
                --round "${i}" \
                --name "${dataset}_${model_name}_adas${iteration}_maxiter_${max_iter}" \
                --network "ctfnet${j}" \
                --container_name "ctf_env${j}" \
                --adas-iter-round "${iteration}" \
                # --task_mask "/home/boyiwei/NYU-Agents-ADAS/analysis/test_set_task_mask.txt"
        done
        )   > "output_${j}.txt" 2>&1 & # Redirect both stdout and stderr to output_i.txt
    done
    wait
    
    # Calculate and display iteration time
    ITER_ELAPSED_TIME=$(( SECONDS - ITER_START_TIME ))
    echo "Iteration $iteration completed in ${ITER_ELAPSED_TIME} seconds."
done

# Calculate and display total time
TOTAL_ELAPSED_TIME=$(( SECONDS - TOTAL_START_TIME ))
echo "All evaluations completed in ${TOTAL_ELAPSED_TIME} seconds ($(( TOTAL_ELAPSED_TIME / 60 )) minutes)."

    