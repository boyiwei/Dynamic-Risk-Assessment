#!/usr/bin/env bash

trap "echo 'Terminating all subprocesses...'; kill 0" SIGINT

IMAGES_TO_KEEP=("ctfenv:latest" "sweagent/swe-agent:latest", "sweagent/enigma:latest")
# clean docker image
# Get all images in the format "repository:tag imageID"
cleanup_docker_images() {
    echo "Cleaning up docker images"
    all_images=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}')
    # Loop through each image
    while IFS= read -r line; do
        repo_tag=$(echo "$line" | awk '{print $1}')
        image_id=$(echo "$line" | awk '{print $2}')
        
        keep=false
        for keep_image in "${IMAGES_TO_KEEP[@]}"; do
            if [[ "$repo_tag" == "$keep_image" ]]; then
                keep=true
                break
            fi
        done

        if [ "$keep" = false ]; then
            echo "Removing image $repo_tag ($image_id)"
            docker rmi "$image_id"
        fi
    done <<< "$all_images"
    echo "Done cleaning up docker images"
}
# docker stop $(docker ps -q) # Stop docker container
# docker ps -a -q | xargs -r docker rm
# Change the dataset into the correct branch
for i in {1..10}; do
    container_name="ctf_env${i}"
    docker stop "${container_name}"
    docker rm "${container_name}"
done


cd ~/SWE-agent-datasets
git checkout main
echo "Using the main branch (NYU format) of the cybersecurity dataset for evaluation"
cd ~/nyuctf_agents


model_name="Qwen2.5-Coder-32B-Instruct"
# Intercode evaluation
# dataset="nyu_ctf_test"
max_iter=20
parallelism=10


for rep_round in {14..20}; do

rm -rf ~/nyuctf_agents/llm_ctf/templates/iteration* # remove the previous template

for iter_prompt_round in {1..20}; do

python analysis/grade_benchmark.py --iter_prompt --n_rounds $rep_round --max_k $iter_prompt_round --test_set

if [ $iter_prompt_round -eq 1 ]; then
    logs_dir=~/nyuctf_agents/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_20_round${rep_round}
else
    logs_dir=~/nyuctf_agents/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_iterprompt$((iter_prompt_round - 1))_maxiter_20_round${rep_round}
fi


echo "Starting prompt refinement iteration $iter_prompt_round"

python iter_prompt_refinement/save_state/generate_prompt_refinement.py  --iteration $iter_prompt_round --logs_dir $logs_dir --round ${rep_round} --skip_existing --test_set

echo "Convert the refined prompt to the correct format"


python iter_prompt_refinement/save_state/add_prompt_to_template.py --iteration $iter_prompt_round --round ${rep_round}

# Launch evaluations for i in 1,2,3,4 in parallel and log output to output_i.txt
for dataset in "intercode_ctf" ; do
echo "Evaluating dataset: ${dataset}"
for j in $(seq 1 $parallelism); do
    (
      # Calculate task parallel id within the same benchmark
      for i in $rep_round; do
      sub_dataset_name="${dataset}_segment${parallelism}_${j}"
          python run_evaluation.py \
            --dataset "${sub_dataset_name}" \
            --model_name "${model_name}" \
            --max_iterations "${max_iter}" \
            --config config/local_config.yaml \
            --round "${i}" \
            --name "${dataset}_${model_name}_iterprompt${iter_prompt_round}_maxiter_${max_iter}" \
            --network "ctfnet${j}" \
            --container_name "ctf_env${j}" \
            --iter_prompt_refinement \
            --iter_prompt_round "${iter_prompt_round}" \
            --task_mask "~/nyuctf_agents/analysis/successful_tasks_lists/successful_tasks_intercode_ctf_test_Qwen2.5-Coder-32B-Instruct_maxiter_20_iter_prompt_refinement${rep_round}.txt"
      done
    )   > "output_${j}.txt" 2>&1 & # Redirect both stdout and stderr to output_i.txt
done
wait
done
# # La
echo "All evaluations completed."

done
done


# # finish the previous eval
# for rep_round in {1..5}; do

# rm -rf ~/nyuctf_agents/llm_ctf/templates/iteration* # remove the previous template

# for iter_prompt_round in {7..9}; do

# python analysis/grade_benchmark.py --iter_prompt --n_rounds $rep_round --max_k $iter_prompt_round --test_set

# if [ $iter_prompt_round -eq 1 ]; then
#     logs_dir=~/nyuctf_agents/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_20_round${rep_round}
# else
#     logs_dir=~/nyuctf_agents/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_iterprompt$((iter_prompt_round - 1))_maxiter_20_round${rep_round}
# fi


# echo "Starting prompt refinement iteration $iter_prompt_round"

# python iter_prompt_refinement/save_state/generate_prompt_refinement.py  --iteration $iter_prompt_round --logs_dir $logs_dir --round ${rep_round} --skip_existing --test_set

# echo "Convert the refined prompt to the correct format"


# python iter_prompt_refinement/save_state/add_prompt_to_template.py --iteration $iter_prompt_round --round ${rep_round}

# # Launch evaluations for i in 1,2,3,4 in parallel and log output to output_i.txt
# for dataset in "intercode_ctf" ; do
# echo "Evaluating dataset: ${dataset}"
# for j in $(seq 1 $parallelism); do
#     (
#       # Calculate task parallel id within the same benchmark
#       for i in $rep_round; do
#       sub_dataset_name="${dataset}_segment${parallelism}_${j}"
#           python run_evaluation.py \
#             --dataset "${sub_dataset_name}" \
#             --model_name "${model_name}" \
#             --max_iterations "${max_iter}" \
#             --config config/local_config.yaml \
#             --round "${i}" \
#             --name "${dataset}_${model_name}_iterprompt${iter_prompt_round}_maxiter_${max_iter}" \
#             --network "ctfnet${j}" \
#             --container_name "ctf_env${j}" \
#             --iter_prompt_refinement \
#             --iter_prompt_round "${iter_prompt_round}" \
#             --task_mask "~/nyuctf_agents/analysis/successful_tasks_lists/successful_tasks_intercode_ctf_test_Qwen2.5-Coder-32B-Instruct_maxiter_20_iter_prompt_refinement${rep_round}.txt"
#       done
#     )   > "output_${j}.txt" 2>&1 & # Redirect both stdout and stderr to output_i.txt
# done
# wait
# done
# # # La
# echo "All evaluations completed."

# done
# done