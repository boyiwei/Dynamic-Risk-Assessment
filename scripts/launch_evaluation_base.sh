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
docker stop $(docker ps -q) # Stop docker container
docker ps -a -q | xargs -r docker rm
# Change the dataset into the correct branch
cd ~/SWE-agent-datasets
git checkout main
echo "Using the main branch (NYU format) of the cybersecurity dataset for evaluation"
cd ~/nyuctf_agents

model_name="Qwen2.5-Coder-32B-Instruct"
# Intercode evaluation
# dataset="nyu_ctf_test"
max_iter=20
parallelism=10
# Launch evaluations for i in 1,2,3,4 in parallel and log output to output_i.txt
for max_iter in 10; do
for dataset in "intercode_ctf" ; do
echo "Evaluating dataset: ${dataset}"
for j in $(seq 1 $parallelism); do
    (
      # Calculate task parallel id within the same benchmark
      for i in {11..12}; do
      sub_dataset_name="${dataset}_segment${parallelism}_${j}"
          python run_evaluation.py \
            --dataset "${sub_dataset_name}" \
            --model_name "${model_name}" \
            --max_iterations "${max_iter}" \
            --config config/local_config.yaml \
            --round "${i}" \
            --name "${dataset}_${model_name}_maxiter_${max_iter}" \
            --network "ctfnet${j}" \
            --container_name "ctf_env${j}" \
            --task_mask "~/nyuctf_agents/analysis/test_set_task_mask.txt" \
            

      done
    )   > "output_${j}.txt" 2>&1 & # Redirect both stdout and stderr to output_i.txt
done
wait
done
# # La
done
echo "All evaluations completed."

