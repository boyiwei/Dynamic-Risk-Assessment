#!/usr/bin/env bash

# Trap SIGINT (Ctrl+C) and kill all subprocesses
trap "echo 'Terminating all subprocesses...'; kill 0" SIGINT

IMAGES_TO_KEEP=("ctfenv:latest" "sweagent/swe-agent:latest" "sweagent/enigma:latest")

# Function to clean docker images
cleanup_docker_images() {
    echo "Cleaning up docker images"
    all_images=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}')
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



model_name="Qwen2.5-Coder-32B-Instruct-ft"
# Intercode evaluation
ft_epoch=10
lr=1e-5
ft_dataset="ctf_intercode_nyuagent_singleturn_train"

ft_paradigm="fullparam"
N=20

parallelism=10
# Launch evaluations for i in 1,2,3,4 in parallel and log output to output_i.txt
for dataset in "intercode_ctf"; do
echo "Evaluating dataset: ${dataset}"
for j in $(seq 1 $parallelism); do
    (
      # Calculate task parallel id within the same benchmark
      for i in {13..15}; do
      sub_dataset_name="${dataset}_segment${parallelism}_${j}"
          python run_evaluation.py \
            --dataset "${sub_dataset_name}" \
            --model_name "${model_name}" \
            --N "${N}" \
            --config config/local_config.yaml \
            --round "${i}" \
            --name "${dataset}_${model_name}_ft_${ft_dataset}_${ft_epoch}_lr_${lr}_${ft_paradigm}_maxiter_${N}" \
            --network "ctfnet${j}" \
            --container_name "ctf_env${j}"
      done
    )   > "output_${j}.txt" 2>&1 & # Redirect both stdout and stderr to output_i.txt
done
wait
done
# # La
echo "All evaluations completed."
done
