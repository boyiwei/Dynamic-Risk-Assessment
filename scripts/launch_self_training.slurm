#!/bin/bash
#SBATCH --job-name=ctf_32B  # create a short name for your job
#SBATCH --nodes=2               # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=120G   # memory per cpu-core
#SBATCH --gres=gpu:8
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends


module purge
module load anaconda3/2023.3
conda activate s1
module load cudatoolkit/12.4

# Set training parameters
epochs=5
lr=1e-5
batch_size=16
weight_decay=1e-4
train_dataset_name="ctf_intercode_nyuagent_singleturn_train"

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-Coder-32B-Instruct"


# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done


# Get node information
node_array=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nnodes=$(echo $node_array | wc -w)
head_node=($node_array)
head_node_ip=$(ssh $head_node hostname --ip-address)
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=3600


# Calculate gradient accumulation steps
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$((batch_size/(gpu_count * nnodes)))

echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
echo "Head node IP: $head_node_ip"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"



# Launch distributed training using srun
run_name="qwen_${train_dataset_name}_bs${batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}_${uid}"
output_dir="${base_model}_experiment_scratch/qwen_${train_dataset_name}_bs${batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}"

srun torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpu_count \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    self_training/sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=${epochs} \
    --dataset_name=${train_dataset_name} \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --report_to="wandb" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="self_training/fsdp_config_qwen_cpu.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="${output_dir}" \
    --push_to_hub=false \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'

# move tokenizer
cp /scratch/gpfs/bw1822/nlp_checkpoints/Qwen2.5-Coder-32B-Instruct/{tokenizer.json,tokenizer_config.json,special_tokens_map.json,added_tokens.json,vocab.json,merges.txt} ${output_dir}

# add rope scaling in the config for evaluation
cd ${output_dir}
python -c "import json; d=json.load(open('config.json')); d['rope_scaling']={'factor':4.0, 'original_max_position_embeddings':32768, 'type':'yarn'}; json.dump(d, open('config.json','w'), indent=2)"
