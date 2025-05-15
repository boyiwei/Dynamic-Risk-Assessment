#!/usr/bin/env bash

for iteration in {3..10}; do

logs_dir=~/nyuctf_agents/logs/intercode_ctf_Qwen2.5-Coder-32B-Instruct_iterprompt$((iteration - 1))_maxiter_20_round1


echo "Starting prompt refinement iteration $iteration"

python generate_prompt_refinement.py --iteration $iteration --logs_dir $logs_dir

echo "Convert the refined prompt to the correct format"


python convert_rules_to_template.py --iteration $iteration


done