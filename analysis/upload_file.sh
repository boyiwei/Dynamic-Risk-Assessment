#!/usr/bin/env bash


# files to compress

timestamp=$(date +"%Y%m%d_%H%M%S")
# zip -r ../logs/logs_$timestamp.zip ../logs/boyiwei/intercode_ctf_Qwen2.5-Coder-32B-Instruct_iterprompt*
zip -r ../logs/logs_$timestamp.zip ../logs/boyiwei/intercode_ctf_Qwen2.5-Coder-32B-Instruct_maxiter_20_round*

./dropbox_uploader.sh upload ../logs/logs_$timestamp.zip /ctf_logs

