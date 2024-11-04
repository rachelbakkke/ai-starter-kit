#!/bin/bash
# run_synthetic_dataset.sh

# --model-names "Meta-Llama-3.1-70B-Instruct" \
# --model-names "Meta-Llama-3.1-405B-Instruct" \
# --model-names "Mixtral-8x7B-Instruct-v0.1" \

python src/evaluator.py \
--mode synthetic \
--model-names "Meta-Llama-3.1-70B-Instruct" \
--results-dir "./data/results/llmperf" \
--num-concurrent-requests 1 \
--timeout 600 \
--num-input-tokens 100 \
--num-output-tokens 100 \
--num-requests 1 \
--llm-api sncloud

# Notes:
# 1. For CoE Models, make sure to include the prefix "COE/" before each expert name.
#   For example:
#      --model-names "COE/llama-2-7b-chat-hf"
#          OR
#      --model-names "COE/llama-2-7b-chat-hf COE/llama-2-13b-chat-hf"
#          OR
#      --model-names "COE/llama-2-7b-chat-hf COE/Mistral-7B-Instruct-v0.2"
#          OR
#      --model-names "COE/Meta-Llama-3-8B-Instruct"
#
# 2. For Non-CoE models, use the model name directly and remember to update and source the `.env` file for a new endpoint.
#   For example:
#      --model-names "llama-2-7b-chat-hf"
#          OR
#      --model-names "llama-2-13b-chat-hf"
#          OR
#      --model-names "Mistral-7B-Instruct-v0.2"
#          OR
#      --model-names "Meta-Llama-3-8B-Instruct"
#
# 3. For SambaNovaCloud endpoints, change the llm-api parameter to "sncloud" and use the model name directly.
#   For example:
#      --model-names "llama3-8b"
#          OR
#      --model-names "llama3-8b llama3-70b"