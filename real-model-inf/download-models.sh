#!/bin/bash

# Default not using Hugging Face mirror for downloading models,
# but if you cannot access Hugging Face, you can set it to `true`
# or just add `--use-mirror` to the command line.
use_mirror=false

# Define model list
models=(
    "google-bert/bert-base-uncased"
    "sharpbai/Llama-2-7b-hf"
    "llava-hf/llava-1.5-7b-hf"
)

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use-mirror) use_mirror=true ;;
        *) echo "Unkown arguments: $1"; exit 1 ;;
    esac
    shift
done

# Set Hugging Face mirror address
if [ "$use_mirror" = true ]; then
    echo "Using Hugging Face mirror."
    export HF_ENDPOINT=https://hf-mirror.com
else
    echo "Not using Hugging Face mirror."
    unset HF_ENDPOINT
fi

index=1

# Traverse the model list
for model in "${models[@]}"; do
    # Replace '/' with '--' to form the cache directory name
    model_cache_dir=$(echo "$model" | sed 's/\//--/g')
    cache_path="$HOME/.cache/huggingface/hub/models--$model_cache_dir"

    # Check if the model is already cached
    if [ -d "$cache_path" ]; then
        echo "[$index] Model $model is already cached, skip download."
        echo "  If your model has not been cached, please use the following command to download manually:"
        echo "    huggingface-cli download --resume-download --local-dir-use-symlinks False $model"
    else
        echo "[$index] Downloading model $model ..."
        huggingface-cli download --resume-download --local-dir-use-symlinks False "$model"

        # Check if the download was successful
        if [ $? -eq 0 ]; then
            echo "  Model $model download complete."
        else
            echo "  Model $model download failed, please check your network or model name."
        fi
    fi

    index=$((index + 1))
done
