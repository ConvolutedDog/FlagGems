#!/bin/bash

# Define the base directory and instance list
IterTests_HOME=~/Github/FlagGems-IterTests
DEV=~/Github/FlagGems-dev
Instances=(
    "4090-1" \
    "4090-1-Five" \
    "4090-1-Four" \
    "4090-1-Three" \
    "4090-1-Two" \
    "H100-0" \
    "H100-0-Five" \
    "H100-0-Three" \
    "H100-0-Two" \
    "H100-0-four" \
    "V100-4" \
    "V100-4-Five" \
    "V100-4-Three" \
    "V100-4-Two" \
    "V100-4-four"
)

# Delete directories that are not in Instances list
delete_extra_directories() {
    echo "Deleting extra directories in $IterTests_HOME..."
    for dir in "$IterTests_HOME"/*; do
        if [[ -d "$dir" ]]; then
            dir_name=$(basename "$dir")
            if [[ ! " ${Instances[@]} " =~ " ${dir_name} " ]]; then
                echo "Deleting directory: $dir"
                rm -rf "$dir"
            fi
        fi
    done
    echo "Extra directories deleted."
}

# Update instances
update_instances() {
    for instance in "${Instances[@]}"; do
        instance_dir="$IterTests_HOME/$instance"

        # Create directory if it doesn't exist
        if [[ ! -d "$instance_dir" ]]; then
            echo "Creating directory: $instance_dir"
            mkdir -p "$instance_dir"
        fi

        # Navigate to the instance directory
        cd "$instance_dir" || { echo "Failed to enter directory: $instance_dir"; exit 1; }

        # Delete all files and hidden files in the directory
        echo "Cleaning directory: $instance_dir"
        rm -rf ./* .[^.]* ..?*  # Delete all files and hidden files (excluding . and ..)

        # Copy all files and hidden files from the local directory to the instance directory
        echo "Updating instance: $instance"
        rsync -av --exclude='*.log' --exclude='*.csv' --exclude='*.txt' --exclude='*.xlsx' \
            --exclude='update.sh' "$DEV/" ./

        # Extract the number from the instance name
        # Supports both "H100-0-four" and "V100-4" formats
        if [[ "$instance" =~ -([0-9]+)(-|$) ]]; then
            num=${BASH_REMATCH[1]}
        else
            echo "Warning: No number found in instance name: $instance"
            num=0  # Default to 0 if no number is found
        fi

        # Create cudaenvset.sh with the extracted number
        echo "export CUDA_VISIBLE_DEVICES=$num" > cudaenvset.sh
        chmod +x cudaenvset.sh  # Make the script executable

        echo "Instance updated: $instance, CUDA_VISIBLE_DEVICES set to $num"
    done
}

# Main script execution
delete_extra_directories
update_instances

echo "All instances updated successfully!"
