#!/bin/bash

# Define the base directory and instance list
IterTests_HOME=/home/yangjianchao/Github/FlagGems-IterTests
DEV=/home/yangjianchao/Github/FlagGems-dev
# Instances=(
#     "FlagGems-4090-1" \
#     "FlagGems-4090-2" \
#     "FlagGems-4090-3" \
#     "FlagGems-H100-0" \
#     "FlagGems-V100-4"
# )
Instances=(
    "FlagGems-4090-1" \
    "FlagGems-4090-2" \
    "FlagGems-4090-3" \
    "FlagGems-H100-0"
)

# Loop through each instance
for instance in "${Instances[@]}"; do
    # Navigate to the instance directory
    cd "$IterTests_HOME/$instance" || { echo "Failed to enter directory: $IterTests_HOME/$instance"; exit 1; }

    # Delete all files and hidden files in the directory
    rm -rf ./* .[^.]* ..?*  # Delete all files and hidden files (excluding . and ..)

    # Copy all files and hidden files from the local directory to the instance directory
    rsync -av --exclude='*.log' --exclude='*.csv' --exclude='*.txt' --exclude='*.xlsx' \
        --exclude='update.sh' "$DEV/" ./

    echo "Updated instance: $instance"
done

echo "All instances updated successfully!"
