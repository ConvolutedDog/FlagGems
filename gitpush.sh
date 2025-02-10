#!/bin/bash

# Maximum number of retries
MAX_RETRIES=100
retry_count=0

# Loop until git push succeeds or the maximum number of retries is reached
while [ $retry_count -lt $MAX_RETRIES ]; do
    # Execute git push
    git push

    # Check the exit status of git push
    if [ $? -eq 0 ]; then
        echo "Git push succeeded!"
        exit 0  # Exit the script if successful
    else
        retry_count=$((retry_count + 1))
        echo "Git push failed, retrying... (Attempt: $retry_count/$MAX_RETRIES)"
        sleep 5  # Wait 5 seconds before retrying
    fi
done

# If the maximum number of retries is reached without success
echo "Maximum retries reached ($MAX_RETRIES attempts), git push still failed."
exit 1  # Exit the script with an error code
