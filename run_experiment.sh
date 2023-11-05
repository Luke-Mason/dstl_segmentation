#!/bin/bash

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

# Extract the <number> from the argument
number=$1

# Loop from 0 to 9 (inclusive) and call the Python script
for i in {0..9}; do
    config="experiments/dstl_ex${number}.json"
    echo "Calling srun --mem=50G --cpus-per-task=8 --gres=gpu:1 python3 dstl_train.py --config $config --cl $i"
    srun --mem=50G --cpus-per-task=8 --gres=gpu:1 python3 dstl_train.py --config "$config" --cl "$i"
done
