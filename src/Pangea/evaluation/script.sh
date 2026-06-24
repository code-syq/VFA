#!/bin/bash

cd lmms-eval
conda activate llava-next
model='neulab/Pangea-7B'
tasks=('mmmu_val' 'marvl') # list of tasks for evaluation

for task in "${tasks[@]}"; do
    echo ${task}
    python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model pangea \
        --model_args pretrained=${model} \
        --tasks ${task} \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix ${task} \
        --output_path eval_logs
done
