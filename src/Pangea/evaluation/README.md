## Evaluation on PangeaBench

To run the evaluation on PangeaBench, use the following command:

```bash
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
```

Here, `${model}` refers to either a locally available model or a model on HuggingFace, identified by its repository ID. Note that we use `conv_template=qwen_1_5` for Pangea. You should change to other conv_template when appropriate.

`script.sh` shows an example script to run evaluation.

### Task Specification
All tasks can be found in the directory `lmms-eval/lmms_eval/tasks`. Each task is specified by its corresponding YAML file within the respective subdirectories.

For example, to run the `llava_in_the_wild` benchmark, the task name is defined in the file `lmms-eval/lmms_eval/tasks/llava-in-the-wild/llava-in-the-wild.yaml` with `task: "llava_in_the_wild"`. Thus, to run this task, set `task=llava_in_the_wild` in the command.

### Running Group of Tasks
In some cases, the YAML file may specify a group of tasks. For instance, the file `lmms-eval/lmms_eval/tasks/marvl/marvl.yaml` defines a task group called `marvl`, which includes six individual tasks:

```
  - marvl_id
  - marvl_sw
  - marvl_ta
  - marvl_tr
  - marvl_zh
  - nlvr2
```

If you specify `task=marvl` when running the evaluation, the system will execute all six tasks within this group.
