from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-7b-instruct-vllm',
        path='Qwen/Qwen2.5-7B-Instruct',
        # vLLM by default tries to reserve ~90% of GPU memory; lower this to
        # avoid startup failures when other processes already occupy memory.
        model_kwargs=dict(tensor_parallel_size=2, gpu_memory_utilization=0.9),
        max_out_len=4096,
        batch_size=8192,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
