"""
Pretrained (non-instruct) variant of LLaVA-OneVision 1.5 for lmms_eval.

Skips chat template formatting and system prompt injection, making it suitable
for evaluating mid-training / pretrained checkpoints that haven't learned
chat formatting yet.

Usage:
    python -m lmms_eval \
        --model llava_onevision1_5_pretrained \
        --model_args pretrained=/path/to/stage1.5-checkpoint \
        --tasks seedbench_ppl,mme --limit 8
"""

import copy
import re
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.simple.llava_onevision1_5 import Llava_OneVision1_5

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")


@register_model("llava_onevision1_5_pretrained")
class Llava_OneVision1_5_Pretrained(Llava_OneVision1_5):
    """
    Pretrained (non-instruct) mode for LLaVA-OneVision 1.5.

    Key differences from the parent class:
    - No chat template wrapping (no <|im_start|>/<|im_end|> tokens)
    - No system prompt injection (default system_prompt="")
    - Raw prompt format: <vision_placeholders>\n<text>
    - Still uses processor for tokenization and vision token expansion
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        system_prompt: Optional[str] = "",
        **kwargs,
    ) -> None:
        # Default to empty system prompt for pretrained models
        super().__init__(pretrained=pretrained, system_prompt=system_prompt, **kwargs)
        eval_logger.info(
            f"[Pretrained mode] Chat template DISABLED. "
            f"system_prompt={'(empty)' if not self.system_prompt else repr(self.system_prompt)}"
        )

    @staticmethod
    def _count_visuals(processed_visuals: list) -> Tuple[int, int]:
        """Count images and videos in a processed_visuals list."""
        n_images = sum(1 for v in processed_visuals if isinstance(v, dict) and v.get("type") == "image")
        n_videos = sum(1 for v in processed_visuals if isinstance(v, dict) and v.get("type") == "video")
        return n_images, n_videos

    def _build_raw_prompt(self, text: str, num_images: int = 0, num_videos: int = 0) -> str:
        """
        Build a raw prompt with vision placeholders but NO chat template formatting.

        For Qwen2.5-VL based models, each image needs:
            <|vision_start|><|image_pad|><|vision_end|>
        Each video needs:
            <|vision_start|><|video_pad|><|vision_end|>

        The processor will later expand <|image_pad|>/<|video_pad|> to the
        correct number of tokens based on image resolution.
        """
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        for _ in range(num_images):
            parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        for _ in range(num_videos):
            parts.append("<|vision_start|><|video_pad|><|vision_end|>")
        if text:
            parts.append(text)
        return "\n".join(parts)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (loglikelihood, pretrained)")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            # Build the visual content for the message (needed by process_vision_info)
            processed_visuals = []
            if visual is not None and visual != []:
                for v in visual:
                    if isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov")):
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": v,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                    elif isinstance(v, Image.Image):
                        processed_visuals.append({"type": "image", "image": v.convert("RGB")})

            # Remove <image> placeholder from context
            ctx = contexts.replace("<image>", "").strip() if contexts else ""

            # Build prompts using chat template (matches training format)
            # Context: system + user turn with vision + generation prompt
            message_ctx = []
            if self.system_prompt:
                message_ctx.append({"role": "system", "content": self.system_prompt})
            message_ctx.append(
                {
                    "role": "user",
                    "content": processed_visuals + [{"type": "text", "text": ctx}]
                    if processed_visuals
                    else [{"type": "text", "text": ctx}],
                }
            )
            # Full: system + user + assistant (with continuation)
            message_full = copy.deepcopy(message_ctx)
            message_full.append({"role": "assistant", "content": continuation})

            text_ctx = self.processor.apply_chat_template(
                message_ctx, tokenize=False, add_generation_prompt=True,
            )
            text_full = self.processor.apply_chat_template(
                message_full, tokenize=False, add_generation_prompt=False,
            )

            image_inputs, video_inputs = process_vision_info(message_full)

            # Process inputs for full sequence
            inputs_full = self.processor(
                text=[text_full],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Process inputs for context-only to get the length
            inputs_ctx = self.processor(
                text=[text_ctx],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            ctx_length = inputs_ctx.input_ids.shape[1]

            if self.device_map == "auto":
                inputs_full = inputs_full.to("cuda")
            else:
                inputs_full = inputs_full.to(self.device)

            full_input_ids = inputs_full.input_ids

            with torch.inference_mode():
                # Do NOT pass labels — the checkpoint's modeling code may use
                # the wrong vocab_size (top-level config.vocab_size vs
                # text_config.vocab_size) when computing the loss, causing a
                # shape mismatch.  Compute loss manually instead.
                outputs = self.model(**inputs_full)

                logits = outputs.logits
                # Shift: predict token t+1 from logits at position t
                shift_logits = logits[:, ctx_length - 1 : -1, :].contiguous()
                shift_labels = full_input_ids[:, ctx_length:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_input_ids[:, ctx_length:]
            greedy_tokens = greedy_tokens[:, ctx_length - 1 : full_input_ids.shape[1] - 1]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (pretrained)")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            # Build message lists (for process_vision_info) and raw text prompts
            batched_messages = []
            texts = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in (visual_list[i] or []):
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append({"type": "image", "image": visual.convert("RGB")})

                # Message list for process_vision_info (no system message)
                message = [
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}] if processed_visuals else [{"type": "text", "text": context}],
                    }
                ]
                batched_messages.append(message)

                # Raw text prompt (no chat template)
                n_images, n_videos = self._count_visuals(processed_visuals)
                texts.append(self._build_raw_prompt(context, n_images, n_videos))

            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            do_sample = bool(current_gen_kwargs.get("temperature", 0) and current_gen_kwargs["temperature"] > 0)
            gen_args = {
                **inputs,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "num_beams": current_gen_kwargs["num_beams"],
                "max_new_tokens": current_gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
            }
            if do_sample:
                gen_args.update(
                    do_sample=True,
                    temperature=float(current_gen_kwargs.get("temperature", 1.0)),
                    top_p=float(current_gen_kwargs.get("top_p", 1.0)),
                )
            with torch.inference_mode():
                cont = self.model.generate(**gen_args)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
