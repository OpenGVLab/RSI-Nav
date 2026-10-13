# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen3_vl_attention_class

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForImageTextToText,
    Qwen3VLForConditionalGeneration,
    Trainer,
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.fa2_compat import maybe_patch_qwen35_flash_attention


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  


def set_model(model_args, model):
    """Unfreeze subsets for full fine-tune (Qwen3-VL / Qwen3-VL-MoE / Qwen3.5 when structure matches)."""
    visual = getattr(model, "visual", None)
    if visual is not None:
        for p in visual.parameters():
            p.requires_grad = model_args.tune_mm_vision
        merger = getattr(visual, "merger", None)
        if merger is not None:
            for p in merger.parameters():
                p.requires_grad = model_args.tune_mm_mlp

    lm = getattr(model, "language_model", None)
    if lm is not None:
        for p in lm.parameters():
            p.requires_grad = model_args.tune_mm_llm

    if hasattr(model, "lm_head") and model.lm_head is not None:
        model.lm_head.requires_grad = model_args.tune_mm_llm


def train(attn_implementation="flash_attention_2"):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    path = model_args.model_name_or_path
    chosen_attn = os.environ.get("ATTN_IMPLEMENTATION", attn_implementation)

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    mt = getattr(config, "model_type", None) or ""
    maybe_patch_qwen35_flash_attention(mt, chosen_attn)

    load_kw = dict(
        cache_dir=training_args.cache_dir,
        attn_implementation=chosen_attn,
        # torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        dtype=(torch.bfloat16 if training_args.bf16 else None),
        trust_remote_code=True,
    )

    if mt in ("qwen3_5", "qwen3_5_moe"):
        model = AutoModelForImageTextToText.from_pretrained(path, **load_kw)
        data_args.model_type = "qwen3_5"
    elif mt == "qwen3_vl_moe":
        model = AutoModelForImageTextToText.from_pretrained(path, **load_kw)
        data_args.model_type = "qwen3vl"
    elif mt == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, **load_kw)
        data_args.model_type = "qwen3vl"
    else:
        raise ValueError(
            f"Unsupported model_type={mt!r} for path={path!r}. "
            "Use Qwen3-VL / Qwen3-VL-MoE / Qwen3.5 (multimodal) checkpoints."
        )

    print(
        f"Initialized model from {model_args.model_name_or_path} "
        f"({model.__class__.__name__}, data_args.model_type={data_args.model_type})"
    )
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    if data_args.model_type == "qwen3_5" and (
        data_args.data_flatten or data_args.data_packing
    ):
        print(
            "qwen3_5: data_flatten/data_packing is only wired for Qwen3-VL in this repo; "
            "disabling for this run."
        )
        data_args.data_flatten = False
        data_args.data_packing = False

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen3_vl_attention_class()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    set_model(model_args, model)

    _rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if _rank == 0:
        vis = getattr(model, "visual", None)
        if vis is not None and hasattr(vis, "print_trainable_parameters"):
            vis.print_trainable_parameters()
        inner = getattr(model, "language_model", None) or getattr(model, "model", None)
        if inner is not None and hasattr(inner, "print_trainable_parameters"):
            inner.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
