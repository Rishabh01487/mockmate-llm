"""
train_qlora.py
==============

QLoRA fine-tuning for `deepseek-ai/deepseek-coder-6.7b-base` on the
ChatML-formatted LeetCode dataset produced by `prepare_data.py`.

Stack
-----
- transformers (4.44) AutoModelForCausalLM + BitsAndBytes 4-bit NF4 quant
- peft  LoraConfig (targeting q/k/v/o + gate/up/down proj)
- trl   SFTTrainer with completion-only loss masking (DataCollatorForCompletionOnlyLM)
- accelerate (no deepspeed / no FSDP needed at 7B + QLoRA on 24GB)

Memory budget on RTX 3090/4090 (24GB)
------------------------------------
- Base weights (4-bit):        ~3.5 GB
- LoRA params (trainable):     ~80 MB
- Activations (bs=4, seq=1024): ~12 GB
- Optimizer states (8-bit Adam): ~1 GB
- Buffers / fragmentation:    ~2 GB
Total: ~19 GB → safe headroom.

Usage
-----
    # 1) Activate your env, then:
    export HF_HOME=./hf_cache
    export TOKENIZERS_PARALLELISM=false
    python train_qlora.py \
        --data_dir ./data_processed \
        --base_model deepseek-ai/deepseek-coder-6.7b-base \
        --output_dir ./checkpoints/deepseek-leetcode-qlora \
        --epochs 3 --batch_size 4 --grad_accum 4 --lr 2e-4 --max_seq_len 1024

    # 2) Push to hub (optional):
    python train_qlora.py --push --hub_repo your-username/deepseek-leetcode-qlora \
        --output_dir ./checkpoints/deepseek-leetcode-qlora

Notes
-----
- The script auto-falls back to `codellama/CodeLlama-7b-hf` if the primary
  model is gated or unreachable.
- `--use_flash_attention` requires flash-attn 2.x; off by default for portability.
- Resume from any checkpoint by passing `--resume_from_checkpoint`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


# =========================================================================
# Constants
# =========================================================================
DEFAULT_BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-base"
FALLBACK_BASE_MODEL = "codellama/CodeLlama-7b-hf"

# LoRA targets — covers DeepSeek & Llama architectures (q/k/v/o + mlp gate/up/down)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ChatML tokens (DeepSeek-Coder-Base is not natively ChatML, but we add the
# special tokens so it learns the format from scratch.)
CHATML_SPECIAL_TOKENS = {
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
}

# The response template — used by DataCollatorForCompletionOnlyLM to mask
# everything BEFORE this string in the chatml_text, so loss is only computed
# on the assistant's response (including the code block + Big-O block).
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# =========================================================================
# Args
# =========================================================================
@dataclass
class TrainArgs:
    data_dir: str = "./data_processed"
    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = "./checkpoints/deepseek-leetcode-qlora"
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    max_seq_len: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    use_flash_attention: bool = False
    bf16: bool = True
    seed: int = 42
    hub_repo: Optional[str] = None
    push: bool = False
    resume_from_checkpoint: Optional[str] = None


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="QLoRA fine-tune deepseek-coder on LeetCode.")
    p.add_argument("--data_dir", type=str, default=TrainArgs.data_dir)
    p.add_argument("--base_model", type=str, default=TrainArgs.base_model)
    p.add_argument("--output_dir", type=str, default=TrainArgs.output_dir)
    p.add_argument("--epochs", type=int, default=TrainArgs.epochs)
    p.add_argument("--batch_size", type=int, default=TrainArgs.batch_size)
    p.add_argument("--grad_accum", type=int, default=TrainArgs.grad_accum)
    p.add_argument("--lr", type=float, default=TrainArgs.lr)
    p.add_argument("--max_seq_len", type=int, default=TrainArgs.max_seq_len)
    p.add_argument("--warmup_ratio", type=float, default=TrainArgs.warmup_ratio)
    p.add_argument("--weight_decay", type=float, default=TrainArgs.weight_decay)
    p.add_argument("--max_grad_norm", type=float, default=TrainArgs.max_grad_norm)
    p.add_argument("--lora_r", type=int, default=TrainArgs.lora_r)
    p.add_argument("--lora_alpha", type=int, default=TrainArgs.lora_alpha)
    p.add_argument("--lora_dropout", type=float, default=TrainArgs.lora_dropout)
    p.add_argument("--save_steps", type=int, default=TrainArgs.save_steps)
    p.add_argument("--eval_steps", type=int, default=TrainArgs.eval_steps)
    p.add_argument("--logging_steps", type=int, default=TrainArgs.logging_steps)
    p.add_argument("--use_flash_attention", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=TrainArgs.seed)
    p.add_argument("--hub_repo", type=str, default=None)
    p.add_argument("--push", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    ns = p.parse_args()
    return TrainArgs(**{k: getattr(ns, k) for k in TrainArgs.__annotations__})


# =========================================================================
# Model + tokenizer loading
# =========================================================================
def load_tokenizer(model_name: str, max_seq_len: int):
    print(f"[tokenizer] Loading {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Add ChatML special tokens if not present
    n_added = tok.add_special_tokens(CHATML_SPECIAL_TOKENS)
    print(f"[tokenizer] Added {n_added} new special tokens "
          f"(total vocab now = {len(tok)})")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(args: TrainArgs, vocab_size: int):
    print(f"[model] Loading {args.base_model} in 4-bit NF4 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    attn_impl = "flash_attention_2" if args.use_flash_attention else "sdpa"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        print(f"[model] Failed to load {args.base_model}: {e}", file=sys.stderr)
        if args.base_model == DEFAULT_BASE_MODEL:
            print(f"[model] Falling back to {FALLBACK_BASE_MODEL}", file=sys.stderr)
            args.base_model = FALLBACK_BASE_MODEL
            return load_model(args, vocab_size)
        raise

    # Resize embeddings to account for the new <|im_start|> / <|im_end|> tokens
    model.resize_token_embeddings(vocab_size)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "config"):
        model.config.use_cache = False  # conflicts with grad-checkpointing
    return model


def build_lora_config(args: TrainArgs) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


# =========================================================================
# Dataset loading
# =========================================================================
def load_dataset(data_dir: str):
    path = Path(data_dir) / "chatml_dataset"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run prepare_data.py first.")
    dd = load_from_disk(str(path))
    print(f"[data] Loaded {dd}")
    # Map chatml_text -> the field SFTTrainer expects by default ('text')
    def _rename(ex):
        ex["text"] = ex["chatml_text"]
        return ex
    dd = dd.map(_rename, remove_columns=["chatml_text"])
    return dd


# =========================================================================
# Main training loop
# =========================================================================
def train(args: TrainArgs) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)

    # --- data ---
    ds = load_dataset(args.data_dir)
    print(f"[data] train={len(ds['train'])} val={len(ds['val'])}")

    # --- tokenizer + model ---
    tok = load_tokenizer(args.base_model, args.max_seq_len)
    model = load_model(args, vocab_size=len(tok))
    lora_cfg = build_lora_config(args)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- trainer config ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        report_to="tensorboard",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
    )

    # --- completion-only loss masking ---
    response_template_ids = tok.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tok,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=lora_cfg,
        processing_class=tok,
        data_collator=collator,
    )

    # --- kick off training ---
    print("[train] Starting training ...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # --- merge & export a single .safetensors (optional) ---
    # Note: merging a 4-bit model requires reloading in fp16/bf16 first.
    # We skip merge here for VRAM safety; do it offline with merge_adapter.py.

    # --- push to hub ---
    if args.push and args.hub_repo:
        print(f"[push] Uploading LoRA adapter to {args.hub_repo}")
        trainer.push_to_hub(args.hub_repo)

    print(f"\n[done] Adapter saved to: {args.output_dir}")
    print("Next steps:")
    print(f"  1. Smoke test:   python inference.py --adapter {args.output_dir}")
    print(f"  2. Launch API:   uvicorn app_api:app --port 8000 "
          f"--env ADAPTER_PATH={args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
