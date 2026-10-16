#!/usr/bin/env python3
"""
GRPO Stage 2 Offline Training Script - Spatial Reasoning Enhancement
==================================================================

Architecture:
  - Stage 1 offline: 3D information (Pi3+SAM3) is pre-computed and stored in JSONL data
  - Stage 2 online: GRPO trains the model to reason based on 3D info + images

Training Flow (per training step):
  1. Sample a prompt (all GPUs sync on the same prompt)
  2. Each GPU generates num_generations // num_gpus completions
  3. all_gather all completions, compute reward
  4. Group normalize to obtain advantage
  5. Each GPU computes GRPO loss for local completions
  6. DeepSpeed gradient_accumulation + all_reduce → optimizer step

Configuration Requirements:
  - 2 x GPU, DeepSpeed ZeRO-2
  - gradient_accumulation_steps = num_generations // num_gpus = 4
  - train_micro_batch_size_per_gpu = 1

Usage:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \\
      train.py --data_path data.jsonl --output_dir outputs/
"""

import os
import sys
import gc
import json
import shutil
import re
import math
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.distributed as dist
import numpy as np
import deepspeed

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup,
)
from qwen_vl_utils import process_vision_info

# ======================================================================
#  Logging
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpo_stage2")


# ======================================================================
#  Dataset
# ======================================================================
class GRPOStage2Dataset:
    """Load Stage 2 GRPO training data (pre-computed 3D information)."""

    def __init__(self, data_path: str, max_samples: int = -1):
        self.samples: List[Dict[str, Any]] = []
        required_keys = ["question", "images", "formatted_3d", "gt_answer", "sample_type"]

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(k in sample for k in required_keys):
                    if len(sample.get("images", [])) > 1:  # exclude single frame
                        self.samples.append(sample)

        if 0 < max_samples < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        self.pos_count = sum(1 for s in self.samples if s["sample_type"] == "positive")
        self.neg_count = len(self.samples) - self.pos_count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ======================================================================
#  Reward Function
# ======================================================================
def parse_answer(response: str) -> str:
    """Parse A/B/C/D from model response."""
    if not response:
        return "UNKNOWN"
    patterns = [
        r"<answer>\s*([A-D])\s*</answer>",
        r"[Aa]nswer[:\s]+\**([A-D])",
        r"[Tt]he answer is[:\s]+\**([A-D])",
    ]
    for pattern in patterns:
        m = re.search(pattern, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: last isolated A-D
    matches = list(re.finditer(r"\b([A-D])\b", response))
    if matches:
        return matches[-1].group(1).upper()
    return "UNKNOWN"


def compute_rewards(
    completions: List[str],
    gt_answer: str,
    sample_type: str,
    accuracy_weight_positive: float = 1.0,
    accuracy_weight_negative: float = 2.0,
    format_weight: float = 0.5,
) -> List[float]:
    """
    Compute rewards for a list of completions.

    Positive (baseline wrong + 3D correct): accuracy_weight_positive points if correct
    Negative (baseline correct + 3D wrong): accuracy_weight_negative points if correct
    Format (contains <answer>X</answer>): format_weight points
    """
    rewards = []
    gt_upper = gt_answer.strip().upper()

    for comp in completions:
        parsed = parse_answer(comp)
        is_correct = parsed == gt_upper

        # Accuracy reward
        if sample_type == "positive":
            acc_reward = accuracy_weight_positive if is_correct else 0.0
        else:  # negative
            acc_reward = accuracy_weight_negative if is_correct else 0.0

        # Format reward
        fmt_reward = 0.0
        if re.search(r"<answer>\s*[A-D]\s*</answer>", comp, re.IGNORECASE):
            fmt_reward = format_weight

        rewards.append(acc_reward + fmt_reward)

    return rewards


# ======================================================================
#  Message Construction
# ======================================================================
def build_messages(sample: Dict[str, Any]) -> List[Dict]:
    """Construct Stage 2 conversational messages (system + user with images + 3D info)."""
    system_prompt = sample.get(
        "stage2_system",
        "You are a spatial reasoning assistant with 3D perception.\n"
        "Use both the images and the 3D spatial data to answer. "
        "Think step by step and combine visual cues with 3D information. "
        "You MUST end with <answer>X</answer> where X is A, B, C, or D.",
    )
    user_prompt = sample.get(
        "stage2_prompt",
        f"Question: {sample['question']}\n\n"
        f"3D Spatial Information:\n{sample['formatted_3d']}\n\n"
        "Answer the question by combining the images above with the "
        "3D spatial information. Think step by step, then give your final answer.\n\n"
        "Reasoning:",
    )

    user_content: List[Dict] = []
    for img_path in sample["images"]:
        user_content.append({"type": "image", "image": img_path})
    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages


# ======================================================================
#  GRPO Loss
# ======================================================================
def get_per_token_logps(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, **mm_kwargs
) -> torch.Tensor:
    """Compute log probability for each token."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **mm_kwargs,
    )
    logits = outputs.logits[:, :-1, :]        # (B, L-1, V)
    targets = input_ids[:, 1:]                 # (B, L-1)
    # Compute row by row to reduce memory peaks
    per_token_logps = []
    for logits_row, target_row in zip(logits, targets):
        log_probs = logits_row.log_softmax(dim=-1)
        token_logp = torch.gather(log_probs, dim=1, index=target_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_logp)
    return torch.stack(per_token_logps)  # (B, L-1)


def compute_grpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: Optional[torch.Tensor],
    completion_mask: torch.Tensor,
    advantage: float,
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """Compute GRPO clipped surrogate loss for a single completion."""
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    adv = torch.tensor(advantage, device=per_token_logps.device, dtype=per_token_logps.dtype)
    surr1 = ratio * adv
    surr2 = clipped_ratio * adv
    per_token_loss = -torch.min(surr1, surr2)

    # KL Penalty
    if beta > 0 and ref_per_token_logps is not None:
        kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1.0
        )
        per_token_loss = per_token_loss + beta * kl

    # Masked average
    denom = completion_mask.sum(dim=1).clamp(min=1.0)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()
    return loss


# ======================================================================
#  Generation Mode Context Manager
# ======================================================================
@contextmanager
def generation_mode(model):
    """Temporarily switch to generation mode: eval + use_cache + disable gradient checkpointing."""
    was_training = model.training
    orig_use_cache = getattr(model.config, "use_cache", False)

    # Detect if gradient checkpointing is enabled (version compatible)
    had_gc = getattr(model, "is_gradient_checkpointing", False) or \
             getattr(model, "gradient_checkpointing", False)

    model.eval()
    model.config.use_cache = True
    if had_gc:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    try:
        yield model
    finally:
        model.config.use_cache = False
        if had_gc:
            try:
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception:
                pass
        if was_training:
            model.train()


# ======================================================================
#  Trainer
# ======================================================================
class GRPOStage2Trainer:
    """GRPO Stage 2 Trainer (independent of TRL, from scratch)."""

    def __init__(self, args):
        self.args = args
        self._setup()

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------
    def _setup(self):
        args = self.args

        # ---------- Distributed ----------
        deepspeed.init_distributed()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.is_main = self.global_rank == 0

        # Data order uses the same seed (synced across GPUs), generation uses diff seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed + self.local_rank + 100)
        torch.cuda.manual_seed_all(args.seed + self.local_rank + 100)

        if self.is_main:
            logger.info(f"World size: {self.world_size}, Local rank: {self.local_rank}")

        # ---------- Data ----------
        self.dataset = GRPOStage2Dataset(args.data_path, max_samples=args.max_samples)
        if self.is_main:
            logger.info(
                f"Dataset: {len(self.dataset)} items "
                f"(Positive {self.dataset.pos_count}, Negative {self.dataset.neg_count})"
            )

        if len(self.dataset) == 0:
            raise ValueError("No valid multi-image training samples found")

        # ---------- Processor ----------
        self.processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        # ---------- Model ----------
        if self.is_main:
            logger.info(f"Loading training model: {args.model_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        # Gradient checkpointing requires turn off use_cache
        model.config.use_cache = False

        # Freeze vision module
        if args.freeze_vision:
            frozen_count = 0
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False
                    frozen_count += 1
            if self.is_main:
                logger.info(f"Froze vision modules: {frozen_count} parameter groups")

        # Gradient checkpointing
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if self.is_main:
            logger.info(f"Params: {trainable:,} trainable / {total:,} total")

        # ---------- Reference Model (KL Penalty) ----------
        self.ref_model = None
        if args.beta > 0:
            if self.is_main:
                logger.info("Loading reference model (for KL penalty)")
            self.ref_model = AutoModelForImageTextToText.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            self.ref_model.config.use_cache = False  # Keep consistent with training model
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
            self.ref_model.to(self.device)

        # ---------- Optimizer & Scheduler ----------
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )

        total_steps = len(self.dataset) * args.num_epochs
        warmup_steps = min(args.warmup_steps, total_steps // 10)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ---------- DeepSpeed ----------
        with open(args.ds_config, "r") as f:
            ds_config = json.load(f)
        # Verify gradient_accumulation_steps match
        if args.num_generations < 2 or args.num_generations % self.world_size:
            raise ValueError("num_generations must be >=2 and divisible by world_size")
        self.num_gen_per_gpu = args.num_generations // self.world_size
        ds_config["gradient_accumulation_steps"] = self.num_gen_per_gpu
        ds_config["train_micro_batch_size_per_gpu"] = 1

        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
            model_parameters=trainable_params,
        )

        if self.is_main:
            logger.info(
                f"DeepSpeed Init Complete | "
                f"ZeRO stage: {ds_config['zero_optimization']['stage']} | "
                f"gradient_accumulation_steps: {ds_config['gradient_accumulation_steps']} | "
                f"num_gen_per_gpu: {self.num_gen_per_gpu}"
            )

        # ---------- Training State ----------
        self.global_step = 0
        self.start_epoch = 0
        self._metrics_log: List[Dict] = []

        # Attempt to resume
        self._try_resume()

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------
    def _try_resume(self):
        """Attempt to resume from latest DeepSpeed checkpoint."""
        ckpt_dir = Path(self.args.output_dir) / "checkpoints"
        if not ckpt_dir.exists():
            return

        # Find the latest tag
        tags = sorted(
            [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if not tags:
            return

        latest_tag = tags[-1]
        if self.is_main:
            logger.info(f"Resuming from checkpoint: {latest_tag}")

        _, client_state = self.model_engine.load_checkpoint(str(ckpt_dir), tag=latest_tag)
        if client_state:
            self.global_step = client_state.get("global_step", 0)
            self.start_epoch = client_state.get("epoch", 0)
            if self.is_main:
                logger.info(f"  Resumed to step={self.global_step}, epoch={self.start_epoch}")

    def _save_checkpoint(self, tag: str, epoch: int):
        """Save DeepSpeed checkpoint + HuggingFace model. Keep max_checkpoints removing oldest."""
        ckpt_dir = Path(self.args.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        client_state = {
            "global_step": self.global_step,
            "epoch": epoch,
        }
        self.model_engine.save_checkpoint(str(ckpt_dir), tag=tag, client_state=client_state)

        # Simultaneously save HF format (main process only)
        if self.is_main:
            hf_dir = Path(self.args.output_dir) / "hf_model" / tag
            hf_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.model_engine.module
            unwrapped.save_pretrained(str(hf_dir))
            self.processor.save_pretrained(str(hf_dir))
            logger.info(f"  HF model saved: {hf_dir}")

        # Keep step_* type checkpoints, removing oldest once max_checkpoints is reached
        max_ckpts = getattr(self.args, "max_checkpoints", 15)
        if max_ckpts <= 0:
            return
        step_tags = sorted(
            [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if self.is_main and len(step_tags) > max_ckpts:
            for old_tag in step_tags[: len(step_tags) - max_ckpts]:
                old_ckpt = ckpt_dir / old_tag
                if old_ckpt.exists():
                    shutil.rmtree(old_ckpt)
                    if self.is_main:
                        logger.info(f"  Deleted old checkpoint: {old_tag}")
                old_hf = Path(self.args.output_dir) / "hf_model" / old_tag
                if old_hf.exists():
                    shutil.rmtree(old_hf)

    # ------------------------------------------------------------------
    #  Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_completions(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[str], List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate num_gen_per_gpu completions for a prompt.

        Returns:
            completions_text: Decoded text for each completion
            completions_ids:  Token id tensors for each completion
            prompt_ids:       Prompt token ids (1, P)
            mm_cache:         Multimodal cache (pixel_values, image_grid_thw)
        """
        messages = build_messages(sample)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        prompt_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }

        prompt_ids = prompt_inputs["input_ids"]  # (1, P)
        prompt_length = prompt_ids.size(1)

        # Cache multimodal inputs
        mm_cache: Dict[str, torch.Tensor] = {}
        for key in ["pixel_values", "image_grid_thw"]:
            if key in prompt_inputs:
                mm_cache[key] = prompt_inputs[key]

        completions_text: List[str] = []
        completions_ids: List[torch.Tensor] = []

        raw_model = self.model_engine.module

        with generation_mode(raw_model):
            for _ in range(self.num_gen_per_gpu):
                output_ids = raw_model.generate(
                    **prompt_inputs,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=True,
                    temperature=self.args.temperature,
                    top_p=0.95,
                    pad_token_id=self.pad_token_id,
                )
                comp_ids = output_ids[0, prompt_length:]
                comp_text = self.processor.decode(comp_ids, skip_special_tokens=True)

                completions_text.append(comp_text)
                completions_ids.append(comp_ids.detach())

                del output_ids

        torch.cuda.empty_cache()
        return completions_text, completions_ids, prompt_ids, mm_cache

    # ------------------------------------------------------------------
    #  Single Completion Loss
    # ------------------------------------------------------------------
    def _compute_single_loss(
        self,
        prompt_ids: torch.Tensor,       # (1, P)
        comp_ids: torch.Tensor,          # (C,)
        advantage: float,
        mm_cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute GRPO loss for one prompt + completion."""
        prompt_length = prompt_ids.size(1)
        comp_length = comp_ids.size(0)

        # Concat input_ids
        input_ids = torch.cat([prompt_ids[0], comp_ids]).unsqueeze(0)  # (1, P+C)
        attention_mask = torch.ones_like(input_ids)

        # Completion mask (shifted logps space)
        total_len = input_ids.size(1) - 1  # shifted
        comp_mask = torch.zeros(1, total_len, device=self.device, dtype=torch.bfloat16)
        comp_start = prompt_length - 1  # shifted offset
        comp_end = comp_start + comp_length

        # EOS truncation: mask down to first EOS
        eos_positions = (comp_ids == self.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            actual_comp_len = eos_positions[0].item() + 1  # include EOS
        else:
            actual_comp_len = comp_length
        comp_end = min(comp_start + actual_comp_len, total_len)
        comp_mask[0, comp_start:comp_end] = 1.0

        if comp_mask.sum() < 1:
            # Empty completion, return zero loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Current policy log probs
        per_token_logps = get_per_token_logps(
            self.model_engine, input_ids, attention_mask, **mm_cache
        )
        # old logps = detach current (num_iterations=1)
        old_per_token_logps = per_token_logps.detach()

        # Reference policy log probs
        ref_per_token_logps = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, **mm_cache
                )

        # GRPO loss
        loss = compute_grpo_loss(
            per_token_logps,
            old_per_token_logps,
            ref_per_token_logps,
            comp_mask,
            advantage,
            epsilon=self.args.epsilon,
            beta=self.args.beta,
        )

        return loss

    # ------------------------------------------------------------------
    #  Training Step
    # ------------------------------------------------------------------
    def _train_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Processes full GRPO step for a sample. Returns dict with details."""
        step_start = time.time()

        # ---- Phase 1: Generation ----
        gen_start = time.time()
        completions_text, completions_ids, prompt_ids, mm_cache = (
            self._generate_completions(sample)
        )
        gen_time = time.time() - gen_start

        # ---- Phase 2: Cross-GPU Gather ----
        all_completions_gathered: List[Optional[List[str]]] = [None] * self.world_size
        dist.all_gather_object(all_completions_gathered, completions_text)
        all_completions: List[str] = []
        for gpu_comps in all_completions_gathered:
            all_completions.extend(gpu_comps)

        # ---- Phase 3: Compute Rewards + Advantages ----
        rewards = compute_rewards(
            all_completions,
            sample["gt_answer"],
            sample["sample_type"],
            accuracy_weight_positive=self.args.accuracy_weight_positive,
            accuracy_weight_negative=self.args.accuracy_weight_negative,
            format_weight=self.args.format_weight,
        )

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean().item()
        std_r = rewards_t.std().item()
        min_r = rewards_t.min().item()
        max_r = rewards_t.max().item()
        advantages = ((rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)).tolist()

        # Advantage for current GPU
        local_start = self.global_rank * self.num_gen_per_gpu
        local_advantages = advantages[local_start: local_start + self.num_gen_per_gpu]

        # ---- Phase 4: GRPO loss + backward + step ----
        train_start = time.time()
        self.model_engine.module.train()
        step_losses = []

        for i in range(self.num_gen_per_gpu):
            try:
                loss = self._compute_single_loss(
                    prompt_ids, completions_ids[i], local_advantages[i], mm_cache
                )
                self.model_engine.backward(loss)
                self.model_engine.step()
                step_losses.append(loss.item())
            except RuntimeError:
                # Distributed backward must not continue with unsynchronized dummy gradients.
                raise
            finally:
                # Clear intermediate tensors
                torch.cuda.empty_cache()

        train_time = time.time() - train_start
        total_time = time.time() - step_start

        # ---- Aggregate all info ----
        gt = sample["gt_answer"].strip().upper()
        parsed_answers = [parse_answer(c) for c in all_completions]
        correct_flags = [p == gt for p in parsed_answers]
        accuracy = sum(correct_flags) / max(len(correct_flags), 1)
        avg_loss = sum(step_losses) / max(len(step_losses), 1)
        lr = self.optimizer.param_groups[0]["lr"]

        # GPU memory
        gpu_mem_used = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        gpu_mem_total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats(self.device)

        # Find best completion
        best_idx = int(rewards_t.argmax().item())

        return {
            # Core metrics
            "loss": avg_loss,
            "reward_mean": mean_r,
            "reward_std": std_r,
            "reward_min": min_r,
            "reward_max": max_r,
            "accuracy": accuracy,
            "lr": lr,
            # Times
            "gen_time": gen_time,
            "train_time": train_time,
            "total_time": total_time,
            # Sample info
            "sample_type": sample["sample_type"],
            "gt_answer": gt,
            "question": sample["question"][:120],  # truncated
            "num_images": len(sample.get("images", [])),
            # Per-generation details
            "all_rewards": rewards,
            "all_advantages": advantages,
            "all_parsed_answers": parsed_answers,
            "all_correct": correct_flags,
            "all_completions": all_completions,
            # Best completion
            "best_idx": best_idx,
            "best_reward": rewards[best_idx],
            "best_answer": parsed_answers[best_idx],
            "best_completion": all_completions[best_idx],
            # Memory
            "gpu_mem_gb": gpu_mem_used,
            "gpu_mem_total_gb": gpu_mem_total,
            # Loss details
            "step_losses": step_losses,
        }

    # ------------------------------------------------------------------
    #  Detailed Logging
    # ------------------------------------------------------------------
    def _print_step_details(
        self, step_info: Dict[str, Any], global_step: int, epoch: int,
        local_step: int, total_in_epoch: int,
    ):
        """Print detailed training info at each step (main process only)."""
        gt = step_info["gt_answer"]
        rewards = step_info["all_rewards"]
        advantages = step_info["all_advantages"]
        parsed = step_info["all_parsed_answers"]
        correct = step_info["all_correct"]
        n_gen = len(rewards)
        n_per_gpu = n_gen // self.world_size

        lines = []
        lines.append(f"\n{'═' * 80}")
        lines.append(
            f"  Step {global_step} | Epoch {epoch} "
            f"[{local_step}/{total_in_epoch}] | "
            f"Type: {step_info['sample_type']} | GT: {gt} | "
            f"Images: {step_info['num_images']}"
        )
        lines.append(f"  Q: {step_info['question']}...")
        lines.append(f"{'─' * 80}")

        # Per generation rewards
        lines.append("  Generations:")
        for gpu_id in range(self.world_size):
            parts = []
            for j in range(n_per_gpu):
                idx = gpu_id * n_per_gpu + j
                r = rewards[idx]
                ans = parsed[idx]
                ok = "✓" if correct[idx] else "✗"
                parts.append(f"G{idx}={r:.2f}({ans}{ok})")
            lines.append(f"    GPU{gpu_id}: {' | '.join(parts)}")

        # reward/advantage statistics
        lines.append(
            f"  Reward:    mean={step_info['reward_mean']:.3f}  "
            f"std={step_info['reward_std']:.3f}  "
            f"min={step_info['reward_min']:.2f}  "
            f"max={step_info['reward_max']:.2f}"
        )
        adv_str = " ".join(f"{a:+.3f}" for a in advantages)
        lines.append(f"  Advantage: [{adv_str}]")

        # loss details
        loss_str = " ".join(f"{l:.4f}" for l in step_info["step_losses"])
        lines.append(
            f"  Loss:      avg={step_info['loss']:.4f}  "
            f"per_comp=[{loss_str}]"
        )

        # training hyperparameters
        lines.append(
            f"  LR: {step_info['lr']:.2e} | "
            f"Beta: {self.args.beta} | Eps: {self.args.epsilon} | "
            f"Temp: {self.args.temperature}"
        )
        lines.append(
            f"  GPU Mem: {step_info['gpu_mem_gb']:.1f}/{step_info['gpu_mem_total_gb']:.0f} GB | "
            f"Time: {step_info['total_time']:.1f}s "
            f"(gen={step_info['gen_time']:.1f}s train={step_info['train_time']:.1f}s)"
        )

        # Best completion full display
        best = step_info["best_completion"]
        best_r = step_info["best_reward"]
        best_ans = step_info["best_answer"]
        best_ok = "✓" if best_ans == gt else "✗"
        lines.append(f"{'─' * 80}")
        lines.append(
            f"  ★ Best Completion (G{step_info['best_idx']}, "
            f"R={best_r:.2f}, Ans={best_ans}{best_ok}):"
        )
        # Limit display length
        best_display = best if len(best) <= 500 else best[:250] + "\n    ...\n    " + best[-200:]
        for bline in best_display.split("\n"):
            lines.append(f"    {bline}")
        lines.append(f"{'═' * 80}")

        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    #  Main Training Loop
    # ------------------------------------------------------------------
    def train(self):
        args = self.args

        if self.is_main:
            logger.info("=" * 70)
            logger.info("GRPO Stage 2 Training Started")
            logger.info(f"  Dataset size:       {len(self.dataset)}")
            logger.info(f"  Total epochs:       {args.num_epochs}")
            logger.info(f"  Gen per prompt:     {args.num_generations}")
            logger.info(f"  Gen per GPU:        {self.num_gen_per_gpu}")
            logger.info(f"  Learning rate:      {args.learning_rate}")
            logger.info(f"  Beta (KL):          {args.beta}")
            logger.info(f"  Epsilon (clip):     {args.epsilon}")
            logger.info(f"  Temperature:        {args.temperature}")
            logger.info(f"  Freeze vision:      {args.freeze_vision}")
            logger.info(f"  Pos accuracy reward:{args.accuracy_weight_positive}")
            logger.info(f"  Neg accuracy reward:{args.accuracy_weight_negative}")
            logger.info(f"  Format reward:      {args.format_weight}")
            logger.info(f"  Output dir:         {args.output_dir}")
            logger.info("=" * 70)

        # Log file
        log_file = None
        if self.is_main:
            log_dir = Path(args.output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(
                log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                "w",
                encoding="utf-8",
            )

        total_steps_done = self.global_step

        for epoch in range(self.start_epoch, args.num_epochs):
            # Data order for this epoch (consistent across all GPUs)
            epoch_rng = random.Random(args.seed + epoch)
            indices = list(range(len(self.dataset)))
            epoch_rng.shuffle(indices)

            # Skip completed steps
            steps_to_skip = total_steps_done - epoch * len(self.dataset)
            if steps_to_skip > 0 and steps_to_skip < len(indices):
                indices = indices[steps_to_skip:]
                if self.is_main:
                    logger.info(f"Epoch {epoch + 1}: Skipping first {steps_to_skip} steps")
            elif steps_to_skip >= len(indices):
                if self.is_main:
                    logger.info(f"Epoch {epoch + 1}: Completed, skipping")
                continue

            epoch_metrics: List[Dict] = []

            for local_step, idx in enumerate(indices):
                sample = self.dataset[idx]

                try:
                    metrics = self._train_step(sample)
                    epoch_metrics.append(metrics)
                    self.global_step += 1

                    # ---- Print detailed log every step ----
                    if self.is_main:
                        self._print_step_details(
                            metrics, self.global_step, epoch + 1,
                            local_step + 1, len(indices),
                        )

                    # ---- Write to JSONL log file ----
                    if self.is_main and log_file:
                        # Write reduced version (without full completion text to save space)
                        log_entry = {
                            "global_step": self.global_step,
                            "epoch": epoch + 1,
                            "loss": metrics["loss"],
                            "reward_mean": metrics["reward_mean"],
                            "reward_std": metrics["reward_std"],
                            "reward_min": metrics["reward_min"],
                            "reward_max": metrics["reward_max"],
                            "accuracy": metrics["accuracy"],
                            "lr": metrics["lr"],
                            "sample_type": metrics["sample_type"],
                            "gt_answer": metrics["gt_answer"],
                            "all_rewards": metrics["all_rewards"],
                            "all_parsed_answers": metrics["all_parsed_answers"],
                            "all_correct": metrics["all_correct"],
                            "all_advantages": metrics["all_advantages"],
                            "step_losses": metrics["step_losses"],
                            "best_idx": metrics["best_idx"],
                            "best_reward": metrics["best_reward"],
                            "best_answer": metrics["best_answer"],
                            "gpu_mem_gb": metrics["gpu_mem_gb"],
                            "gen_time": metrics["gen_time"],
                            "train_time": metrics["train_time"],
                            "total_time": metrics["total_time"],
                        }
                        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        log_file.flush()

                    # ---- Sliding window summary (every log_every steps) ----
                    if self.is_main and self.global_step % args.log_every == 0:
                        window = epoch_metrics[-args.log_every:]
                        w_loss = np.mean([m["loss"] for m in window])
                        w_reward = np.mean([m["reward_mean"] for m in window])
                        w_acc = np.mean([m["accuracy"] for m in window])
                        w_pos = np.mean(
                            [m["accuracy"] for m in window if m["sample_type"] == "positive"] or [0]
                        )
                        w_neg = np.mean(
                            [m["accuracy"] for m in window if m["sample_type"] == "negative"] or [0]
                        )
                        logger.info(
                            f"  📊 Sliding Window (last {len(window)} steps): "
                            f"Loss={w_loss:.4f} | Reward={w_reward:.3f} | "
                            f"Acc={w_acc:.0%} (pos={w_pos:.0%} neg={w_neg:.0%})"
                        )

                    # ---- Save checkpoint ----
                    if self.global_step % args.save_every == 0:
                        self._save_checkpoint(f"step_{self.global_step}", epoch)
                        if self.is_main:
                            logger.info(f"  💾 Checkpoint saved: step_{self.global_step}")

                except Exception as e:
                    logger.error(
                        f"[Rank {self.local_rank}] Error at step {self.global_step}: {e}"
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

                    raise

            # Epoch summary
            if self.is_main and epoch_metrics:
                avg_loss = np.mean([m["loss"] for m in epoch_metrics])
                avg_reward = np.mean([m["reward_mean"] for m in epoch_metrics])
                avg_acc = np.mean([m["accuracy"] for m in epoch_metrics])
                pos_acc = np.mean(
                    [m["accuracy"] for m in epoch_metrics if m["sample_type"] == "positive"]
                    or [0]
                )
                neg_acc = np.mean(
                    [m["accuracy"] for m in epoch_metrics if m["sample_type"] == "negative"]
                    or [0]
                )
                logger.info(
                    f"\n{'=' * 70}\n"
                    f"Epoch {epoch + 1} Summary:\n"
                    f"  Loss:     {avg_loss:.4f}\n"
                    f"  Reward:   {avg_reward:.3f}\n"
                    f"  Accuracy: {avg_acc:.1%} (Positive {pos_acc:.1%} | Negative {neg_acc:.1%})\n"
                    f"{'=' * 70}\n"
                )

            # Save epoch checkpoint (save on multiple epochs; skip for single epoch, only save final)
            if args.num_epochs > 1:
                self._save_checkpoint(f"epoch_{epoch + 1}", epoch + 1)

        # Final save (always save to hf_model/final/)
        self._save_checkpoint("final", args.num_epochs)
        if self.is_main:
            logger.info("Training complete! Final model saved.")
            if log_file:
                log_file.close()


# ======================================================================
#  Argument Parsing
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Stage 2 Training")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Training data JSONL path")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples, -1=all")

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument("--freeze_vision", action="store_true", default=True,
                        help="Freeze vision encoder")
    parser.add_argument("--no_freeze_vision", dest="freeze_vision", action="store_false")

    # DeepSpeed
    parser.add_argument(
        "--ds_config",
        type=str,
        default=str(Path(__file__).resolve().parent / "ds_config_zero2.json"),
    )

    # GRPO Hyperparams
    parser.add_argument("--num_generations", type=int, default=8, help="Generations per prompt")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")

    # Reward weights
    parser.add_argument("--accuracy_weight_positive", type=float, default=1.0,
                        help="Positive sample accuracy reward")
    parser.add_argument("--accuracy_weight_negative", type=float, default=2.0,
                        help="Negative sample accuracy reward")
    parser.add_argument("--format_weight", type=float, default=0.5,
                        help="Format correct reward")

    # Training hyperparams
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Output and logging
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_every", type=int, default=5, help="Print log every N steps")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--max_checkpoints", type=int, default=0, help="Max step checkpoints to keep; 0 retains all")
    parser.add_argument("--skip_errors", action="store_true", default=False,
                        help="Skip error samples and continue training")
    parser.add_argument("--no_skip_errors", dest="skip_errors", action="store_false")

    # DeepSpeed launcher injects --local_rank
    parser.add_argument("--local_rank", type=int, default=-1, help="(Auto injected by DeepSpeed)")

    return parser.parse_args()


# ======================================================================
#  Entry Point
# ======================================================================
if __name__ == "__main__":
    args = parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.skip_errors:
        raise ValueError("--skip_errors is not supported for distributed training; fix the failed sample and resume.")
    trainer = GRPOStage2Trainer(args)
    trainer.train()
