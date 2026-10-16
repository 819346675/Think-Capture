"""
MMSI-Bench Evaluation Script

Test Baseline and Agent performance on MMSI-Bench
"""

import os
import ast
import sys
import json
import time
import logging
import re
import base64
import io
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / "ours" / "evaluation") not in sys.path:
    sys.path.insert(0, str(_ROOT / "ours" / "evaluation"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Qwen3VLModel:
    """Qwen3-VL Model Wrapper"""

    def __init__(self,
                 model_path: str = "models/Qwen3-VL-4B-Instruct",
                 device: str = "cuda",
                 max_new_tokens: int = 2048,
                 checkpoint_path: str = None):
        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading Qwen3-VL model: {model_path}")
        if checkpoint_path:
            logger.info(f"  Will load GRPO checkpoint: {checkpoint_path}")

        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load trained weights if checkpoint provided
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path) / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

            logger.info(f"Loading checkpoint weights: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

            # DeepSpeed checkpoint format: weights under 'module' key
            if "module" in checkpoint:
                state_dict = checkpoint["module"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                # If top-level is state_dict
                state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor) or (isinstance(v, dict) and "weight" in str(v))}

            # Remove potential DeepSpeed prefixes
            if state_dict and any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Load weights
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=True)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
            logger.info("Checkpoint weights loaded")

        self.model.eval()
        logger.info("Model loaded")

    def generate(self, images: List[Image.Image], prompt: str, system_prompt: str = None) -> str:
        from qwen_vl_utils import process_vision_info

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)


def decode_images(image_data: str) -> List[Image.Image]:
    """Decode base64 images"""
    images_list = ast.literal_eval(image_data)
    if not isinstance(images_list, (list, tuple)) or not all(isinstance(x, str) for x in images_list):
        raise ValueError('Expected a list of base64-encoded images')
    images = []
    for img_b64 in images_list:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)
    return images


def parse_answer(response: str) -> str:
    """Parse answer"""
    patterns = [
        r'<answer>\s*([A-D])',
        r'[Aa]nswer[:\s]+\**([A-D])',
        r'[Tt]he answer is[:\s]+\**([A-D])',
        r'\*\*([A-D])\*\*',
        r'^([A-D])[.:\s]',
        r'\b([A-D])\b\s*$',
    ]
    for p in patterns:
        match = re.search(p, response, re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Try to find any A/B/C/D as fallback
    for match in re.finditer(r'\b([A-D])\b', response):
        return match.group(1).upper()

    return "UNKNOWN"


def run_baseline(vlm, images: List[Image.Image], question: str) -> Dict:
    """Run Baseline"""
    system_prompt = """You are a spatial reasoning assistant. Analyze the images carefully and answer the question.
For multiple choice questions, your final answer must be a single letter (A, B, C, or D).
Think step by step about the spatial relationships shown in the images, then provide your answer."""

    prompt = f"""Question: {question}

Look at the images carefully and answer the question. Think about the spatial relationships.

<reasoning>"""

    start = time.time()
    response = vlm.generate(images, prompt, system_prompt)
    elapsed = time.time() - start

    return {
        'response': response,
        'answer': parse_answer(response),
        'time': elapsed
    }


def run_agent(vlm, tool_wrapper, images: List[Image.Image], question: str, category: str = None, prompt_style: str = "eval") -> Dict:
    """Run Agent (with tool); category selects 3D template; prompt_style: eval=eval template, training=GRPO training template."""
    from agent_tool_wrapper import format_tool_result_for_model
    if prompt_style == "training":
        from stage2_prompt_constants import STAGE2_SYSTEM_TRAINING, build_stage2_prompt_training
    else:
        from stage2_prompt_constants import STAGE2_SYSTEM, build_stage2_prompt

    # Stage 1: Tell model to decide targets
    stage1_system = """You are a spatial reasoning assistant with access to a 3D perception tool.
Output format:
<thinking>Analyze what spatial information is needed.</thinking>
<objects_to_segment>
List objects that need 3D measurement, one per line.
Example:
- car
- person
</objects_to_segment>"""

    stage1_prompt = f"""Question: {question}

Which objects need 3D spatial measurement to answer this question?

<thinking>"""

    start = time.time()
    stage1_response = vlm.generate(images, stage1_prompt, stage1_system)

    # Parse objects
    objects = []
    pattern = r'<objects_to_segment>(.*?)</objects_to_segment>'
    match = re.search(pattern, stage1_response, re.DOTALL)
    if match:
        for line in match.group(1).strip().split('\n'):
            line = re.sub(r'^[-•*]\s*', '', line.strip())
            if line and len(line) > 1:
                objects.append(line.lower())

    if not objects:
        # Extract keywords as fallback common objects
        common = ["car", "person", "object", "camera", "scene", "building", "road"]
        q_lower = question.lower()
        objects = [w for w in common if w in q_lower][:3]
        if not objects:
            objects = ["object"]

    objects = objects[:5]

    tool_result = {"status": "error"}

    # Stage 2: Call tools
    # Save temporary images
    import tempfile
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    for i, img in enumerate(images):
        path = f"{temp_dir}/img_{i}.jpg"
        img.save(path)
        image_paths.append(path)

    tool_request = {
        "pipeline": "pi3+sam3",
        "semantic_targets": objects,
        "confidence_threshold": 0.3
    }

    try:
        tool_result = tool_wrapper.execute_tool_request(image_paths, tool_request)
        formatted_3d = format_tool_result_for_model(
            tool_result, max_per_category=2, question=question
        )
    except Exception as e:
        logger.warning(f"Tool execution failed: {e}")
        formatted_3d = "3D tool unavailable."

    # Debug: verify 3D Info is extracted into Stage 2 prompt
    preview = (formatted_3d or "")[:280].replace("\n", " ")
    logger.info(f"[3D injection into Stage2] len={len(formatted_3d or '')} chars, preview: {preview}...")

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Stage 3: Answer output
    if prompt_style == "training":
        stage2_system = STAGE2_SYSTEM_TRAINING
        stage2_prompt = build_stage2_prompt_training(question, formatted_3d)
    else:
        stage2_system = STAGE2_SYSTEM
        stage2_prompt = build_stage2_prompt(question, formatted_3d)

    response = vlm.generate(images, stage2_prompt, stage2_system)
    elapsed = time.time() - start

    return {
        'response': response,
        'answer': parse_answer(response),
        'objects_requested': objects,
        'formatted_3d': formatted_3d,
        'tool_status': tool_result.get('status', 'error'),
        'time': elapsed
    }


def main(num_samples: int = 20, run_agent_eval: bool = True, position_only: bool = False, prompt_style: str = "eval", checkpoint_path: str = None, agent_only: bool = False, per_category: int = 0, only_category: str = None):
    logger.info("="*60)
    logger.info("MMSI-Bench Evaluation" + (" [Position only]" if position_only else "") + (f" [Only {only_category}]" if only_category else ""))
    if prompt_style == "training":
        logger.info("Stage2 template: training (Think step by step)")
    else:
        logger.info("Stage2 template: eval (default)")
    logger.info("="*60)

    # Load dataset
    data_path = Path(os.environ.get("MMSI_TSV", "MMSI-Bench/data/MMSI_bench.tsv"))
    df = pd.read_csv(data_path, sep='\t')

    # Position relation only: filter specific geometries, motion, etc
    if position_only:
        df = df[df['category'].str.startswith('Positional Space')].reset_index(drop=True)
        logger.info(f"Position only: Total {len(df)} samples")

    # Only evaluate a single category
    if only_category:
        df = df[df['category'] == only_category].reset_index(drop=True)
        logger.info(f"Only category = {only_category}: Total {len(df)} samples")

    # Sample based on per_category counts
    if per_category > 0:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for idx, row in df.iterrows():
            by_cat[row['category']].append(idx)

        selected_indices = []
        selected_counts = {}
        rng = np.random.default_rng(42)
        for cat, indices in by_cat.items():
            k = min(per_category, len(indices))
            if k <= 0:
                continue
            chosen = rng.choice(indices, size=k, replace=False)
            selected_indices.extend(chosen.tolist())
            selected_counts[cat] = k

        df = df.loc[selected_indices].reset_index(drop=True)
        logger.info(f"Evaluating {per_category} samples per category, actual: {selected_counts}")
    else:
        # Evaluate subset unless category limited
        if not only_category and num_samples > 0 and num_samples < len(df):
            df = df.head(num_samples)

    logger.info(f"Evaluation samples count: {len(df)}")

    # Init model
    vlm = Qwen3VLModel(model_path=os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"), checkpoint_path=checkpoint_path)

    tool_wrapper = None
    if run_agent_eval:
        from agent_tool_wrapper import AgentToolWrapper
        tool_wrapper = AgentToolWrapper(
            sam3_checkpoint=os.environ.get("SAM3_CHECKPOINT"),
            pi3_checkpoint=os.environ.get("PI3_CHECKPOINT"),
            use_mock=False,
            device="cuda"
        )

    # Evaluation
    results = []
    baseline_correct = 0
    agent_correct = 0

    for i, row in df.iterrows():
        logger.info(f"\nSample {i+1}/{len(df)}: {row['category']}")

        try:
            images = decode_images(row['image'])
            question = row['question']
            gt = row['answer']

            # Baseline (skip if agent_only=True)
            baseline_result = None
            if not agent_only:
                baseline_result = run_baseline(vlm, images, question)
                b_correct = baseline_result['answer'] == gt
                if b_correct:
                    baseline_correct += 1
                logger.info(f"Baseline: {baseline_result['answer']} ({'✓' if b_correct else '✗'}) GT={gt}")

            result = {
                'index': row['index'],
                'category': row['category'],
                'gt': gt,
                'baseline': baseline_result
            }

            # Agent
            if run_agent_eval:
                agent_result = run_agent(vlm, tool_wrapper, images, question, prompt_style=prompt_style)
                a_correct = agent_result['answer'] == gt
                if a_correct:
                    agent_correct += 1
                logger.info(f"Agent: {agent_result['answer']} ({'✓' if a_correct else '✗'})")
                result['agent'] = agent_result

            results.append(result)

        except Exception as e:
            logger.error(f"Sample processing failed: {e}")
            results.append({'index': row['index'], 'category': row['category'], 'error': str(e)})

    # Summarize results
    n = len(df)
    if n == 0:
        raise ValueError('No evaluation samples selected')
    baseline_acc = None
    if not agent_only:
        baseline_acc = baseline_correct / n * 100

    print("\n" + "="*60)
    print("MMSI-Bench Evaluation Results")
    print("="*60)
    print(f"Sample count: {n}")
    if not agent_only:
        print(f"Baseline: {baseline_correct}/{n} = {baseline_acc:.1f}%")
    else:
        print("Baseline: Skipped (agent_only=True)")

    if run_agent_eval:
        agent_acc = agent_correct / n * 100
        print(f"Agent: {agent_correct}/{n} = {agent_acc:.1f}%")
        if baseline_acc is not None:
            print(f"Diff: {agent_acc - baseline_acc:+.1f}%")

    # Results by category
    POSITION_PREFIX = "Positional Space "
    if run_agent_eval and results:
        from collections import defaultdict
        by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            c = r.get('category', 'unknown')
            by_cat[c]['total'] += 1
            if 'agent' in r and r['agent'].get('answer') == r.get('gt'):
                by_cat[c]['correct'] += 1
        # List stats for position relation categories
        position_cats = [c for c in sorted(by_cat.keys()) if c.startswith(POSITION_PREFIX)]
        pos_correct = sum(by_cat[c]['correct'] for c in position_cats)
        pos_total = sum(by_cat[c]['total'] for c in position_cats)
        print("\n--- Agent Acc by category (Position only) ---")
        for c in position_cats:
            t = by_cat[c]['total']
            k = by_cat[c]['correct']
            print(f"  {c}: {k}/{t} = {100*k/t:.1f}%")
        if pos_total > 0:
            print(f"  Positional Total: {pos_correct}/{pos_total} = {100*pos_correct/pos_total:.1f}%")

    # Save
    output_dir = Path("outputs/mmsi_bench_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_position_only" if position_only else ""
    if prompt_style == "training":
        suffix = suffix + "_training_prompt"
    ckpt_suffix = ""
    if checkpoint_path:
        # Checkpoint id
        ckpt_name = Path(checkpoint_path).name
        ckpt_suffix = f"_ckpt_{ckpt_name}"

    # Position only accuracy summary
    agent_acc_position_only = None
    if run_agent_eval and results:
        by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            c = r.get('category', 'unknown')
            by_cat[c]['total'] += 1
            if 'agent' in r and r['agent'].get('answer') == r.get('gt'):
                by_cat[c]['correct'] += 1
        position_cats = [c for c in by_cat.keys() if c.startswith(POSITION_PREFIX)]
        pos_total = sum(by_cat[c]['total'] for c in position_cats)
        pos_correct = sum(by_cat[c]['correct'] for c in position_cats)
        if pos_total > 0:
            agent_acc_position_only = 100 * pos_correct / pos_total
            summary_position_only = {'correct': pos_correct, 'total': pos_total, 'acc_pct': round(agent_acc_position_only, 2)}
        else:
            summary_position_only = None
    else:
        summary_position_only = None

    summary = {
        'num_samples': n,
        'num_errors': sum('error' in r for r in results),
        'per_category': per_category,
        'only_category': only_category,
        'position_only': position_only,
        'prompt_style': prompt_style,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
        'baseline_acc': baseline_acc,
        'agent_acc': agent_acc if run_agent_eval else None,
        'agent_acc_position_only': agent_acc_position_only,
        'position_only_summary': summary_position_only,
        'results': results
    }

    out_name = f"eval{suffix}_{n}samples{ckpt_suffix}.json"
    with open(output_dir / out_name, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved to: {output_dir / out_name}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MMSI-Bench Evaluation")
    parser.add_argument('--num_samples', type=int, default=20, help="Num samples, 0 for all")
    parser.add_argument('--baseline_only', action='store_true', help="Run only baseline")
    parser.add_argument('--position_only', action='store_true', help="Only compute position relation")
    parser.add_argument('--prompt_style', type=str, default='eval', choices=['eval', 'training'])
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--agent_only', action='store_true')
    parser.add_argument('--per_category', type=int, default=0)
    parser.add_argument('--only_category', type=str, default=None)
    args = parser.parse_args()

    main(num_samples=args.num_samples, run_agent_eval=not args.baseline_only, position_only=args.position_only,
         prompt_style=args.prompt_style, checkpoint_path=args.checkpoint_path, agent_only=args.agent_only,
         per_category=args.per_category, only_category=args.only_category)
