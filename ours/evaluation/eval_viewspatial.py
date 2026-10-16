"""
ViewSpatial-Bench Evaluation Script
Baseline and two-stage agent evaluation.
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add path
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from pathlib import Path


class Qwen3VLModel:
    """Qwen3-VL model wrapper (consistent with MMSI-Bench)"""

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-4B-Instruct", checkpoint_path: str = None):
        logger.info(f"Loading Qwen3-VL model: {model_path}")
        if checkpoint_path:
            logger.info(f"  Will load GRPO checkpoint: {checkpoint_path}")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load weights if checkpoint is provided
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path) / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

            logger.info(f"Loading checkpoint weights: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

            # DeepSpeed checkpoint format
            if "module" in checkpoint:
                state_dict = checkpoint["module"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                # Top level state dict
                state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor) or (isinstance(v, dict) and "weight" in str(v))}

            # Remove DeepSpeed prefix
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
        logger.info("Model loaded successfully")

    def generate(self, images: List[Image.Image], prompt: str, system_prompt: str = None, max_tokens: int = 2048) -> str:
        """Generate response (consistent with MMSI-Bench)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )

        output_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response


def parse_answer(response: str) -> str:
    """Parse answer (consistent with MMSI-Bench)"""
    patterns = [
        r'<answer>\s*([A-D])',
        r'[Aa]nswer[:\s]+\**([A-D])',
        r'[Tt]he answer is[:\s]+\**([A-D])',
        r'\*\*([A-D])\*\*',
        r'\b([A-D])\.',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    for match in re.finditer(r'\b([A-D])\b', response):
        return match.group(1).upper()

    return "UNKNOWN"


def run_baseline(vlm, images: List[Image.Image], question: str) -> Dict:
    """Run Baseline (consistent with MMSI-Bench)"""
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


def run_agent(vlm, tool_wrapper, images: List[Image.Image], question: str, prompt_style: str = "eval") -> Dict:
    """Run Agent. prompt_style: eval=eval template, training=GRPO training template (Think step by step)."""
    from agent_tool_wrapper import format_tool_result_for_model
    if prompt_style == "training":
        from stage2_prompt_constants import STAGE2_SYSTEM_TRAINING, build_stage2_prompt_training
    else:
        from stage2_prompt_constants import STAGE2_SYSTEM, build_stage2_prompt

    # Stage 1: Model decides what objects are needed
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
        # Try to extract keywords from question
        common = ["counter", "refrigerator", "table", "chair", "door", "window", "wall", "floor",
                  "bed", "sofa", "cabinet", "sink", "toilet", "person", "object"]
        q_lower = question.lower()
        objects = [w for w in common if w in q_lower][:3]
        if not objects:
            objects = ["object"]

    objects = objects[:5]

    tool_result = {"status": "error"}

    # Stage 2: Call tools
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
        formatted_3d = format_tool_result_for_model(tool_result, max_per_category=2, question=question)
    except Exception as e:
        logger.warning(f"Tool invocation failed: {e}")
        formatted_3d = "3D tool unavailable."

    # Clear temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Stage 3: Answer with 3D info (eval or training templates)
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


def load_image(image_paths: List[str], base_dir: Path) -> List[Image.Image]:
    """Load ViewSpatial images"""
    images = []

    for img_path in image_paths:
        # Remove possible ViewSpatial-Bench/ prefix
        clean_path = img_path
        if img_path.startswith("ViewSpatial-Bench/"):
            clean_path = img_path[len("ViewSpatial-Bench/"):]

        # Try different path combinations
        possible_paths = [
            base_dir / clean_path,
            base_dir / img_path,
            base_dir / Path(img_path).name,
            Path(img_path)
        ]

        loaded = False
        for p in possible_paths:
            if p.exists():
                img = Image.open(p).convert("RGB")
                images.append(img)
                loaded = True
                break

        if not loaded:
            raise FileNotFoundError(f"Cannot load image: {img_path}")

    return images


def main(num_samples: int = 20, per_type: int = 0, run_agent_eval: bool = True, prompt_style: str = "eval", checkpoint_path: str = None, agent_only: bool = False):
    logger.info("="*60)
    logger.info("ViewSpatial-Bench Evaluation")
    if prompt_style == "training":
        logger.info("Stage2 template: training (consistent with GRPO, Think step by step)")
    else:
        logger.info("Stage2 template: eval (default)")
    logger.info("="*60)

    # Load local ViewSpatial-Bench annotation JSON
    eval_json = Path(os.environ.get("VIEWSPATIAL_JSON", "ViewSpatial-Bench/eval/ViewSpatial-Bench.json"))
    with open(eval_json, "r") as f:
        raw_data = json.load(f)
    logger.info(f"Original annotation count: {len(raw_data)}")

    # Sample by category
    if per_type > 0:
        from collections import defaultdict
        by_type = defaultdict(list)
        for item in raw_data:
            t = item.get("question_type", "unknown")
            by_type[t].append(item)
        data = []
        selected = {}
        rng = np.random.default_rng(42)
        for t, items in by_type.items():
            k = min(per_type, len(items))
            if k <= 0:
                continue
            indices = rng.choice(len(items), size=k, replace=False)
            chosen = [items[i] for i in indices]
            data.extend(chosen)
            selected[t] = k
        logger.info(f"Sampled at most {per_type} per question_type, actual: {selected}")
    else:
        # Sample (num_samples <= 0 means all)
        if num_samples > 0 and num_samples < len(raw_data):
            indices = np.random.default_rng(42).choice(len(raw_data), num_samples, replace=False)
            data = [raw_data[i] for i in indices]
        else:
            data = raw_data

    logger.info(f"Evaluation sample count: {len(data)}")

    # Image base directory
    base_dir = Path(os.environ.get("VIEWSPATIAL_IMAGE_ROOT", "ViewSpatial-Bench"))

    # Init model
    vlm = Qwen3VLModel(model_path=os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"), checkpoint_path=checkpoint_path)

    tool_wrapper = None
    if run_agent_eval:
        from agent_tool_wrapper import AgentToolWrapper
        tool_wrapper = AgentToolWrapper(
            sam3_checkpoint=os.environ.get("SAM3_CHECKPOINT"),
            pi3_checkpoint=os.environ.get("PI3_CHECKPOINT"),
            use_mock=False
        )

    results = []
    baseline_correct = 0
    agent_correct = 0

    for i, item in enumerate(data):
        q_type = item['question_type']
        logger.info(f"\nSample {i+1}/{len(data)}: {q_type}")

        try:
            # Load image
            image_paths = item['image_path']
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            images = load_image(image_paths, base_dir)

            if not images:
                logger.warning(f"Could not load any images, skipping")
                results.append({'index': i, 'error': 'no images'})
                continue

            # Question and options
            question = item['question']
            choices = item['choices']
            # Append options (consistent with MMSI-Bench)
            full_question = f"{question}\nOptions:\n{choices}"

            # Extract GT answer
            if not item['answer']:
                raise ValueError('Missing ground-truth answer')
            gt = item['answer'][0]

            # Baseline (optional, skip if agent_only)
            baseline_result = None
            if not agent_only:
                baseline_result = run_baseline(vlm, images, full_question)
                b_correct = baseline_result['answer'] == gt
                if b_correct:
                    baseline_correct += 1
                logger.info(f"Baseline: {baseline_result['answer']} ({'✓' if b_correct else '✗'}) GT={gt}")

            result = {
                'index': i,
                'question_type': q_type,
                'gt': gt,
                'baseline': baseline_result
            }

            # Agent
            if run_agent_eval:
                agent_result = run_agent(vlm, tool_wrapper, images, full_question, prompt_style=prompt_style)
                a_correct = agent_result['answer'] == gt
                if a_correct:
                    agent_correct += 1
                logger.info(f"Agent: {agent_result['answer']} ({'✓' if a_correct else '✗'})")
                result['agent'] = agent_result

            results.append(result)

        except Exception as e:
            logger.error(f"Sample processing failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({'index': i, 'error': str(e)})

    # Summary
    n = len(data)  # Include failed samples in the denominator.
    if n == 0:
        print("No samples processed successfully")
        return

    baseline_acc = None
    if not agent_only:
        baseline_acc = baseline_correct / n * 100

    print("\n" + "="*60)
    print("ViewSpatial-Bench Evaluation Results")
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
            print(f"Difference: {agent_acc - baseline_acc:+.1f}%")

    # Save
    output_dir = Path("outputs/viewspatial_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_suffix = "" if prompt_style == "eval" else "_training_prompt"
    ckpt_suffix = ""
    if checkpoint_path:
        ckpt_name = Path(checkpoint_path).name
        ckpt_suffix = f"_ckpt_{ckpt_name}"

    out_name = f"eval_{n}samples{prompt_suffix}{ckpt_suffix}.json"

    summary = {
        'num_samples': n,
        'num_errors': sum('error' in r for r in results),
        'prompt_style': prompt_style,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
        'baseline_acc': baseline_acc,
        'agent_acc': agent_acc if run_agent_eval else None,
        'results': results
    }
    with open(output_dir / out_name, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved: {output_dir} ({out_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=20,
                        help="Number of eval samples, applies if per_type=0; <=0 means all.")
    parser.add_argument('--per_type', type=int, default=0,
                        help="Number of samples per question_type.")
    parser.add_argument('--baseline_only', action='store_true')
    parser.add_argument('--prompt_style', type=str, default='eval', choices=['eval', 'training'],
                        help='eval or training template for comparison')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='GRPO checkpoint directory path')
    parser.add_argument('--agent_only', action='store_true',
                        help='Run Agent only (skip Baseline), useful for checkpoint eval')
    args = parser.parse_args()

    main(num_samples=args.num_samples,
         per_type=args.per_type,
         run_agent_eval=not args.baseline_only,
         prompt_style=args.prompt_style,
         checkpoint_path=args.checkpoint_path,
         agent_only=args.agent_only)
