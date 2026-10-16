"""
MindCube Evaluation Script (Full Agent Pipeline)

- Evaluation logic: consistent with MMSI / ViewSpatial
  - Baseline: direct VLM inference
  - Agent: VLM decides which objects to measure -> AgentToolWrapper calls Pi3+SAM3 ->
           format_tool_result_for_model generates "3D Analysis" -> answer using Stage2 prompt
- Data source: MindCube/data/raw/MindCube_tinybench.jsonl
  - Each line contains: images, question, gt_answer, type, category, etc.
"""

import os
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# Logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Paths for module imports
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR  #
EVAL_DIR = THIS_DIR / "ours" / "evaluation"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


class Qwen3VLModel:
    """Qwen3-VL Model Wrapper"""

    def __init__(self, model_path: str = "models/Qwen3-VL-4B-Instruct", checkpoint_path: str = None):
        logger.info(f"Loading Qwen3-VL model: {model_path}")
        if checkpoint_path:
            logger.info(f"  Will load GRPO checkpoint: {checkpoint_path}")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # If checkpoint provided, load trained weights
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

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Generate response"""
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content: List[Dict[str, Any]] = []
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
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        output_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response


def parse_answer(response: str) -> str:
    """Parse A/B/C/D"""
    if not response:
        return "UNKNOWN"

    patterns = [
        r"<answer>\s*([A-D])",
        r"[Aa]nswer[:\s]+\**([A-D])",
        r"[Tt]he answer is[:\s]+\**([A-D])",
        r"\*\*([A-D])\*\*",
        r"\b([A-D])\.",
    ]

    for pattern in patterns:
        m = re.search(pattern, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    for m in re.finditer(r"\b([A-D])\b", response):
        return m.group(1).upper()

    return "UNKNOWN"


def run_baseline(vlm: Qwen3VLModel, images: List[Image.Image], question: str) -> Dict[str, Any]:
    """Baseline: direct inference"""
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
        "response": response,
        "answer": parse_answer(response),
        "time": elapsed,
    }


def run_agent(vlm: Qwen3VLModel, tool_wrapper, images: List[Image.Image], question: str, prompt_style: str = "eval") -> Dict[str, Any]:
    """Agent: 2 stages + 3D tool"""
    from agent_tool_wrapper import format_tool_result_for_model
    if prompt_style == "training":
        from stage2_prompt_constants import STAGE2_SYSTEM_TRAINING, build_stage2_prompt_training
    else:
        from stage2_prompt_constants import STAGE2_SYSTEM, build_stage2_prompt

    # Stage 1: Ask model what objects to measure
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
    objects: List[str] = []
    pattern = r"<objects_to_segment>(.*?)</objects_to_segment>"
    match = re.search(pattern, stage1_response, re.DOTALL)
    if match:
        for line in match.group(1).strip().split("\n"):
            line = re.sub(r"^[-•*]\s*", "", line.strip())
            if line and len(line) > 1:
                objects.append(line.lower())

    if not objects:
        # Fallback keyword extraction for common objects
        common = [
            "sneaker",
            "bottle",
            "table",
            "sofa",
            "tv",
            "curtain",
            "bed",
            "chair",
            "door",
            "window",
            "wall",
            "floor",
            "lamp",
            "plant",
            "object",
        ]
        q_lower = question.lower()
        objects = [w for w in common if w in q_lower][:3]
        if not objects:
            objects = ["object"]

    objects = objects[:5]

    tool_result = {"status": "error"}

    # Stage 2: Call 3D tool
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    image_paths: List[str] = []
    try:
        for i, img in enumerate(images):
            path = f"{temp_dir}/img_{i}.jpg"
            img.save(path)
            image_paths.append(path)

        tool_request = {
            "pipeline": "pi3+sam3",
            "semantic_targets": objects,
            "confidence_threshold": 0.3,
        }

        try:
            tool_result = tool_wrapper.execute_tool_request(image_paths, tool_request)
            formatted_3d = format_tool_result_for_model(
                tool_result,
                max_per_category=2,
                question=question,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Tool call failed: {e}")
            formatted_3d = "3D tool unavailable."
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Stage 3: Answer with 3D info
    if prompt_style == "training":
        stage2_system = STAGE2_SYSTEM_TRAINING
        stage2_prompt = build_stage2_prompt_training(question, formatted_3d)
    else:
        stage2_system = STAGE2_SYSTEM
        stage2_prompt = build_stage2_prompt(question, formatted_3d)

    response = vlm.generate(images, stage2_prompt, stage2_system, max_tokens=2048)
    elapsed = time.time() - start

    return {
        "response": response,
        "answer": parse_answer(response),
        "objects_requested": objects,
        "formatted_3d": formatted_3d,
        "tool_status": tool_result.get("status", "error"),
        "time": elapsed,
    }


def load_mindcube_samples(data_path: Path, num_samples: int) -> List[Dict[str, Any]]:
    """Load samples from jsonl; num_samples<=0 means all."""
    samples: List[Dict[str, Any]] = []
    with data_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if num_samples > 0 and num_samples < len(samples):
        return samples[:num_samples]
    return samples


def main(
    num_samples: int = 173,
    run_agent_eval: bool = True,
    prompt_style: str = "eval",
    checkpoint_path: str = None,
    agent_only: bool = False,
) -> None:
    logger.info("=" * 60)
    logger.info("MindCube Evaluation (Baseline vs Agent)")
    if prompt_style == "training":
        logger.info("Stage2 template: training (Think step by step)")
    else:
        logger.info("Stage2 template: eval (default)")
    logger.info("=" * 60)

    data_path = Path(os.environ.get("MINDCUBE_JSONL", "MindCube/data/raw/MindCube_tinybench.jsonl"))
    data_dir = Path(os.environ.get("MINDCUBE_IMAGE_ROOT", "MindCube/data"))

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    samples = load_mindcube_samples(data_path, num_samples)
    logger.info(f"Eval samples count: {len(samples)} (num_samples={num_samples})")

    # Initialize model and tools
    vlm = Qwen3VLModel(model_path=os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"), checkpoint_path=checkpoint_path)

    tool_wrapper = None
    if run_agent_eval:
        from agent_tool_wrapper import AgentToolWrapper

        tool_wrapper = AgentToolWrapper(
            sam3_checkpoint=os.environ.get("SAM3_CHECKPOINT"),
            pi3_checkpoint=os.environ.get("PI3_CHECKPOINT"),
            use_mock=False,
            device="cuda",
        )

    results: List[Dict[str, Any]] = []
    baseline_correct = 0
    agent_correct = 0

    for i, sample in enumerate(samples):
        category = sample.get("category", [])
        sample_type = sample.get("type", "unknown")
        logger.info(f"\nSample {i + 1}/{len(samples)}: type={sample_type}, category={category}")

        try:
            # Load images
            images: List[Image.Image] = []
            for img_rel in sample["images"]:
                full_path = data_dir / img_rel
                if full_path.exists():
                    images.append(Image.open(full_path).convert("RGB"))
                else:
                    raise FileNotFoundError(f"Image not found: {full_path}")

            if not images:
                logger.warning("Could not load any images, skipping sample")
                results.append({"index": i, "error": "no images"})
                continue

            question = sample["question"]
            gt = sample["gt_answer"]

            # Baseline (skip if agent_only)
            baseline_result = None
            if not agent_only:
                baseline_result = run_baseline(vlm, images, question)
                b_correct = baseline_result["answer"] == gt
                if b_correct:
                    baseline_correct += 1
                logger.info(f"Baseline: {baseline_result['answer']} ({'✓' if b_correct else '✗'}) GT={gt}")
            else:
                logger.info("Baseline disabled (agent_only=True), evaluating Agent only.")

            result: Dict[str, Any] = {
                "index": i,
                "type": sample_type,
                "category": category,
                "gt": gt,
                "baseline": baseline_result,
            }

            # Agent
            if run_agent_eval and tool_wrapper is not None:
                agent_result = run_agent(vlm, tool_wrapper, images, question, prompt_style=prompt_style)
                a_correct = agent_result["answer"] == gt
                if a_correct:
                    agent_correct += 1
                logger.info(f"Agent: {agent_result['answer']} ({'✓' if a_correct else '✗'})")
                result["agent"] = agent_result

            results.append(result)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Sample processing failed: {e}")
            results.append({"index": i, "error": str(e)})

    # Summary
    n = len(samples)  # Include failed samples in the denominator.
    if n == 0:
        print("No successfully processed samples")
        return

    baseline_acc = None
    if not agent_only:
        baseline_acc = baseline_correct / n * 100

    print("\n" + "=" * 60)
    print("MindCube Eval Results")
    print("=" * 60)
    print(f"Sample count: {n}")
    if not agent_only:
        print(f"Baseline: {baseline_correct}/{n} = {baseline_acc:.1f}%")
    else:
        print("Baseline: Skipped (agent_only=True)")

    agent_acc = None
    if run_agent_eval:
        agent_acc = agent_correct / n * 100
        print(f"Agent: {agent_correct}/{n} = {agent_acc:.1f}%")
        if baseline_acc is not None:
            print(f"Diff: {agent_acc - baseline_acc:+.1f}%")

    # Save results
    output_dir = Path("outputs/mindcube_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_n = n
    prompt_suffix = "" if prompt_style == "eval" else "_training_prompt"
    ckpt_suffix = ""
    if checkpoint_path:
        # Extract checkpoint id (e.g. step_400)
        ckpt_name = Path(checkpoint_path).name
        ckpt_suffix = f"_ckpt_{ckpt_name}"

    fname_suffix = f"{effective_n}samples{prompt_suffix}{ckpt_suffix}"

    summary = {
        "num_samples": effective_n,
        "num_errors": sum("error" in r for r in results),
        "prompt_style": prompt_style,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "baseline_acc": baseline_acc,
        "agent_acc": agent_acc,
        "results": results,
    }

    with (output_dir / f"eval_{fname_suffix}.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved: {output_dir}/eval_{fname_suffix}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=173,
        help="Number of samples to evaluate; num_samples<=0 means all.",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Run Baseline only, no Agent.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="eval",
        choices=["eval", "training"],
        help="eval=Eval Stage2 template; training=GRPO training template (Think step by step)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="GRPO checkpoint directory path (e.g. checkpoints/step_400), will load mp_rank_00_model_states.pt from it",
    )
    parser.add_argument(
        "--agent_only",
        action="store_true",
        help="Run Agent only (skip Baseline), useful for checkpoint + 3D info comparison.",
    )
    args = parser.parse_args()

    main(
        num_samples=args.num_samples,
        run_agent_eval=not args.baseline_only,
        prompt_style=args.prompt_style,
        checkpoint_path=args.checkpoint_path,
        agent_only=args.agent_only,
    )
