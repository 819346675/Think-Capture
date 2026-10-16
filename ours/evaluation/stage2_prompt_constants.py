"""
Single source of truth for Stage2 template: identical to Stage2 in eval_mindcube.run_agent.
Evaluation and GRPO training must use constants and functions in this file to ensure prompt alignment.

- eval version: for concise evaluation (Use the 3D spatial data to answer / Keep reasoning brief)
- training version: consistent with GRPO training (Use both images and 3D / Think step by step), facilitating comparison.
"""

# ---------- Eval Version (Default, do not delete) ----------
# Letter-for-letter identical with stage2_system in eval_mindcube.run_agent
STAGE2_SYSTEM = """You are a spatial reasoning assistant with 3D perception.
Use the 3D spatial data to answer. You MUST end with <answer>X</answer> where X is A, B, C, or D."""


def build_stage2_prompt(question: str, formatted_3d: str) -> str:
    """Format strictly aligned with stage2_prompt in eval_mindcube.run_agent (eval version)."""
    return f"""Question: {question}

3D Spatial Information:
{formatted_3d}

Answer the question using the 3D data. Keep reasoning brief.

Reasoning:"""


# ---------- Training Version (consistent with user_prompt in GRPO train.py, helps with comparison) ----------
STAGE2_SYSTEM_TRAINING = (
    "You are a spatial reasoning assistant with 3D perception.\n"
    "Use both the images and the 3D spatial data to answer. "
    "Think step by step and combine visual cues with 3D information. "
    "You MUST end with <answer>X</answer> where X is A, B, C, or D."
)


def build_stage2_prompt_training(question: str, formatted_3d: str) -> str:
    """Format consistent with user_prompt in GRPO train.py (training version)."""
    return (
        f"Question: {question}\n\n"
        f"3D Spatial Information:\n{formatted_3d}\n\n"
        "Answer the question by combining the images above with the "
        "3D spatial information. Think step by step, then give your final answer.\n\n"
        "Reasoning:"
    )


# Letter-for-letter identical with the "good performance" eval template for alignment check (do not modify expected value here, sync with constants above if altered)
_EXPECTED_SYSTEM = """You are a spatial reasoning assistant with 3D perception.
Use the 3D spatial data to answer. You MUST end with <answer>X</answer> where X is A, B, C, or D."""


def _expected_prompt(question: str, formatted_3d: str) -> str:
    return f"""Question: {question}

3D Spatial Information:
{formatted_3d}

Answer the question using the 3D data. Keep reasoning brief.

Reasoning:"""


def verify_alignment() -> bool:
    """Verify current constants are identical to expected eval template; can be called prior to training/eval."""
    ok = (STAGE2_SYSTEM.strip() == _EXPECTED_SYSTEM.strip())
    q, d3 = "Which is left?", "3D Analysis:\nA at (0,0,0)."
    ok = ok and (build_stage2_prompt(q, d3).strip() == _expected_prompt(q, d3).strip())
    return ok


if __name__ == "__main__":
    assert verify_alignment(), "Stage2 prompt constants do not match expected eval template."
    print("OK: Stage2 prompt aligned with eval (single source of truth).")
