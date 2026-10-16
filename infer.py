"""Run the two-stage Think-Capture pipeline on local images."""
import argparse
import json
import os
from pathlib import Path

from PIL import Image
from agent_tool_wrapper import AgentToolWrapper
from eval_mindcube import Qwen3VLModel, run_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--model_path', default=os.environ.get('VLM_MODEL', 'Qwen/Qwen3-VL-4B-Instruct'))
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--prompt_style', choices=['eval', 'training'], default='eval')
    parser.add_argument('--output', help='Optional JSON output path; must not already exist')
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error(f'Output already exists: {args.output}')
    images = [Image.open(path).convert('RGB') for path in args.images]
    vlm = Qwen3VLModel(args.model_path, args.checkpoint_path)
    tool = AgentToolWrapper(device='cuda')
    result = run_agent(vlm, tool, images, args.question, prompt_style=args.prompt_style)
    print(result['response'])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
