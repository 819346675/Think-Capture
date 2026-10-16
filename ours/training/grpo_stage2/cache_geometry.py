"""Build positive/negative Stage 2 records from a MindCube-format training split."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image
from agent_tool_wrapper import AgentToolWrapper
from eval_mindcube import Qwen3VLModel, run_baseline, run_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', required=True, help='Training-split JSONL, not an evaluation split')
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_path', default=os.environ.get('VLM_MODEL', 'Qwen/Qwen3-VL-4B-Instruct'))
    parser.add_argument('--max_samples', type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {kind: output_dir / f'grpo_stage2_{kind}_quality.jsonl'
             for kind in ('positive', 'negative')}
    if any(path.exists() for path in paths.values()):
        parser.error('Output files already exist; use a new output directory')
    vlm = Qwen3VLModel(args.model_path)
    tool = AgentToolWrapper(device='cuda')
    counts = {'positive': 0, 'negative': 0, 'agreement': 0, 'skipped': 0}
    with open(args.data_path, encoding='utf-8') as source, \
            open(paths['positive'], 'x', encoding='utf-8') as positive, \
            open(paths['negative'], 'x', encoding='utf-8') as negative:
        streams = {'positive': positive, 'negative': negative}
        seen = 0
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            if args.max_samples > 0 and seen >= args.max_samples:
                break
            seen += 1
            sample = json.loads(line)
            image_paths = [str((Path(args.image_root) / p).resolve()) for p in sample['images']]
            if len(image_paths) < 2:
                counts['skipped'] += 1
                continue
            images = [Image.open(path).convert('RGB') for path in image_paths]
            gt = sample['gt_answer'].strip().upper()
            if gt not in ('A', 'B', 'C', 'D'):
                raise ValueError(f'Invalid answer at line {line_no}: {gt}')
            baseline = run_baseline(vlm, images, sample['question'])
            agent = run_agent(vlm, tool, images, sample['question'])
            if agent['tool_status'] != 'success':
                counts['skipped'] += 1
                print(f'Line {line_no}: geometry unavailable; skipped', flush=True)
                continue
            b_correct, a_correct = baseline['answer'] == gt, agent['answer'] == gt
            if b_correct == a_correct:
                counts['agreement'] += 1
                continue
            kind = 'positive' if a_correct else 'negative'
            record = dict(question=sample['question'], images=image_paths,
                          formatted_3d=agent['formatted_3d'], gt_answer=gt,
                          sample_type=kind, source_line=line_no,
                          baseline_answer=baseline['answer'], agent_answer=agent['answer'])
            streams[kind].write(json.dumps(record, ensure_ascii=False) + '\n')
            streams[kind].flush()
            counts[kind] += 1
            print(f'Line {line_no}: {kind}; {counts}', flush=True)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
