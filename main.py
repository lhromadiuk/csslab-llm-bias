import argparse
import json
from pathlib import Path

from prompt_loader import load_prompts
from vllm_runner import DEFAULT_MODEL_ID, load_model, run_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM over prompts from a file.")
    parser.add_argument("--prompts-file", default="prompts.txt")
    parser.add_argument("--prompt-mode", choices=["lines", "file"], default="lines")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-file", default="logs/generations.jsonl")

    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(
        args.prompts_file,
        mode=args.prompt_mode,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not prompts:
        raise ValueError(f"No prompts loaded from {args.prompts_file}")

    llm = load_model(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for index, prompt in enumerate(prompts, start=1):
            print(f"\n=== PROMPT {index}/{len(prompts)} ===")
            print(prompt)
            answer = run_prompt(
                llm,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print("\n=== ANSWER ===")
            print(answer)

            record = {
                "index": index,
                "prompt": prompt,
                "answer": answer,
                "model_id": args.model_id,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote generations to {output_path}")


if __name__ == "__main__":
    main()
