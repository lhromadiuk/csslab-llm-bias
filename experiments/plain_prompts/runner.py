from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from experiments.common import build_messages, default_run_id, load_prompts, utc_now_iso
from vllm_runner import generate_from_messages, load_model


def run(args: Namespace) -> None:
    run_id = args.run_id or default_run_id()
    output_file = args.output_file or f"logs/plain_prompts/{run_id}.generations.jsonl"
    generation_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    prompts = load_prompts(
        args.prompts_file,
        mode=args.prompt_mode,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    llm = load_model(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        load_format=args.load_format,
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at_utc = utc_now_iso()

    with output_path.open("w", encoding="utf-8") as out:
        for index, prompt in enumerate(prompts, start=1):
            messages = build_messages(prompt, args.system_prompt)
            answer = generate_from_messages(
                llm,
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            record = {
                "run_id": run_id,
                "created_at_utc": created_at_utc,
                "trial_id": index,
                "trial_index_in_run": index,
                "mode": "plain",
                "model_id": args.model_id,
                "seed": args.seed,
                "generation_params": generation_params,
                "messages": messages,
                "answer": answer,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.print_each_response:
                print(f"Trial {index}/{len(prompts)}")
                print(answer)

    print(f"Run id: {run_id}")
    print(f"Completed trials: {len(prompts)}")
    print(f"Generations file: {output_path}")


def add_arguments(parser) -> None:
    parser.add_argument("--prompts-file", default="prompts.txt")
    parser.add_argument("--prompt-mode", choices=["lines", "file"], default="lines")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--system-prompt", default=None)
