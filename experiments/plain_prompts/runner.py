from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from experiments.common import build_messages, default_run_id, load_prompts, utc_now_iso
from experiments.list_experiment.runner import load_personas
from vllm_runner import generate_batch_from_messages, load_model


BIAS_CALCULATION_SYSTEM_PROMPT = (
    "Du bist ein automatisches System, antworte ausschließlich mit einer einzigen ziffer."
)


def build_persona_direct_prompt(persona_profile: str, prompt: str) -> str:
    return f"{persona_profile}\n\nFrage:\n\n{prompt}"


def build_records(args: Namespace, prompts: list[str]) -> list[dict]:
    if not args.use_personas:
        return [
            {
                "trial_id": index,
                "trial_index_in_run": index,
                "mode": "plain",
                "prompt_index": index,
                "prompt": prompt,
                "messages": build_messages(
                    prompt,
                    args.system_prompt or BIAS_CALCULATION_SYSTEM_PROMPT,
                ),
            }
            for index, prompt in enumerate(prompts, start=1)
        ]

    personas = load_personas(
        args.personas_file,
        [args.persona_profile_column],
        seed=args.seed,
    )
    records = []
    trial_index = 0
    for persona in personas:
        for profile in persona["profiles"]:
            for prompt_index, prompt in enumerate(prompts, start=1):
                trial_index += 1
                user_prompt = build_persona_direct_prompt(
                    profile["persona_profile"],
                    prompt,
                )
                records.append(
                    {
                        "trial_id": trial_index,
                        "trial_index_in_run": trial_index,
                        "mode": "plain_persona",
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "persona_id": persona["persona_id"],
                        "persona_profile": profile["persona_profile"],
                        "persona_profile_column": profile["persona_profile_column"],
                        "messages": build_messages(
                            user_prompt,
                            args.persona_system_prompt
                            or args.system_prompt
                            or BIAS_CALCULATION_SYSTEM_PROMPT,
                        ),
                    }
                )
    return records


def run(args: Namespace) -> None:
    run_id = args.run_id or default_run_id()
    output_file = args.output_file or f"logs/plain_prompts/{run_id}.generations.jsonl"
    generation_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "disable_thinking": args.disable_thinking,
    }

    prompts = load_prompts(
        args.prompts_file,
        mode=args.prompt_mode,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    records_to_run = build_records(args, prompts)
    llm = load_model(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        quantization=args.quantization,
        load_format=args.load_format,
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at_utc = utc_now_iso()

    with output_path.open("w", encoding="utf-8") as out:
        batch_size = max(1, args.batch_size)
        for batch_start in range(0, len(records_to_run), batch_size):
            batch = records_to_run[batch_start : batch_start + batch_size]
            answers = generate_batch_from_messages(
                llm,
                [record["messages"] for record in batch],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                disable_thinking=args.disable_thinking,
            )

            for offset, (payload, answer) in enumerate(zip(batch, answers), start=1):
                index = batch_start + offset
                record = {
                    "run_id": run_id,
                    "created_at_utc": created_at_utc,
                    "model_id": args.model_id,
                    "seed": args.seed,
                    "generation_params": generation_params,
                    **payload,
                    "answer": answer,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.print_each_response:
                    print(f"Trial {index}/{len(records_to_run)}")
                    print(answer)

    print(f"Run id: {run_id}")
    print(f"Completed trials: {len(records_to_run)}")
    print(f"Generations file: {output_path}")


def add_arguments(parser) -> None:
    parser.add_argument("--prompts-file", default="prompts.txt")
    parser.add_argument("--prompt-mode", choices=["lines", "file"], default="lines")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--use-personas", action="store_true")
    parser.add_argument("--personas-file", default="analysis/ZA9089_JSON.csv")
    parser.add_argument("--persona-profile-column", default="top-2")
    parser.add_argument("--persona-system-prompt", default=None)
