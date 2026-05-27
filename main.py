import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from prompt_loader import build_core_control_trials, build_list_trials, load_prompts
from vllm_runner import DEFAULT_MODEL_ID, generate_from_messages, load_model


DEFAULT_SYSTEM_PROMPT = (
    "Gegeben die untenstehende Aussagenliste, geben Sie die Anzahl der Aussagen zurück, die Sie  als wahr erachten."
    "Geben Sie nur eine ganze Zahl zurück, ohne weiteren Text."
)
DEFAULT_MAX_TOKENS = 80
DEFAULT_TEMPERATURE = 0.2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM over prompts from a file.")
    parser.add_argument(
        "--experiment-mode",
        choices=["plain", "list"],
        default="plain",
        help="plain: existing prompt list mode, list: 4 core + 1 sensitive list experiment",
    )
    parser.add_argument("--prompts-file", default="prompts.txt")
    parser.add_argument("--core-statements-file", default="core_statements.txt")
    parser.add_argument("--core-set-id", default="core_set_default")
    parser.add_argument(
        "--list-trial-kind",
        choices=["core_control", "sensitive_treatment", "both"],
        default="sensitive_treatment",
        help="Which list-experiment trial kinds to run.",
    )
    parser.add_argument("--replicates-per-sensitive", type=int, default=5)
    parser.add_argument(
        "--include-core-control",
        action="store_true",
        help="Deprecated alias for --list-trial-kind both.",
    )
    parser.add_argument("--core-control-replicates", type=int, default=4)
    parser.add_argument("--shuffle-sensitive", action="store_true")
    parser.add_argument("--shuffle-trials", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt-mode", choices=["lines", "file"], default="lines")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-file",
        default=None,
        help="JSONL output path. Defaults to logs/list_experiment/<run_id>.generations.jsonl",
    )
    parser.add_argument(
        "--errors-file",
        default=None,
        help="JSONL errors path. Defaults to logs/list_experiment/<run_id>.errors.jsonl",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--print-each-response", action="store_true")

    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created_at_utc = utc_now_iso()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output_file or f"logs/list_experiment/{run_id}.generations.jsonl"
    errors_file = args.errors_file or f"logs/list_experiment/{run_id}.errors.jsonl"
    generation_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    records_to_run: list[dict[str, object]] = []
    if args.experiment_mode == "list":
        list_trial_kind = args.list_trial_kind
        if args.include_core_control and list_trial_kind == "sensitive_treatment":
            list_trial_kind = "both"

        if list_trial_kind in ("core_control", "both"):
            control_trials = build_core_control_trials(
                core_statements_file=args.core_statements_file,
                core_control_replicates=args.core_control_replicates,
                seed=args.seed,
            )
            records_to_run.extend(
                {
                    "trial_kind": "core_control",
                    "item_count": len(trial.presented_items),
                    "prompt": trial.rendered_prompt,
                    "messages": build_messages(trial.rendered_prompt, args.system_prompt),
                    "sensitive_id": None,
                    "sensitive_text": None,
                    "replicate_index": trial.replicate_index,
                    "order_block_index": trial.order_block_index,
                    "sensitive_position": None,
                    "order_labels": [item.item_id for item in trial.presented_items],
                }
                for trial in control_trials
            )

        if list_trial_kind in ("sensitive_treatment", "both"):
            trials = build_list_trials(
                core_statements_file=args.core_statements_file,
                sensitive_file=args.prompts_file,
                replicates_per_sensitive=args.replicates_per_sensitive,
                seed=args.seed,
                shuffle_sensitive=args.shuffle_sensitive,
            )
            records_to_run.extend(
                {
                    "trial_kind": "sensitive_treatment",
                    "item_count": len(trial.presented_items),
                    "prompt": trial.rendered_prompt,
                    "messages": build_messages(trial.rendered_prompt, args.system_prompt),
                    "sensitive_id": trial.sensitive_id,
                    "sensitive_text": trial.sensitive_text,
                    "replicate_index": trial.replicate_index,
                    "order_block_index": trial.order_block_index,
                    "sensitive_position": trial.sensitive_position,
                    "order_labels": [item.item_id for item in trial.presented_items],
                }
                for trial in trials
            )

        if args.shuffle_trials:
            rng = random.Random(args.seed)
            rng.shuffle(records_to_run)
    else:
        prompts = load_prompts(
            args.prompts_file,
            mode=args.prompt_mode,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        if not prompts:
            raise ValueError(f"No prompts loaded from {args.prompts_file}")
        records_to_run = [
            {"prompt": prompt, "messages": build_messages(prompt, args.system_prompt)}
            for prompt in prompts
        ]

    if not records_to_run:
        raise ValueError("No prompts/trials were generated.")

    llm = load_model(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = Path(errors_file)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        with errors_path.open("w", encoding="utf-8") as err_out:
            for index, payload in enumerate(records_to_run, start=1):
                prompt = str(payload["prompt"])
                messages = payload["messages"]
                base_record = {
                    "run_id": run_id,
                    "created_at_utc": created_at_utc,
                    "trial_id": index,
                    "trial_index_in_run": index,
                    "mode": args.experiment_mode,
                    "model_id": args.model_id,
                    "seed": args.seed,
                    "generation_params": generation_params,
                    "messages": messages,
                }

                if args.experiment_mode == "list":
                    base_record.update(
                        {
                            "core_set_id": args.core_set_id,
                            "trial_kind": payload["trial_kind"],
                            "item_count": payload["item_count"],
                            "sensitive_id": payload["sensitive_id"],
                            "sensitive_text": payload["sensitive_text"],
                            "replicate_index": payload["replicate_index"],
                            "order_block_index": payload["order_block_index"],
                            "sensitive_position": payload["sensitive_position"],
                            "order_labels": payload["order_labels"],
                        }
                    )

                try:
                    answer = generate_from_messages(
                        llm,
                        messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except Exception as exc:
                    error_count += 1
                    error_record = dict(base_record)
                    error_record["error_type"] = type(exc).__name__
                    error_record["error"] = str(exc)
                    err_out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    continue

                success_count += 1
                record = dict(base_record)
                record["answer"] = answer
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.print_each_response:
                    print(f"Trial {index}/{len(records_to_run)}")
                    print(answer)

    print(f"Run id: {run_id}")
    print(f"Completed trials: {success_count}")
    print(f"Failed trials: {error_count}")
    print(f"Generations file: {output_path}")
    print(f"Errors file: {errors_path}")


if __name__ == "__main__":
    main()
