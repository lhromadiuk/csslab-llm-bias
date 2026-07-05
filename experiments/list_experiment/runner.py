from __future__ import annotations

from argparse import Namespace
import ast
import csv
import json
from pathlib import Path
import random

from experiments.common import build_messages, default_run_id, load_prompts, utc_now_iso
from experiments.list_experiment.config import (
    DEFAULT_CORE_STATEMENTS_FILE,
    DEFAULT_LOG_DIR,
    DEFAULT_PERSONA_SYSTEM_PROMPT,
    DEFAULT_PERSONAS_FILE,
    DEFAULT_SENSITIVE_STATEMENTS_FILE,
    DEFAULT_SYSTEM_PROMPT,
)
from experiments.list_experiment.trials import (
    PromptItem,
    build_core_control_trials,
    build_list_trials,
    load_core_items,
    render_statements,
)
from vllm_runner import generate_batch_from_messages, load_model


def build_trial_records(args: Namespace) -> list[dict[str, object]]:
    if args.assignment_mode in ("persona_random", "persona_topn"):
        return build_persona_random_records(args)

    records_to_run: list[dict[str, object]] = []
    if args.list_trial_kind in ("core_control", "both"):
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
                "persona_id": None,
                "persona_profile_column": None,
            }
            for trial in control_trials
        )

    if args.list_trial_kind in ("sensitive_treatment", "both"):
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
                "persona_id": None,
                "persona_profile_column": None,
            }
            for trial in trials
        )

    if args.shuffle_trials:
        rng = random.Random(args.seed)
        rng.shuffle(records_to_run)

    return records_to_run


def build_persona_random_records(args: Namespace) -> list[dict[str, object]]:
    profile_columns = [args.persona_profile_column]
    if args.assignment_mode == "persona_topn":
        profile_columns = parse_profile_columns(args.persona_profile_columns)

    personas = load_personas(
        personas_file=args.personas_file,
        persona_profile_columns=profile_columns,
        persona_sample_size=args.persona_sample_size
        if args.assignment_mode == "persona_topn"
        else None,
        seed=args.seed,
    )
    sensitive_statements = load_prompts(args.prompts_file)
    core_items = load_core_items(args.core_statements_file)
    rng = random.Random(args.seed)

    records = []
    for persona_index, persona in enumerate(personas, start=1):
        persona_id = persona["persona_id"]
        for profile in persona["profiles"]:
            persona_profile = profile["persona_profile"]
            persona_profile_column = profile["persona_profile_column"]

            trial_items = list(core_items)
            rng.shuffle(trial_items)
            prompt = build_persona_user_prompt(persona_profile, trial_items)
            records.append(
                {
                    "trial_kind": "core_control",
                    "item_count": len(trial_items),
                    "prompt": prompt,
                    "messages": build_messages(prompt, args.persona_system_prompt),
                    "sensitive_id": None,
                    "sensitive_text": None,
                    "replicate_index": persona_index,
                    "order_block_index": None,
                    "sensitive_position": None,
                    "order_labels": [item.item_id for item in trial_items],
                    "persona_id": persona_id,
                    "persona_profile_column": persona_profile_column,
                }
            )

            for sensitive_index, sensitive_text in enumerate(
                sensitive_statements, start=1
            ):
                sensitive_id = f"sens_{sensitive_index}"
                trial_items = build_random_treatment_items(
                    core_items=core_items,
                    sensitive_id=sensitive_id,
                    sensitive_text=sensitive_text,
                    rng=rng,
                )
                prompt = build_persona_user_prompt(persona_profile, trial_items)
                sensitive_position = next(
                    index
                    for index, item in enumerate(trial_items, start=1)
                    if item.kind == "sensitive"
                )
                records.append(
                    {
                        "trial_kind": "sensitive_treatment",
                        "item_count": len(trial_items),
                        "prompt": prompt,
                        "messages": build_messages(prompt, args.persona_system_prompt),
                        "sensitive_id": sensitive_id,
                        "sensitive_text": sensitive_text,
                        "replicate_index": persona_index,
                        "order_block_index": None,
                        "sensitive_position": sensitive_position,
                        "order_labels": [item.item_id for item in trial_items],
                        "persona_id": persona_id,
                        "persona_profile_column": persona_profile_column,
                    }
                )

    if args.shuffle_trials:
        rng.shuffle(records)

    return records


def build_persona_user_prompt(
    persona_profile: str,
    trial_items: list[PromptItem],
) -> str:
    return f"{persona_profile}\n\nAussagen:\n\n{render_statements(trial_items)}"


def load_personas(
    personas_file: str,
    persona_profile_columns: list[str],
    persona_sample_size: int | None = None,
    seed: int | None = None,
) -> list[dict[str, str]]:
    path = Path(personas_file)
    if path.suffix == ".csv":
        try:
            file = path.open(encoding="utf-8", newline="")
            rows = list(csv.DictReader(file))
            file.close()
        except UnicodeDecodeError:
            with path.open(encoding="latin-1", newline="") as file:
                rows = list(csv.DictReader(file))

        if persona_sample_size is not None:
            rng = random.Random(seed)
            rows = rng.sample(rows, persona_sample_size)

        if rows and all(column in rows[0] for column in persona_profile_columns):
            return [
                {
                    "persona_id": row.get("respid") or f"persona_{index}",
                    "profiles": [
                        {
                            "persona_profile": format_profile(row[column]),
                            "persona_profile_column": column,
                        }
                        for column in persona_profile_columns
                    ],
                }
                for index, row in enumerate(rows, start=1)
                if all(row.get(column) for column in persona_profile_columns)
            ]

        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        return [
            {
                "persona_id": row.get("persona_id") or f"persona_{index}",
                "profiles": [
                    {
                        "persona_profile": row["persona_text"],
                        "persona_profile_column": "persona_text",
                    }
                ],
            }
            for index, row in enumerate(rows, start=1)
            if row.get("persona_text")
        ]

    persona_texts = load_prompts(path)
    return [
        {
            "persona_id": f"persona_{index}",
            "profiles": [
                {
                    "persona_profile": persona_text,
                    "persona_profile_column": "text",
                }
            ],
        }
        for index, persona_text in enumerate(persona_texts, start=1)
    ]


def parse_profile_columns(columns_text: str) -> list[str]:
    allowed_columns = {"core", "top-2", "top-4", "top-8", "top-16"}
    columns = [column.strip() for column in columns_text.split(",") if column.strip()]
    for column in columns:
        if column not in allowed_columns:
            raise ValueError(f"Unknown persona profile column: {column}")
    return columns


def format_profile(value: str) -> str:
    try:
        data = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value

    return "\n\n".join(f"{key}: {answer}" for key, answer in data.items())


def build_random_treatment_items(
    core_items: list[PromptItem],
    sensitive_id: str,
    sensitive_text: str,
    rng: random.Random,
) -> list[PromptItem]:
    core_order = list(core_items)
    rng.shuffle(core_order)
    sensitive_position = rng.randint(1, 5)

    trial_items = []
    core_index = 0
    for position in range(1, 6):
        if position == sensitive_position:
            trial_items.append(
                PromptItem(
                    item_id=sensitive_id,
                    text=sensitive_text,
                    kind="sensitive",
                )
            )
        else:
            trial_items.append(core_order[core_index])
            core_index += 1

    return trial_items


def run(args: Namespace) -> None:
    run_id = args.run_id or default_run_id()
    output_file = args.output_file or DEFAULT_LOG_DIR / f"{run_id}.generations.jsonl"
    generation_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "disable_thinking": args.disable_thinking,
    }

    records_to_run = build_trial_records(args)
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

    batch_size = max(1, args.batch_size)
    with output_path.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(records_to_run), batch_size):
            batch = records_to_run[batch_start : batch_start + batch_size]
            batch_messages = [payload["messages"] for payload in batch]
            answers = generate_batch_from_messages(
                llm,
                batch_messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                disable_thinking=args.disable_thinking,
            )

            for offset, (payload, answer) in enumerate(zip(batch, answers), start=1):
                index = batch_start + offset
                messages = payload["messages"]
                record = {
                    "run_id": run_id,
                    "created_at_utc": created_at_utc,
                    "trial_id": index,
                    "trial_index_in_run": index,
                    "mode": "list",
                    "model_id": args.model_id,
                    "seed": args.seed,
                    "generation_params": generation_params,
                    "messages": None if payload["persona_id"] is not None else messages,
                    "core_set_id": args.core_set_id,
                    "trial_kind": payload["trial_kind"],
                    "item_count": payload["item_count"],
                    "sensitive_id": payload["sensitive_id"],
                    "sensitive_text": payload["sensitive_text"],
                    "replicate_index": payload["replicate_index"],
                    "order_block_index": payload["order_block_index"],
                    "sensitive_position": payload["sensitive_position"],
                    "order_labels": payload["order_labels"],
                    "persona_id": payload["persona_id"],
                    "persona_profile_column": payload["persona_profile_column"],
                    "answer": answer,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.print_each_response:
                    print(f"Trial {index}/{len(records_to_run)}")
                    print(answer)

            out.flush()

    print(f"Run id: {run_id}")
    print(f"Completed trials: {len(records_to_run)}")
    print(f"Generations file: {output_path}")


def add_arguments(parser) -> None:
    parser.add_argument(
        "--prompts-file",
        default=str(DEFAULT_SENSITIVE_STATEMENTS_FILE),
        help="Sensitive statements file, one statement per line.",
    )
    parser.add_argument(
        "--core-statements-file",
        default=str(DEFAULT_CORE_STATEMENTS_FILE),
    )
    parser.add_argument("--core-set-id", default="core_set_default")
    parser.add_argument(
        "--list-trial-kind",
        choices=["core_control", "sensitive_treatment", "both"],
        default="sensitive_treatment",
        help="Which list-experiment trial kinds to run.",
    )
    parser.add_argument("--replicates-per-sensitive", type=int, default=5)
    parser.add_argument("--core-control-replicates", type=int, default=4)
    parser.add_argument(
        "--assignment-mode",
        choices=["balanced", "persona_random", "persona_topn"],
        default="balanced",
        help="balanced: old block design, persona_random: one profile column for all personas, persona_topn: random N personas across multiple profile columns.",
    )
    parser.add_argument(
        "--personas-file",
        default=str(DEFAULT_PERSONAS_FILE),
        help="Persona file used by --assignment-mode persona_random.",
    )
    parser.add_argument(
        "--persona-profile-column",
        choices=["core", "top-2", "top-4", "top-8", "top-16"],
        default="core",
        help="CSV column to use as the respondent profile in persona_random mode.",
    )
    parser.add_argument(
        "--persona-profile-columns",
        default="core,top-2,top-4,top-8,top-16",
        help="Comma-separated CSV columns for persona_topn mode.",
    )
    parser.add_argument(
        "--persona-sample-size",
        type=int,
        default=None,
        help="Random number of personas to use in persona_topn mode.",
    )
    parser.add_argument(
        "--persona-system-prompt",
        default=DEFAULT_PERSONA_SYSTEM_PROMPT,
    )
    parser.add_argument("--shuffle-sensitive", action="store_true")
    parser.add_argument("--shuffle-trials", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
