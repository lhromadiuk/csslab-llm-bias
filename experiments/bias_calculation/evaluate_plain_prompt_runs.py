from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SENSITIVE_COLUMNS = [
    "ma01b",
    "ma02",
    "ma03",
    "ma04",
    "mp02",
    "ca13",
    "ca08",
    "fr10",
    "fr04b",
    "fr03b",
    "pi08",
    "mm03",
    "mm04",
    "mm05",
]

DEFAULT_RUNS = [
    (
        "mistral",
        "logs/plain_prompts/mistral_persona_direct.generations.jsonl",
        "logs/plain_prompts/mistral_persona_reversed.generations.jsonl",
    ),
    (
        "qwen",
        "logs/plain_prompts/qwen_persona_direct_no_think.generations.jsonl",
        "logs/plain_prompts/qwen_persona_reverse_no_think.generations.jsonl",
    ),
    (
        "llama8",
        "logs/plain_prompts/llama8_persona_direct.generations.jsonl",
        "logs/plain_prompts/llama8_persona_reversed.generations.jsonl",
    ),
]


def parse_numeric_response(raw_text: object, max_val: int) -> int | None:
    if raw_text is None:
        return None
    clean_text = str(raw_text).strip().rstrip(".")
    parenthesized = re.fullmatch(r"\(\s*(\d+)\s*\)", clean_text)
    if parenthesized:
        clean_text = parenthesized.group(1)
    if not re.fullmatch(r"\d+", clean_text):
        return None
    value = int(clean_text)
    if value < 1 or value > max_val:
        return None
    return value


def prompt_max_value(prompt_index: int) -> int:
    return 7 if prompt_index in [1, 2, 3, 4, 5, 12, 13, 14] else 4


def prompt_range(prompt_index: int) -> int:
    return prompt_max_value(prompt_index) - 1


def binarize(prompt_index: int, value: int) -> int:
    column = SENSITIVE_COLUMNS[prompt_index - 1]
    if column in ["ma01b", "ma02", "ma03", "ma04", "mp02", "mm03", "mm04"]:
        return int(value >= 5)
    if column in ["ca13", "ca08", "fr10", "fr04b", "fr03b", "pi08"]:
        return int(value <= 2)
    if column == "mm05":
        return int(value <= 3)
    raise ValueError(f"No binarization rule for prompt {prompt_index}: {column}")


def load_plain_jsonl(path: Path, condition: str, model_key: str) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            prompt_index = int(record["prompt_index"])
            value = parse_numeric_response(
                record.get("answer"),
                max_val=prompt_max_value(prompt_index),
            )
            rows.append(
                {
                    "model": model_key,
                    "condition": condition,
                    "run_id": record.get("run_id"),
                    "persona_id": record.get("persona_id"),
                    "prompt_index": prompt_index,
                    "allbus_variable": SENSITIVE_COLUMNS[prompt_index - 1],
                    "answer_raw": record.get("answer"),
                    "answer_numeric": value,
                    "answer_binary": np.nan if value is None else binarize(prompt_index, value),
                }
            )
    return pd.DataFrame(rows)


def summarize_condition(df: pd.DataFrame, allbus: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model", "condition", "prompt_index", "allbus_variable"], dropna=False)
        .agg(
            n_total=("answer_raw", "size"),
            valid_runs=("answer_numeric", "count"),
            invalid_runs=("answer_numeric", lambda x: int(x.isna().sum())),
            llm_mean=("answer_numeric", "mean"),
            llm_answer_std=("answer_numeric", "std"),
            pi_direct=("answer_binary", "mean"),
        )
        .reset_index()
    )
    summary["llm_binary_distribution"] = summary["pi_direct"].map(
        lambda x: np.nan if pd.isna(x) else f"{1 - x},{x}"
    )
    summary = summary.merge(
        allbus[["allbus_variable", "sensitive_id", "sensitive_text", "pi_allbus", "n_allbus"]],
        on="allbus_variable",
        how="left",
    )
    summary["bias_direct"] = summary["pi_direct"] - summary["pi_allbus"]
    summary["absolute_bias_direct"] = summary["bias_direct"].abs()
    return summary


def summarize_ordering(df: pd.DataFrame, allbus: pd.DataFrame) -> pd.DataFrame:
    direct = df[df["condition"] == "direct"]
    reversed_df = df[df["condition"] == "reversed"]
    paired = direct.merge(
        reversed_df,
        on=["model", "persona_id", "prompt_index", "allbus_variable"],
        suffixes=("_direct", "_reversed"),
    )
    rows = []
    for keys, group in paired.groupby(["model", "prompt_index", "allbus_variable"]):
        model, prompt_index, allbus_variable = keys
        direct_valid = group["answer_numeric_direct"].notna()
        reversed_valid = group["answer_numeric_reversed"].notna()
        both_valid = direct_valid & reversed_valid
        rows.append(
            {
                "model": model,
                "prompt_index": prompt_index,
                "allbus_variable": allbus_variable,
                "paired_personas": len(group),
                "both_valid": int(both_valid.sum()),
                "direct_mean": group.loc[direct_valid, "answer_numeric_direct"].mean(),
                "reversed_mean": group.loc[reversed_valid, "answer_numeric_reversed"].mean(),
                "mean_delta_reversed_minus_direct": (
                    group.loc[both_valid, "answer_numeric_reversed"]
                    - group.loc[both_valid, "answer_numeric_direct"]
                ).mean(),
                "order_delta_original_minus_reversed": (
                    group.loc[both_valid, "answer_numeric_direct"]
                    - group.loc[both_valid, "answer_numeric_reversed"]
                ).mean(),
                "scale_range": prompt_range(int(prompt_index)),
                "direct_pi": group.loc[direct_valid, "answer_binary_direct"].mean(),
                "reversed_pi": group.loc[reversed_valid, "answer_binary_reversed"].mean(),
                "pi_delta_reversed_minus_direct": (
                    group.loc[both_valid, "answer_binary_reversed"]
                    - group.loc[both_valid, "answer_binary_direct"]
                ).mean(),
            }
        )
    out = pd.DataFrame(rows)
    out = out.merge(
        allbus[["allbus_variable", "sensitive_id", "sensitive_text", "pi_allbus"]],
        on="allbus_variable",
        how="left",
    )
    out["normalized_order_delta_original_minus_reversed"] = (
        out["order_delta_original_minus_reversed"] / out["scale_range"]
    )
    out["absolute_normalized_order_bias"] = (
        out["normalized_order_delta_original_minus_reversed"].abs()
    )
    out["direct_bias"] = out["direct_pi"] - out["pi_allbus"]
    out["reversed_bias"] = out["reversed_pi"] - out["pi_allbus"]
    out["absolute_bias_delta_reversed_minus_direct"] = (
        out["reversed_bias"].abs() - out["direct_bias"].abs()
    )
    return out


def parse_run_specs(specs: list[str]) -> list[tuple[str, Path, Path]]:
    if not specs:
        return [(name, Path(direct), Path(reversed_path)) for name, direct, reversed_path in DEFAULT_RUNS]

    parsed = []
    for spec in specs:
        parts = spec.split(":", maxsplit=2)
        if len(parts) != 3:
            raise ValueError("--run must be model:direct_jsonl:reversed_jsonl")
        parsed.append((parts[0], Path(parts[1]), Path(parts[2])))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allbus-file",
        default="outputs/allbus_approval_rates.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plain_prompt_evaluation",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="model:direct_jsonl:reversed_jsonl. Can be passed multiple times.",
    )
    args = parser.parse_args()

    allbus = pd.read_csv(args.allbus_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for model_key, direct_path, reversed_path in parse_run_specs(args.run):
        if direct_path.exists():
            frames.append(load_plain_jsonl(direct_path, "direct", model_key))
        else:
            print(f"Skipping missing direct file for {model_key}: {direct_path}")
        if reversed_path.exists():
            frames.append(load_plain_jsonl(reversed_path, "reversed", model_key))
        else:
            print(f"Skipping missing reversed file for {model_key}: {reversed_path}")

    if not frames:
        raise SystemExit("No JSONL files found.")

    records = pd.concat(frames, ignore_index=True)
    condition_summary = summarize_condition(records, allbus)

    records.to_csv(output_dir / "plain_prompt_records.csv", index=False)
    condition_summary.to_csv(output_dir / "plain_prompt_bias_by_condition.csv", index=False)

    if {"direct", "reversed"}.issubset(set(records["condition"])):
        ordering_summary = summarize_ordering(records, allbus)
        ordering_summary.to_csv(output_dir / "plain_prompt_ordering_effects.csv", index=False)
    else:
        ordering_summary = pd.DataFrame()

    print(f"Records: {len(records)}")
    print(f"Condition summary: {output_dir / 'plain_prompt_bias_by_condition.csv'}")
    if not ordering_summary.empty:
        print(f"Ordering effects: {output_dir / 'plain_prompt_ordering_effects.csv'}")


if __name__ == "__main__":
    main()
