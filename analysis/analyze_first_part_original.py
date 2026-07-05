import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


OUTPUT_DIR = Path("outputs/first_part_original")
ALLBUS_FILE = "outputs/allbus_approval_rates.csv"

BUCKETS = [
    {
        "bucket": "no_persona",
        "model_label": "Llama-70B",
        "files": [
            "logs/list_experiment/20260704T184203Z.generations.jsonl",
            "logs/list_experiment/20260704T190746Z.generations.jsonl",
        ],
        "parse_mode": "first_number",
    },
    {
        "bucket": "no_persona",
        "model_label": "Qwen3-8B",
        "files": ["logs/list_experiment/qwen_original_no_persona_no_think.generations.jsonl"],
        "parse_mode": "first_number",
    },
    {
        "bucket": "no_persona",
        "model_label": "Mistral-7B",
        "files": ["logs/list_experiment/20260703T163342Z.generations.jsonl"],
        "parse_mode": "first_number",
    },
    {
        "bucket": "persona",
        "model_label": "Llama-70B",
        "files": ["logs/list_experiment/20260624T033052Z.generations.jsonl"],
        "parse_mode": "first_number",
    },
    {
        "bucket": "persona",
        "model_label": "Qwen3-8B",
        "files": ["logs/list_experiment/qwen_with_persona_no_think.generations.jsonl"],
        "parse_mode": "first_number",
    },
    {
        "bucket": "persona",
        "model_label": "Mistral-7B",
        "files": ["logs/list_experiment/20260703T163206Z.generations.jsonl"],
        "parse_mode": "first_number",
    },
]


def answer_to_int(answer, parse_mode):
    text = str(answer).strip()
    if parse_mode == "exact":
        match = re.fullmatch(r"-?\d+\.?", text)
    else:
        match = re.search(r"-?\d+", text)
    if not match:
        return None
    return int(match.group().rstrip("."))


def ttest_ci(control, treatment):
    if len(control) < 2 or len(treatment) < 2:
        return pd.Series({"ci_lower": pd.NA, "ci_upper": pd.NA, "p_value": pd.NA})

    var_control = control.var()
    var_treatment = treatment.var()
    se = math.sqrt(var_control / len(control) + var_treatment / len(treatment))
    if se == 0 or pd.isna(se):
        return pd.Series({"ci_lower": pd.NA, "ci_upper": pd.NA, "p_value": pd.NA})

    diff = treatment.mean() - control.mean()
    df_num = (var_control / len(control) + var_treatment / len(treatment)) ** 2
    df_den = ((var_control / len(control)) ** 2 / (len(control) - 1)) + (
        (var_treatment / len(treatment)) ** 2 / (len(treatment) - 1)
    )
    dof = df_num / df_den
    ci = stats.t.interval(0.95, dof, loc=diff, scale=se)
    p_value = stats.ttest_ind(treatment, control, equal_var=False).pvalue
    return pd.Series({"ci_lower": ci[0], "ci_upper": ci[1], "p_value": p_value})


def load_bucket(bucket_info):
    rows = []
    bad_json = 0
    for jsonl_file in bucket_info["files"]:
        with open(jsonl_file, encoding="utf-8") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue
                row["source_file"] = jsonl_file
                rows.append(row)

    df = pd.DataFrame(rows)
    df["answer_int"] = df["answer"].apply(
        lambda value: answer_to_int(value, bucket_info["parse_mode"])
    )
    df["is_control"] = df["trial_kind"] == "core_control"
    df["is_treatment"] = df["trial_kind"].isin(["sensitive_treatment", "list"])
    df["valid_answer"] = False
    df.loc[df["is_control"] & df["answer_int"].between(0, 4), "valid_answer"] = True
    df.loc[df["is_treatment"] & df["answer_int"].between(0, 5), "valid_answer"] = True
    df["invalid"] = ~df["valid_answer"]
    df["bucket"] = bucket_info["bucket"]
    df["model_label"] = bucket_info["model_label"]
    df["bad_json_lines"] = bad_json
    return df


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
allbus = pd.read_csv(ALLBUS_FILE)

all_rows = []
for bucket_info in BUCKETS:
    df = load_bucket(bucket_info)
    valid = df[df["valid_answer"]].copy()
    control = valid[valid["is_control"]]["answer_int"]
    treatment_all = valid[valid["is_treatment"]]

    sensitive_ids = sorted(
        treatment_all["sensitive_id"].dropna().unique(),
        key=lambda value: int(str(value).replace("sens_", "")),
    )

    for sensitive_id in sensitive_ids:
        treatment = treatment_all[treatment_all["sensitive_id"] == sensitive_id][
            "answer_int"
        ]
        ci = ttest_ci(control, treatment)
        all_rows.append(
            {
                "bucket": bucket_info["bucket"],
                "model_label": bucket_info["model_label"],
                "parse_mode": bucket_info["parse_mode"],
                "files": ";".join(bucket_info["files"]),
                "sensitive_id": sensitive_id,
                "mean_control": control.mean(),
                "mean_treatment": treatment.mean(),
                "pi_le": treatment.mean() - control.mean(),
                "var_control": control.var(),
                "var_treatment": treatment.var(),
                "sd_control": control.std(),
                "sd_treatment": treatment.std(),
                "avg_response_sd": pd.Series([control.std(), treatment.std()]).mean(),
                "n_valid_control": len(control),
                "n_valid_treatment": len(treatment),
                "invalid_rate_total": df["invalid"].mean(),
                "bad_json_lines": int(df["bad_json_lines"].iloc[0]),
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "p_value": ci["p_value"],
            }
        )

summary = pd.DataFrame(all_rows)
summary = summary.merge(
    allbus[["sensitive_id", "allbus_variable", "pi_allbus"]],
    on="sensitive_id",
    how="left",
)
summary["bias_le"] = summary["pi_le"] - summary["pi_allbus"]
summary.to_csv(OUTPUT_DIR / "first_part_item_summary.csv", index=False)

model_summary = (
    summary.groupby(["bucket", "model_label"], as_index=False)
    .agg(
        mean_pi_le=("pi_le", "mean"),
        mean_bias_le=("bias_le", "mean"),
        mean_abs_bias_le=("bias_le", lambda values: values.abs().mean()),
        sd_bias_le=("bias_le", "std"),
        sd_pi_le=("pi_le", "std"),
        avg_response_sd=("avg_response_sd", "mean"),
        mean_invalid_rate=("invalid_rate_total", "mean"),
        mean_n_control=("n_valid_control", "mean"),
        mean_n_treatment=("n_valid_treatment", "mean"),
    )
)
model_summary.to_csv(OUTPUT_DIR / "first_part_model_summary.csv", index=False)

print("\nModel summary:")
print(model_summary.to_string(index=False))
print(f"\nSaved: {OUTPUT_DIR}")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))
sns.barplot(data=model_summary, x="model_label", y="mean_abs_bias_le", hue="bucket")
plt.ylabel("Mean absolute Bias_LE")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_abs_bias_by_model.png", dpi=200)
plt.close()

plt.figure(figsize=(13, 6))
sns.barplot(data=summary, x="sensitive_id", y="bias_le", hue="model_label")
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("Bias_LE = pi_LE - pi_ALLBUS")
plt.xlabel("Sensitive item")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bias_le_by_item_and_model.png", dpi=200)
plt.close()

plt.figure(figsize=(13, 6))
plot_data = summary.melt(
    id_vars=["bucket", "model_label", "sensitive_id"],
    value_vars=["pi_le", "pi_allbus"],
    var_name="metric",
    value_name="value",
)
sns.lineplot(
    data=plot_data,
    x="sensitive_id",
    y="value",
    hue="model_label",
    style="metric",
    markers=True,
    dashes=False,
)
plt.ylabel("Prevalence / approval rate")
plt.xlabel("Sensitive item")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pi_le_vs_allbus_by_model.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(data=model_summary, x="model_label", y="mean_invalid_rate", hue="bucket")
plt.ylabel("Invalid response rate")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "invalid_rate_by_model.png", dpi=200)
plt.close()
