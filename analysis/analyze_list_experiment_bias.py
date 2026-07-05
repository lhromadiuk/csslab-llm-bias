import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


JSONL_FILES = [
    "logs/list_experiment/20260703T211701Z.generations.jsonl",
]
ALLBUS_FILE = "outputs/allbus_approval_rates.csv"
DIRECT_FILE = ""  # optional csv with columns: sensitive_id, pi_direct
OUTPUT_DIR = Path("outputs/list_experiment_bias")
VARIANTS_PER_ALLBUS_ITEM = 3  # wording alteration file has 3 variants per original item


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_int(answer):
    text = str(answer).strip()
    match = re.fullmatch(r"-?\d+\.?", text)
    if match:
        return int(text.rstrip("."))
    return None


def sensitive_number(sensitive_id):
    if pd.isna(sensitive_id):
        return None
    return int(str(sensitive_id).replace("sens_", ""))


def allbus_sensitive_id(sensitive_id, has_wording_variants):
    number = sensitive_number(sensitive_id)
    if number is None:
        return pd.NA
    if has_wording_variants:
        base_number = math.ceil(number / VARIANTS_PER_ALLBUS_ITEM)
    else:
        base_number = number
    return f"sens_{base_number}"


def wording_variant(sensitive_id, has_wording_variants):
    number = sensitive_number(sensitive_id)
    if number is None:
        return pd.NA
    if not has_wording_variants:
        return 1
    return ((number - 1) % VARIANTS_PER_ALLBUS_ITEM) + 1


def ci_for_difference(control, treatment):
    n_control = len(control)
    n_treatment = len(treatment)
    var_control = control.var()
    var_treatment = treatment.var()

    if n_control < 2 or n_treatment < 2:
        return pd.Series({"ci_lower": pd.NA, "ci_upper": pd.NA, "p_value": pd.NA})

    se = math.sqrt(var_control / n_control + var_treatment / n_treatment)
    if se == 0 or pd.isna(se):
        return pd.Series({"ci_lower": pd.NA, "ci_upper": pd.NA, "p_value": pd.NA})

    df_num = (var_control / n_control + var_treatment / n_treatment) ** 2
    df_den = ((var_control / n_control) ** 2 / (n_control - 1)) + (
        (var_treatment / n_treatment) ** 2 / (n_treatment - 1)
    )
    dof = df_num / df_den
    mean_diff = treatment.mean() - control.mean()
    ci = stats.t.interval(0.95, dof, loc=mean_diff, scale=se)
    ttest = stats.ttest_ind(treatment, control, equal_var=False)
    return pd.Series({"ci_lower": ci[0], "ci_upper": ci[1], "p_value": ttest.pvalue})


rows = []
bad_lines = 0
for jsonl_file in JSONL_FILES:
    with open(jsonl_file, encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            row["source_file"] = jsonl_file
            rows.append(row)

if bad_lines:
    print(f"Skipped {bad_lines} non-JSON lines.")

df = pd.DataFrame(rows)
df["answer_int"] = df["answer"].apply(get_int)
df["is_control"] = df["trial_kind"] == "core_control"
df["is_treatment"] = df["trial_kind"].isin(["sensitive_treatment", "list"])

df["valid_answer"] = False
df.loc[df["is_control"] & df["answer_int"].between(0, 4), "valid_answer"] = True
df.loc[df["is_treatment"] & df["answer_int"].between(0, 5), "valid_answer"] = True
df["invalid"] = ~df["valid_answer"]

invalid_summary = (
    df.assign(sensitive_group=df["sensitive_id"].fillna("control"))
    .groupby(["model_id", "source_file", "sensitive_group", "trial_kind"], dropna=False)
    .agg(n_total=("invalid", "size"), n_invalid=("invalid", "sum"))
    .reset_index()
)
invalid_summary["invalid_rate"] = invalid_summary["n_invalid"] / invalid_summary["n_total"]
invalid_summary.to_csv(OUTPUT_DIR / "invalid_summary.csv", index=False)

valid = df[df["valid_answer"]].copy()

summary_rows = []
group_columns = ["model_id", "source_file", "persona_profile_column"]

for group_values, group in valid.groupby(group_columns, dropna=False):
    group_info = dict(zip(group_columns, group_values))
    control = group[group["is_control"]]["answer_int"]
    sensitive_ids = sorted(group[group["is_treatment"]]["sensitive_id"].dropna().unique())

    for sensitive_id in sensitive_ids:
        treatment_rows = group[group["sensitive_id"] == sensitive_id]
        treatment = treatment_rows["answer_int"]
        ci = ci_for_difference(control, treatment)
        sensitive_texts = treatment_rows["sensitive_text"].dropna()
        sensitive_text = sensitive_texts.iloc[0] if len(sensitive_texts) else ""

        summary_rows.append(
            {
                **group_info,
                "sensitive_id": sensitive_id,
                "sensitive_text": sensitive_text,
                "mean_control": control.mean(),
                "mean_treatment": treatment.mean(),
                "pi_le": treatment.mean() - control.mean(),
                "var_control": control.var(),
                "var_treatment": treatment.var(),
                "n_valid_control": len(control),
                "n_valid_treatment": len(treatment),
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "p_value": ci["p_value"],
            }
        )

summary = pd.DataFrame(summary_rows)
max_sensitive_number = summary["sensitive_id"].apply(sensitive_number).max()
has_wording_variants = max_sensitive_number > 14
summary["allbus_sensitive_id"] = summary["sensitive_id"].apply(
    lambda value: allbus_sensitive_id(value, has_wording_variants)
)
summary["wording_variant"] = summary["sensitive_id"].apply(
    lambda value: wording_variant(value, has_wording_variants)
)

allbus = pd.read_csv(ALLBUS_FILE)
summary = summary.merge(
    allbus[["sensitive_id", "allbus_variable", "pi_allbus"]].rename(
        columns={"sensitive_id": "allbus_sensitive_id"}
    ),
    on="allbus_sensitive_id",
    how="left",
)
summary["bias_le"] = summary["pi_le"] - summary["pi_allbus"]

if DIRECT_FILE:
    direct = pd.read_csv(DIRECT_FILE)
    summary = summary.merge(direct[["sensitive_id", "pi_direct"]], on="sensitive_id", how="left")
    summary["bias_direct"] = summary["pi_direct"] - summary["pi_allbus"]
else:
    summary["pi_direct"] = pd.NA
    summary["bias_direct"] = pd.NA

summary.to_csv(OUTPUT_DIR / "bias_summary_by_item.csv", index=False)

model_summary = (
    summary.groupby(["model_id", "source_file", "persona_profile_column"], dropna=False)
    .agg(
        mean_abs_bias_le=("bias_le", lambda values: values.abs().mean()),
        mean_bias_le=("bias_le", "mean"),
        mean_pi_le=("pi_le", "mean"),
        mean_pi_allbus=("pi_allbus", "mean"),
        mean_n_treatment=("n_valid_treatment", "mean"),
    )
    .reset_index()
)
model_summary.to_csv(OUTPUT_DIR / "bias_summary_by_model.csv", index=False)

print("\nItem-level summary:")
print(summary.to_string(index=False))
print("\nModel-level summary:")
print(model_summary.to_string(index=False))
print(f"\nSaved CSV files in {OUTPUT_DIR}")


plt.figure(figsize=(12, 6))
plot_data = summary.sort_values("sensitive_id")
errors = [
    plot_data["pi_le"] - plot_data["ci_lower"],
    plot_data["ci_upper"] - plot_data["pi_le"],
]
plt.errorbar(
    x=plot_data["sensitive_id"],
    y=plot_data["pi_le"],
    yerr=errors,
    fmt="o",
    capsize=3,
    label="List experiment estimate",
)
plt.scatter(
    plot_data["sensitive_id"],
    plot_data["pi_allbus"],
    color="black",
    marker="x",
    label="ALLBUS approval rate",
)
plt.axhline(0, color="gray", linewidth=1)
plt.xticks(rotation=45)
plt.ylabel("Approval / prevalence")
plt.xlabel("Sensitive item")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pi_le_vs_allbus.png", dpi=200)
plt.close()


plt.figure(figsize=(12, 6))
sns.barplot(data=summary, x="sensitive_id", y="bias_le")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(rotation=45)
plt.ylabel("Bias_LE = pi_LE - pi_ALLBUS")
plt.xlabel("Sensitive item")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bias_le_by_item.png", dpi=200)
plt.close()
