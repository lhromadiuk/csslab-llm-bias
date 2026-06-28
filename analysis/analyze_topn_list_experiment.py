import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


JSONL_FILE = "logs/list_experiment/20260624T013437Z.generations.jsonl"
OUTPUT_DIR = Path("outputs/topn_analysis")
CONFIG_ORDER = ["core", "top-2", "top-4", "top-8", "top-16"]
SELECTED_CONFIG = os.environ.get("SELECTED_CONFIG", "top-4")
ALLBUS_APPROVAL_FILE = Path("outputs/allbus_approval_rates.csv")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with open(JSONL_FILE, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

# get integer from model answer
answers = []
for value in df["answer"]:
    match = re.search(r"-?\d+", str(value))
    if match:
        answers.append(int(match.group()))
    else:
        answers.append(None)

df["answer_int"] = answers
df["persona_profile_column"] = df["persona_profile_column"].fillna("no_persona")

df["is_control"] = df["trial_kind"] == "core_control"
df["is_treatment"] = df["trial_kind"].isin(["sensitive_treatment", "list"])

df["valid_answer"] = False
df.loc[df["is_control"] & df["answer_int"].between(0, 4), "valid_answer"] = True
df.loc[df["is_treatment"] & df["answer_int"].between(0, 5), "valid_answer"] = True


# invalid answers
invalid_count = len(df[df["valid_answer"] == False])
invalid_percent = invalid_count / len(df) * 100
print(f"Dropped rows: {invalid_count}/{len(df)} ({invalid_percent:.2f}%)")

df["sensitive_group"] = df["sensitive_id"].fillna("control")
df["invalid"] = df["valid_answer"] == False

invalid_summary = (
    df.groupby(["persona_profile_column", "sensitive_group", "trial_kind"])
    .agg(n_total=("invalid", "size"), n_invalid=("invalid", "sum"))
    .reset_index()
)
invalid_summary["invalid_rate_pct"] = (
    invalid_summary["n_invalid"] / invalid_summary["n_total"] * 100
)

print("\nInvalid response summary:")
print(invalid_summary.to_string(index=False))


# main summary
valid = df[df["valid_answer"] == True].copy()
summary_rows = []

profile_columns = sorted(valid["persona_profile_column"].unique())

for profile_column in profile_columns:
    control_answers = valid[
        (valid["persona_profile_column"] == profile_column)
        & (valid["trial_kind"] == "core_control")
    ]["answer_int"]

    sensitive_ids = sorted(
        valid[
            (valid["persona_profile_column"] == profile_column)
            & (valid["is_treatment"])
        ]["sensitive_id"].dropna().unique()
    )

    for sensitive_id in sensitive_ids:
        treatment_answers = valid[
            (valid["persona_profile_column"] == profile_column)
            & (valid["sensitive_id"] == sensitive_id)
        ]["answer_int"]

        mean_control = control_answers.mean()
        mean_treatment = treatment_answers.mean()
        mean_diff = mean_treatment - mean_control

        var_control = control_answers.var()
        var_treatment = treatment_answers.var()

        n_control = len(control_answers)
        n_treatment = len(treatment_answers)

        ttest = stats.ttest_ind(treatment_answers, control_answers, equal_var=False)
        p_value = ttest.pvalue

        se = math.sqrt(var_control / n_control + var_treatment / n_treatment)
        df_num = (var_control / n_control + var_treatment / n_treatment) ** 2
        df_den = ((var_control / n_control) ** 2 / (n_control - 1)) + (
            (var_treatment / n_treatment) ** 2 / (n_treatment - 1)
        )
        dof = df_num / df_den
        ci = stats.t.interval(0.95, dof, loc=mean_diff, scale=se)

        summary_rows.append(
            {
                "persona_profile_column": profile_column,
                "sensitive_id": sensitive_id,
                "mean_control": mean_control,
                "mean_treatment": mean_treatment,
                "mean_diff": mean_diff,
                "var_control": var_control,
                "var_treatment": var_treatment,
                "n_valid_control": n_control,
                "n_valid_treatment": n_treatment,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "p_value": p_value,
            }
        )

summary = pd.DataFrame(summary_rows)
summary["pi_le"] = summary["mean_diff"]

if ALLBUS_APPROVAL_FILE.exists():
    allbus = pd.read_csv(ALLBUS_APPROVAL_FILE)
    summary = summary.merge(
        allbus[["sensitive_id", "allbus_variable", "pi_allbus"]],
        on="sensitive_id",
        how="left",
    )
    summary["bias_le"] = summary["pi_le"] - summary["pi_allbus"]
else:
    print(
        f"\nALLBUS approval CSV not found: {ALLBUS_APPROVAL_FILE}. "
        "Run bias_calculation/calculate_allbus_approval.py first."
    )
    summary["allbus_variable"] = pd.NA
    summary["pi_allbus"] = pd.NA
    summary["bias_le"] = pd.NA

print("\nSummary table:")
print(summary.to_string(index=False))

summary.to_csv(OUTPUT_DIR / "topn_summary.csv", index=False)
invalid_summary.to_csv(OUTPUT_DIR / "topn_invalid_summary.csv", index=False)


def normalize_series(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return series
    if max_value == min_value:
        return pd.Series(0.5, index=series.index)
    return (series - min_value) / (max_value - min_value)


def jsd_bernoulli(p, q):
    # Jensen-Shannon divergence between two yes/no proportions.
    p = min(max(float(p), 0.0), 1.0)
    q = min(max(float(q), 0.0), 1.0)
    p_dist = [p, 1 - p]
    q_dist = [q, 1 - q]
    m_dist = [(p_dist[0] + q_dist[0]) / 2, (p_dist[1] + q_dist[1]) / 2]

    total = 0
    for dist in [p_dist, q_dist]:
        kl = 0
        for value, middle in zip(dist, m_dist):
            if value > 0:
                kl += value * math.log(value / middle)
        total += kl
    return total / 2


config_summary = pd.DataFrame({"persona_profile_column": CONFIG_ORDER})

invalid_by_config = (
    df.groupby("persona_profile_column", as_index=False)["invalid"]
    .mean()
    .rename(columns={"invalid": "invalid_rate"})
)
invalid_by_config["invalid_rate_pct"] = invalid_by_config["invalid_rate"] * 100
invalid_by_config = invalid_by_config.drop(columns=["invalid_rate"])

variance_by_config = (
    valid.groupby("persona_profile_column", as_index=False)["answer_int"]
    .var()
    .rename(columns={"answer_int": "mean_variance"})
)

bias_by_config = (
    summary.dropna(subset=["bias_le"])
    .assign(abs_bias_le=lambda data: data["bias_le"].abs())
    .groupby("persona_profile_column", as_index=False)["abs_bias_le"]
    .mean()
    .rename(columns={"abs_bias_le": "mean_abs_bias"})
)

jsd_rows = []
for profile_column in CONFIG_ORDER:
    profile_summary = summary[
        (summary["persona_profile_column"] == profile_column)
        & summary["pi_le"].notna()
        & summary["pi_allbus"].notna()
    ]
    jsd_values = []
    for _, row in profile_summary.iterrows():
        jsd_values.append(jsd_bernoulli(row["pi_le"], row["pi_allbus"]))
    if jsd_values:
        jsd_rows.append(
            {
                "persona_profile_column": profile_column,
                "mean_jsd": sum(jsd_values) / len(jsd_values),
            }
        )

jsd_by_config = pd.DataFrame(jsd_rows)

config_summary = config_summary.merge(invalid_by_config, on="persona_profile_column", how="left")
config_summary = config_summary.merge(variance_by_config, on="persona_profile_column", how="left")

config_summary = config_summary.merge(bias_by_config, on="persona_profile_column", how="left")
config_summary = config_summary.merge(jsd_by_config, on="persona_profile_column", how="left")
config_summary.to_csv(OUTPUT_DIR / "config_metric_summary.csv", index=False)

pilot_long = config_summary.melt(
    id_vars=["persona_profile_column"],
    value_vars=["invalid_rate_pct", "mean_jsd", "mean_variance"],
    var_name="metric",
    value_name="value",
).dropna(subset=["value"])

pilot_label_map = {
    "invalid_rate_pct": "invalid rate (↓ better)",
    "mean_jsd": "JSD (↓ better)",
    "mean_variance": "variance (↑ better)",
}

pilot_color_map = {
    "invalid_rate_pct": "red",
    "mean_jsd": "blue",
    "mean_variance": "green",
}

pilot_long["metric"] = pd.Categorical(
    pilot_long["metric"],
    categories=["invalid_rate_pct", "mean_jsd", "mean_variance"],
    ordered=True,
)
pilot_long["value_scaled"] = pilot_long.groupby("metric")["value"].transform(
    normalize_series
)
pilot_long["config_index"] = pilot_long["persona_profile_column"].map(
    {config: index for index, config in enumerate(CONFIG_ORDER)}
)

pilot_selected_config = "top-2"
pilot_selected_index = CONFIG_ORDER.index(pilot_selected_config)

plt.figure(figsize=(12, 6))
pilot_plot_data = pilot_long.sort_values("config_index")
for metric in ["invalid_rate_pct", "mean_jsd", "mean_variance"]:
    metric_data = pilot_plot_data[pilot_plot_data["metric"] == metric]
    plt.plot(
        metric_data["config_index"],
        metric_data["value_scaled"],
        marker="o",
        color=pilot_color_map[metric],
        label=pilot_label_map[metric],
    )
plt.xticks(range(len(CONFIG_ORDER)), CONFIG_ORDER)
plt.xlabel("Attribute Configuration")
plt.ylabel("Normalized Metric Value")
plt.axvline(pilot_selected_index, linestyle="--", color="black", alpha=0.7)
plt.annotate(
    "Selected",
    xy=(pilot_selected_index, 0.95),
    xytext=(pilot_selected_index + 0.15, 0.95),
    arrowprops={"arrowstyle": "->", "color": "black"},
)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path("outputs/pilot_metric_comparison.png"), dpi=200)
plt.close()

config_long = config_summary.melt(
    id_vars=["persona_profile_column"],
    value_vars=["invalid_rate_pct", "mean_variance", "mean_abs_bias"],
    var_name="metric",
    value_name="value",
).dropna(subset=["value"])

metric_label_map = {
    "invalid_rate_pct": "Mean invalid rate (%)",
    "mean_variance": "Mean variance",
    "mean_abs_bias": "Mean absolute bias",
}

config_long["metric"] = pd.Categorical(
    config_long["metric"],
    categories=["invalid_rate_pct", "mean_variance", "mean_abs_bias"],
    ordered=True,
)
config_long["value_scaled"] = config_long.groupby("metric")["value"].transform(
    normalize_series
)
config_long["config_index"] = config_long["persona_profile_column"].map(
    {config: index for index, config in enumerate(CONFIG_ORDER)}
)

selected_index = CONFIG_ORDER.index(SELECTED_CONFIG) if SELECTED_CONFIG in CONFIG_ORDER else None

plt.figure(figsize=(12, 6))
plot_data = config_long.sort_values("config_index")
for metric, label in metric_label_map.items():
    metric_data = plot_data[plot_data["metric"] == metric]
    plt.plot(
        metric_data["config_index"],
        metric_data["value_scaled"],
        marker="o",
        label=label,
    )
plt.xticks(range(len(CONFIG_ORDER)), CONFIG_ORDER)
plt.ylabel("Normalized value within each metric")
plt.xlabel("Attribute configuration")
if selected_index is not None:
    plt.axvline(selected_index, linestyle="--", color="black", alpha=0.6)
plt.grid(True, axis="y", alpha=0.3)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "combined_metric_trends_by_config.png", dpi=200)
plt.close()


# plot 1: mean differences
plt.figure(figsize=(14, 7))
sns.barplot(
    data=summary,
    x="sensitive_id",
    y="mean_diff",
    hue="persona_profile_column",
)
plt.axhline(0, color="black")
plt.ylabel("Mean difference")
plt.xlabel("Sensitive item")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_difference_by_topn.png", dpi=200)
plt.close()


# plot 1a: pi_LE estimates
plt.figure(figsize=(14, 7))
sns.barplot(
    data=summary,
    x="sensitive_id",
    y="pi_le",
    hue="persona_profile_column",
)
plt.axhline(0, color="black")
plt.ylabel("pi_LE = treatment mean - control mean")
plt.xlabel("Sensitive item")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pi_le_by_topn.png", dpi=200)
plt.close()


# plot 1b: LE bias against ALLBUS
if summary["bias_le"].notna().any():
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=summary,
        x="sensitive_id",
        y="bias_le",
        hue="persona_profile_column",
    )
    plt.axhline(0, color="black")
    plt.ylabel("BiasLE = pi_LE - pi_ALLBUS")
    plt.xlabel("Sensitive item")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bias_le_by_topn.png", dpi=200)
    plt.close()


# plot 2: violin distributions
plot_df = valid.copy()
plot_df["condition"] = "treatment"
plot_df.loc[plot_df["trial_kind"] == "core_control", "condition"] = "control"
plot_df["sensitive_plot"] = plot_df["sensitive_id"].fillna("control")

g = sns.catplot(
    data=plot_df,
    x="sensitive_plot",
    y="answer_int",
    hue="condition",
    col="persona_profile_column",
    kind="violin",
    col_wrap=2,
    height=4,
    aspect=1.5,
)
g.set_xticklabels(rotation=45, ha="right")
g.savefig(OUTPUT_DIR / "answer_distributions_violin_by_topn.png", dpi=200)
plt.close("all")


# plot 3: invalid response rate
plt.figure(figsize=(14, 7))
sns.barplot(
    data=invalid_summary,
    x="sensitive_group",
    y="invalid_rate_pct",
    hue="persona_profile_column",
)
plt.ylabel("Invalid response rate (%)")
plt.xlabel("Sensitive item")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "invalid_response_rate_by_topn.png", dpi=200)
plt.close()

print(f"\nSaved files in {OUTPUT_DIR}")
