from pathlib import Path

import pandas as pd


SAV_FILE = "data/ZA8831_v1-3-0.sav"
SENSITIVE_STATEMENTS_FILE = "experiments/list_experiment/data/sensitive_statements.txt"
OUTPUT_FILE = "outputs/allbus_approval_rates.csv"

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


def approval(column, value):
    if pd.isna(value) or value <= 0:
        return None

    if column in ["ca13", "ca08"]:
        return int(value in [1, 2])

    if column.startswith("ma") or column in ["mp02", "mm03", "mm04", "mm05"]:
        return int(value >= 5)

    if column.startswith("fr") or column == "pi08":
        return int(value in [1, 2])

    return None


with open(SENSITIVE_STATEMENTS_FILE, encoding="utf-8") as file:
    sensitive_texts = [line.strip() for line in file if line.strip()]

df = pd.read_spss(SAV_FILE, convert_categoricals=False)

rows = []
for index, column in enumerate(SENSITIVE_COLUMNS, start=1):
    values = []
    for value in df[column]:
        approved = approval(column, value)
        if approved is not None:
            values.append(approved)

    rows.append(
        {
            "sensitive_id": f"sens_{index}",
            "allbus_variable": column,
            "sensitive_text": sensitive_texts[index - 1],
            "pi_allbus": sum(values) / len(values),
            "n_allbus": len(values),
        }
    )

out = pd.DataFrame(rows)
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_FILE, index=False)

print(out.to_string(index=False))
print(f"\nSaved: {OUTPUT_FILE}")
