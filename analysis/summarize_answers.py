import json
import math
import re
import statistics


JSONL_FILE = "logs/list_experiment/20260623T233922Z.generations.jsonl"
TARGET_MARGIN = 0.15


rows = []
with open(JSONL_FILE, encoding="utf-8") as file:
    for line in file:
        rows.append(json.loads(line))

has_sensitive = any(row["sensitive_id"] is not None for row in rows)
groups = {}
all_answers = []

for row in rows:
    match = re.search(r"\d+", str(row["answer"]))
    if match is None:
        continue

    answer = int(match.group())
    all_answers.append(answer)

    if has_sensitive:
        if row["sensitive_id"] is None:
            continue
        group_id = row["sensitive_id"]
        text = row["sensitive_text"]
    else:
        group_id = "control"
        text = ""

    if group_id not in groups:
        groups[group_id] = {
            "text": text,
            "answers": [],
        }

    groups[group_id]["answers"].append(answer)


print("id\tn\tmean\tanswers\ttext")
for group_id, group in groups.items():
    answers = group["answers"]
    mean_answer = sum(answers) / len(answers)
    answers_text = ", ".join(str(answer) for answer in answers)
    print(f"{group_id}\t{len(answers)}\t{mean_answer:.2f}\t{answers_text}\t{group['text']}")


if len(all_answers) >= 2:
    sd_answers = statistics.stdev(all_answers)
    n_raw = math.ceil((1.96 * sd_answers / TARGET_MARGIN) ** 2)
    n_control = math.ceil(n_raw / 4) * 4
    n_treatment = math.ceil(n_raw / 5) * 5

    print()
    print("Sample size suggestion")
    print(f"sd = {sd_answers:.3f}")
    print(f"target margin = {TARGET_MARGIN}")
    print(f"raw n = {n_raw}")
    print(f"CORE_CONTROL_REPLICATES = {n_control}")
    print(f"REPLICATES_PER_SENSITIVE = {n_treatment}")
