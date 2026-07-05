from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Optional, Union

from experiments.common import load_prompts


@dataclass(frozen=True)
class PromptItem:
    item_id: str
    text: str
    kind: str


@dataclass(frozen=True)
class ListTrial:
    sensitive_id: str
    sensitive_text: str
    replicate_index: int
    order_block_index: int
    presented_items: list[PromptItem]
    sensitive_position: int
    rendered_prompt: str


@dataclass(frozen=True)
class CoreControlTrial:
    replicate_index: int
    order_block_index: int
    presented_items: list[PromptItem]
    rendered_prompt: str


def render_statements(items: list[PromptItem]) -> str:
    return "\n".join(item.text for item in items)


def load_core_items(core_statements_file: Union[str, Path]) -> list[PromptItem]:
    core_statements = load_prompts(core_statements_file, mode="lines")
    if len(core_statements) != 4:
        raise ValueError(
            f"Expected exactly 4 core statements in {core_statements_file}, got {len(core_statements)}"
        )

    return [
        PromptItem(item_id=f"core_{index}", text=text, kind="core")
        for index, text in enumerate(core_statements, start=1)
    ]


def build_list_trials(
    core_statements_file: Union[str, Path],
    sensitive_file: Union[str, Path],
    replicates_per_sensitive: int = 1,
    seed: Optional[int] = None,
    shuffle_sensitive: bool = False,
) -> list[ListTrial]:
    if replicates_per_sensitive < 1:
        raise ValueError("replicates_per_sensitive must be >= 1")
    if replicates_per_sensitive % 5 != 0:
        raise ValueError(
            "replicates_per_sensitive must be a multiple of 5 for balanced "
            "order-effect runs"
        )

    sensitive_statements = load_prompts(sensitive_file, mode="lines")
    if not sensitive_statements:
        raise ValueError(f"No sensitive statements found in {sensitive_file}")

    rng = random.Random(seed)
    sensitive_with_ids = [
        (f"sens_{index}", statement)
        for index, statement in enumerate(sensitive_statements, start=1)
    ]
    if shuffle_sensitive:
        rng.shuffle(sensitive_with_ids)

    core_items = load_core_items(core_statements_file)

    trials: list[ListTrial] = []
    for sensitive_id, sensitive_text in sensitive_with_ids:
        for order_block_index in range(1, (replicates_per_sensitive // 5) + 1):
            sensitive_positions = list(range(1, 6))
            rng.shuffle(sensitive_positions)
            core_order = list(core_items)
            rng.shuffle(core_order)

            for position_index, sensitive_position in enumerate(
                sensitive_positions, start=1
            ):
                replicate_index = ((order_block_index - 1) * 5) + position_index
                trial_items: list[PromptItem] = []
                for display_position in range(1, 6):
                    if display_position == sensitive_position:
                        trial_items.append(
                            PromptItem(
                                item_id=sensitive_id,
                                text=sensitive_text,
                                kind="sensitive",
                            )
                        )
                    else:
                        core_index = (display_position - sensitive_position) % 5
                        trial_items.append(core_order[core_index - 1])

                trials.append(
                    ListTrial(
                        sensitive_id=sensitive_id,
                        sensitive_text=sensitive_text,
                        replicate_index=replicate_index,
                        order_block_index=order_block_index,
                        presented_items=trial_items,
                        sensitive_position=sensitive_position,
                        rendered_prompt=render_statements(trial_items),
                    )
                )

    return trials


def build_core_control_trials(
    core_statements_file: Union[str, Path],
    core_control_replicates: int = 4,
    seed: Optional[int] = None,
) -> list[CoreControlTrial]:
    if core_control_replicates < 1:
        raise ValueError("core_control_replicates must be >= 1")
    if core_control_replicates % 4 != 0:
        raise ValueError(
            "core_control_replicates must be a multiple of 4 for balanced "
            "core-only control runs"
        )

    rng = random.Random(seed)
    core_items = load_core_items(core_statements_file)

    trials: list[CoreControlTrial] = []
    for order_block_index in range(1, (core_control_replicates // 4) + 1):
        first_positions = list(range(4))
        rng.shuffle(first_positions)
        core_order = list(core_items)
        rng.shuffle(core_order)

        for position_index, first_position in enumerate(first_positions, start=1):
            replicate_index = ((order_block_index - 1) * 4) + position_index
            trial_items = [
                core_order[(display_position + first_position) % 4]
                for display_position in range(4)
            ]
            trials.append(
                CoreControlTrial(
                    replicate_index=replicate_index,
                    order_block_index=order_block_index,
                    presented_items=trial_items,
                    rendered_prompt=render_statements(trial_items),
                )
            )

    return trials
