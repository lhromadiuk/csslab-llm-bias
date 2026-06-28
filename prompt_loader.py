"""Compatibility imports for older scripts.

New code should import shared prompt loading from ``experiments.common`` and
list-experiment trial builders from ``experiments.list_experiment.trials``.
"""

from experiments.common import load_prompts
from experiments.list_experiment.trials import (
    CoreControlTrial,
    ListTrial,
    PromptItem,
    build_core_control_trials,
    build_list_trials,
    load_core_items,
    render_statements,
)

__all__ = [
    "CoreControlTrial",
    "ListTrial",
    "PromptItem",
    "build_core_control_trials",
    "build_list_trials",
    "load_core_items",
    "load_prompts",
    "render_statements",
]
