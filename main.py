from __future__ import annotations

import argparse

from experiments.list_experiment import runner as list_experiment
from experiments.plain_prompts import runner as plain_prompts
from vllm_runner import DEFAULT_MODEL_ID


DEFAULT_MAX_TOKENS = 80
DEFAULT_TEMPERATURE = 0.2


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-each-response", action="store_true")

    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--load-format", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, default=128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run project tasks such as list experiments or plain prompt generation."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    list_parser = subparsers.add_parser("list", help="Run the list experiment task.")
    add_shared_arguments(list_parser)
    list_experiment.add_arguments(list_parser)

    plain_parser = subparsers.add_parser(
        "plain", help="Run vLLM over prompts from a file."
    )
    add_shared_arguments(plain_parser)
    plain_prompts.add_arguments(plain_parser)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task == "list":
        list_experiment.run(args)
    else:
        plain_prompts.run(args)


if __name__ == "__main__":
    main()
