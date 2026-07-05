from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Optional


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def load_hf_token() -> Optional[str]:
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token.strip()

    token_path = Path.home() / "hfkey"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()

    return None


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.80,
    enforce_eager: bool = False,
    max_num_seqs: int | None = None,
    quantization: str | None = None,
    load_format: str | None = None,
) -> Any:
    from vllm import LLM

    hf_token = load_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    print(f"Model: {model_id}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Enforce eager: {enforce_eager}")
    print(f"Max num seqs: {max_num_seqs or 'auto'}")
    print(f"Quantization: {quantization or 'none'}")
    print(f"Load format: {load_format or 'auto'}")

    model_kwargs = {}
    if quantization:
        model_kwargs["quantization"] = quantization
    if load_format:
        model_kwargs["load_format"] = load_format
    if max_num_seqs is not None:
        model_kwargs["max_num_seqs"] = max_num_seqs

    return LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        trust_remote_code=False,
        **model_kwargs,
    )


def generate_from_messages(
    llm: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 80,
    temperature: float = 0.0,
    disable_thinking: bool = False,
) -> str:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    chat_kwargs = {}
    if disable_thinking:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    outputs = llm.chat(messages, sampling_params=sampling_params, **chat_kwargs)
    return outputs[0].outputs[0].text.strip()


def generate_batch_from_messages(
    llm: Any,
    messages_list: list[list[dict[str, str]]],
    max_tokens: int = 80,
    temperature: float = 0.0,
    disable_thinking: bool = False,
) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    chat_kwargs = {}
    if disable_thinking:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    outputs = llm.chat(messages_list, sampling_params=sampling_params, **chat_kwargs)
    return [output.outputs[0].text.strip() for output in outputs]


def run_prompt(
    llm: Any,
    prompt: str,
    max_tokens: int = 80,
    temperature: float = 0.2,
    disable_thinking: bool = False,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    return generate_from_messages(
        llm,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        disable_thinking=disable_thinking,
    )
