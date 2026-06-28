from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Optional


DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"


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
    quantization: str | None = None,
    load_format: str | None = None,
) -> Any:
    from vllm import LLM

    hf_token = load_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    print(f"Model: {model_id}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Quantization: {quantization or 'none'}")
    print(f"Load format: {load_format or 'auto'}")

    model_kwargs = {}
    if quantization:
        model_kwargs["quantization"] = quantization
    if load_format:
        model_kwargs["load_format"] = load_format

    return LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=False,
        **model_kwargs,
    )


def generate_from_messages(
    llm: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 80,
    temperature: float = 0.0,
) -> str:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    outputs = llm.chat(messages, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def run_prompt(
    llm: Any,
    prompt: str,
    max_tokens: int = 80,
    temperature: float = 0.2,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    return generate_from_messages(
        llm,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
