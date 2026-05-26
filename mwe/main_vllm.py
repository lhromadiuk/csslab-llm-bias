import os

from vllm_runner import DEFAULT_MODEL_ID, load_model, run_prompt


SMOKE_TEST_PROMPT = "Reply with exactly: vLLM setup works."


def main() -> None:
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.80"))

    llm = load_model(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print("Running vLLM smoke test...")
    answer = run_prompt(
        llm,
        SMOKE_TEST_PROMPT,
        max_tokens=int(os.getenv("MAX_TOKENS", "16")),
        temperature=0.0,
    )
    print("Smoke test output:")
    print(answer)


if __name__ == "__main__":
    main()
