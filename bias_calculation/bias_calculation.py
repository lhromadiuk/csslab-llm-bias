import pandas as pd
import numpy as np

from vllm_runner import load_model, run_prompt, generate_from_messages
from bias_calculation.prompt_loader import load_prompts
from vllm import SamplingParams

prompts_list = load_prompts("prompts.txt", mode="survey")


llm=load_model()


for i in range(2):

    print(run_prompt(llm, prompt=prompts_list[0], max_tokens=80, temperature=1.0))


