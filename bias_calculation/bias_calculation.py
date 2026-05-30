import pandas as pd
import numpy as np

import os
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))


parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vllm_runner import load_model, run_prompt, generate_from_messages
from prompt_loader import load_prompts
from vllm import SamplingParams



if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, "..", "prompts.txt")
    prompts_list = load_prompts(prompts_path, mode="lines")

    llm=load_model()

    antworten = []

    for i in range(2):

        messages = [
        {
            "role": "system", 
            "content": "Du bist ein automatisches System, antworte ausschließlich mit einer einzigen ziffer."
        },

        {
            "role": "user", 
            "content": prompts_list[i] 
        }
        ]

        antwort = generate_from_messages(llm, max_tokens=5, temperature=1.0, messages=messages)
        print(antwort)


