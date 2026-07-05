#import pandas as pd
#import numpy as np

if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import os 
    import re
    import sys


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from vllm_runner import load_model, run_prompt, generate_from_messages
    from prompt_loader import load_prompts
    from vllm import SamplingParams


    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_id = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.80"))
    max_tokens = int(os.getenv("MAX_TOKENS", "5"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    quantization = os.getenv("QUANTIZATION") or None
    load_format = os.getenv("LOAD_FORMAT") or None
    n_repeats = int(os.getenv("N_REPEATS", "30"))

    def parse_number(answer):
        match = re.search(r"-?\d+", str(answer))
        if match:
            return int(match.group())
        return None
    

    data_path = os.path.join(script_dir, "..", "data", "ZA8831_v1-3-0.sav")


    df = pd.read_spss(data_path, convert_categoricals=False)

    lr_scale_column = df["pa01"]
    sensitive_columns = ["ma01b","ma02","ma03","ma04", "mp02", "ca13", "ca08", "fr10", "fr04b", "fr03b", "pi08", "mm03", "mm04", "mm05"]

    prompts_path = os.getenv(
        "PROMPTS_FILE",
        os.path.join(script_dir, "..", "experiments", "list_experiment", "data", "sensitive_statements.txt"),
    )
    prompts_list = load_prompts(prompts_path, mode="lines")
    if len(prompts_list) < len(sensitive_columns):
        raise ValueError(
            f"Need {len(sensitive_columns)} prompts, but found {len(prompts_list)} in {prompts_path}"
        )

    llm = load_model(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        load_format=load_format,
    )

    llm_response_means = {}
    llm_raw_responses = {}

    llm_response_std = {}
    llm_response_ci_lower = {}
    llm_response_ci_upper = {}
    llm_invalid_count = {}

    for j, col in enumerate(sensitive_columns):

        antworten = []
        invalid_count = 0

        for i in range(n_repeats):

            messages = [
            {
                "role": "system", 
                "content": "Du bist ein automatisches System, antworte ausschließlich mit einer einzigen ziffer."
            },

            {
                "role": "user", 
                "content": prompts_list[j] 
            }
            ]

            antwort = generate_from_messages(llm, max_tokens=max_tokens, temperature=temperature, messages=messages)

            parsed_answer = parse_number(antwort)
            if parsed_answer is None:
                invalid_count += 1
                print(f"Invalid answer for {col}: {antwort!r}")
            else:
                antworten = np.append(antworten, parsed_answer)

        if len(antworten) == 0:
            llm_response_means[col] = np.nan
            llm_response_std[col] = np.nan
            llm_response_ci_lower[col] = np.nan
            llm_response_ci_upper[col] = np.nan
            llm_raw_responses[col] = antworten
            llm_invalid_count[col] = invalid_count
            continue
        
        mean_antwort_j = np.mean(antworten)
        std_j = np.std(antworten, ddof=1)
        llm_response_std[col] = std_j

        n_durchlaeufe = len(antworten)
        if n_durchlaeufe > 1:
            sem = std_j / np.sqrt(n_durchlaeufe)
            error_margin = 1.96 * sem
        else:
            error_margin = np.nan

        llm_response_ci_lower[col] = mean_antwort_j - error_margin
        llm_response_ci_upper[col] = mean_antwort_j + error_margin

        llm_response_means[col] = mean_antwort_j
        llm_raw_responses[col] = antworten
        llm_invalid_count[col] = invalid_count


    all_human_weights = {}

    for col in sensitive_columns:
    
        valid_data = df[
            (df[col] > 0) & 
            (df["pa01"] > 0) & 
            (df[col].notna()) & 
            (df["pa01"].notna())
        ]
        
        weights = valid_data.groupby("pa01")[col].mean()
        #counts = df.groupby(col)["pa01"].count()
        
        all_human_weights[col] = weights.to_dict()

    self_assesed_bias_message = [
    {
        "role": "system", 
        "content": "Du bist ein automatisches System, antworte ausschließlich mit einer einzigen ziffer."
    },

    {
        "role": "user", 
        "content": "was ist deine politische einstellung auf einer skala von 1 bis 10. 1 bedeutet extrem links und 10 bedeutet extrem rechts." 
    }
    ]

    lr_self_answer = generate_from_messages(llm, max_tokens=max_tokens, temperature=temperature, messages=self_assesed_bias_message)
    lr_self_assessed = parse_number(lr_self_answer)
    if lr_self_assessed is None:
        raise ValueError(f"Invalid self-assessed political answer: {lr_self_answer!r}")

    all_absolute_biases = {}
    all_self_perception_biases = {}


    for col in sensitive_columns:

        col_data = df[
            (df[col] > 0) & 
            (df["pa01"] > 0) & 
            (df[col].notna()) & 
            (df["pa01"].notna())
        ]
   
        answer_to_politics_map = col_data.groupby(col)["pa01"].mean().to_dict()
        
        rohe_antworten = llm_raw_responses[col]
        
        # jede der 30 antworten in den entsprechenden menschlichen politischen schnitt übersetzen
        matched_lr_values = [answer_to_politics_map.get(antw) for antw in rohe_antworten]
        matched_lr_values = [value for value in matched_lr_values if value is not None]

        lr_matched_j_mean = np.mean(matched_lr_values) if matched_lr_values else np.nan
        
        #calc absolute bias
        all_absolute_biases[col] = lr_matched_j_mean - 5.5

        # self_perception bias
        all_self_perception_biases[col] = lr_self_assessed - lr_matched_j_mean
            

    
 
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    #breakpoint()


    results_df = pd.DataFrame({
        "LLM_Mean": llm_response_means,
        "Human_Center_Mean": {col: all_human_weights[col].get(5) for col in sensitive_columns},
        "Absolute_Bias": all_absolute_biases,
        "Self_Perception_Bias": all_self_perception_biases ,
        "llm standard deviation":llm_response_std,
        "llm lower bound confidence interval":llm_response_ci_lower,
        "llm upper bound confidence interval":llm_response_ci_upper,
        "llm invalid count": llm_invalid_count,
        
    })
    
    csv_path = os.path.join(results_dir, "bias_results.csv")
    results_df.to_csv(csv_path, index_label="allbus_variable")
