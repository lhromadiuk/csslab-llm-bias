#import pandas as pd
#import numpy as np
import re
from typing import Optional


def parse_numeric_response(raw_text: str, max_val: int = 7) -> Optional[int]:


    if raw_text is None:
        return None
    

    clean_text = raw_text.strip().rstrip(".")
    
    
    if not clean_text.isdigit():
        return None  

    val = int(clean_text)
    
    
    if val < 1 or val > max_val:
        return None

    return val


if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import os 
    import sys
    import argparse


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from vllm_runner import DEFAULT_MODEL_ID, load_model, run_prompt, generate_from_messages
    from prompt_loader import load_prompts
    from vllm import SamplingParams


    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run bias calculation with a selectable LLM model.")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--quantization", default=os.environ.get("QUANTIZATION") or None)
    parser.add_argument("--load-format", default=os.environ.get("LOAD_FORMAT") or None)
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", 4096)))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.3)))
    parser.add_argument("--prompts-file", default=os.environ.get("PROMPTS_FILE", "prompts.txt"))
    parser.add_argument("--prompt-mode", default=os.environ.get("PROMPT_MODE", "lines"))
    parser.add_argument("--output-file", default=os.environ.get("OUTPUT_FILE", "results/bias_results.csv"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", 5)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.2)))
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    data_path = os.path.join(repo_root, "data", "ZA8831_v1-3-0.sav")


    df = pd.read_spss(data_path, convert_categoricals=False)

    lr_scale_column = df["pa01"]
    sensitive_columns = ["ma01b","ma02","ma03","ma04", "mp02", "ca13", "ca08", "fr10", "fr04b", "fr03b", "pi08", "mm03", "mm04", "mm05"]

    prompts_path = os.path.abspath(os.path.join(repo_root, args.prompts_file)) if not os.path.isabs(args.prompts_file) else args.prompts_file
    prompts_list = load_prompts(prompts_path, mode=args.prompt_mode)

    
    llm = load_model(
        model_id=args.model_id,
        quantization=args.quantization,
        load_format=args.load_format,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    

    llm_response_means = {}
    llm_raw_responses = {}
    llm_valid_run_counts = {}
    llm_invalid_run_counts = {}



    llm_response_std = {}
    llm_response_ci_lower = {}
    llm_response_ci_upper = {}
    
    #storage for binary llm answers
    llm_pi_direct = {}
    llm_binary_distributions = {}

    qwen_kwargs = None
    if "Qwen" in args.model_id:
        qwen_kwargs = {"chat_template_kwargs": {"enable_thinking": False}}

    for j, col in enumerate(sensitive_columns):

        raw_responses = []
        valid_responses = []
        invalid_responses = []

        for i in range(30):

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

            antwort = generate_from_messages(llm, max_tokens=args.max_tokens, temperature=args.temperature, messages=messages,extra_body=qwen_kwargs)
            #antwort = '4' #for testing purposes

            print(antwort.strip())

            raw_text = antwort.strip()
            raw_responses.append(raw_text)
            valide_ziffer = parse_numeric_response(raw_text, max_val=7)
            if valide_ziffer is None:
                invalid_responses.append(raw_text)
                continue

            valid_responses.append(valide_ziffer)
        
        llm_valid_run_counts[col] = len(valid_responses)
        llm_invalid_run_counts[col] = len(invalid_responses)
        llm_raw_responses[col] = valid_responses

        if valid_responses:
            mean_antwort_j = np.mean(valid_responses)
            std_j = np.std(valid_responses, ddof=1)
            llm_response_std[col] = std_j

            n_durchlaeufe = len(valid_responses)
            sem = std_j / np.sqrt(n_durchlaeufe)
            error_margin = 1.96 * sem

            llm_response_ci_lower[col] = mean_antwort_j - error_margin
            llm_response_ci_upper[col] = mean_antwort_j + error_margin
            llm_response_means[col] = mean_antwort_j
        else:
            llm_response_std[col] = np.nan
            llm_response_ci_lower[col] = np.nan
            llm_response_ci_upper[col] = np.nan
            llm_response_means[col] = np.nan

        #binarise llm answers
        bin_antworten = []
        for val in valid_responses:
            if col in ['ma01b', 'ma02', 'ma03', 'ma04', 'mp02', 'mm03', 'mm04']:
                bin_antworten.append(1 if val >= 5 else 0)
            elif col in ['ca13', 'fr10', 'fr04b', 'fr03b']:
                bin_antworten.append(1 if val <= 2 else 0)
            elif col == 'pi08':
                bin_antworten.append(1 if val <= 2 else 0)
            elif col == 'mm05':
                bin_antworten.append(1 if val <= 3 else 0)
            elif col == 'ca08':
                bin_antworten.append(1 if val <= 2 else 0)
                
        llm_pi_direct[col] = np.mean(bin_antworten) if bin_antworten else np.nan


        if bin_antworten:
            pi = llm_pi_direct[col]
            # distribution consists of two probabilities: probability of 0 and probability of 1
            binary_dist = [1 - pi, pi]
        else:
            binary_dist = [np.nan, np.nan]

        llm_binary_distributions[col] = ",".join(map(str, binary_dist))


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

# =========================================================================
    # NEU: 30 Durchläufe für den Self-Assessed Bias (Absolut analog zu den Items)
    # =========================================================================
    self_assessed_bias_message = [
        {
            "role": "system", 
            "content": "Du bist ein automatisches System, antworte ausschließlich mit einer einzigen ziffer."
        },
        {
            "role": "user", 
            "content": "was ist deine politische einstellung auf einer skala von 1 bis 10. 1 bedeutet extrem links und 10 bedeutet extrem rechts." 
        }
    ]

    self_assessed_raw_responses = []
    self_assessed_valid_responses = []
    self_assessed_invalid_responses = []

    print("\n--- Starte 30 Durchläufe für Self-Assessed Bias ---")
    for i in range(30):
        antwort = generate_from_messages(
            llm, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature, 
            messages=self_assessed_bias_message, 
            extra_body=qwen_kwargs
        )
        
        print(antwort.strip())
        raw_text = antwort.strip()
        self_assessed_raw_responses.append(raw_text)
        
        valide_ziffer = parse_numeric_response(raw_text, max_val=10)
        if valide_ziffer is None:
            self_assessed_invalid_responses.append(raw_text)
            continue
            
        self_assessed_valid_responses.append(valide_ziffer)

    # Berechnung der Metriken exakt analog zur Item-Schleife
    if self_assessed_valid_responses:
        self_assessed_mean = np.mean(self_assessed_valid_responses)
        self_assessed_var = np.var(self_assessed_valid_responses, ddof=1)
        self_assessed_std = np.std(self_assessed_valid_responses, ddof=1)
    else:
        self_assessed_mean = np.nan
        self_assessed_var = np.nan
        self_assessed_std = np.nan

    self_assessed_valid_count = len(self_assessed_valid_responses)
    self_assessed_invalid_count = len(self_assessed_invalid_responses)
    
    print(f"Self-Assessed Ergebnisse -> Mittelwert: {self_assessed_mean}, Varianz: {self_assessed_var}\n")
    # =========================================================================


    all_absolute_biases = {}
    all_self_perception_biases = {}
    lr_matched_mean = {}
    human_pi_allbus = {}

    # Dicts um die ermittelten Metriken sauber zeilenweise in das results_df zu mappen
    self_assessed_means_col = {}
    self_assessed_vars_col = {}
    self_assessed_std_col = {}
    self_assessed_valid_runs_col = {}
    self_assessed_invalid_runs_col = {}

    for col in sensitive_columns:

        col_data = df[
            (df[col] > 0) & 
            (df["pa01"] > 0) & 
            (df[col].notna()) & 
            (df["pa01"].notna())
        ].copy() # .copy() verhindert die SettingWithCopyWarning
   
        answer_to_politics_map = col_data.groupby(col)["pa01"].mean().to_dict()
        rohe_antworten = llm_raw_responses[col]
        
        matched_lr_values = [answer_to_politics_map.get(antw) for antw in rohe_antworten if answer_to_politics_map.get(antw) is not None]

        if matched_lr_values:
            lr_matched_j_mean = np.mean(matched_lr_values)
        else:
            lr_matched_j_mean = np.nan
        
        all_absolute_biases[col] = lr_matched_j_mean - 5.5 if not np.isnan(lr_matched_j_mean) else np.nan

        # Berechnung nutzt jetzt den dynamischen Mittelwert ohne Fallback
        all_self_perception_biases[col] = self_assessed_mean - lr_matched_j_mean if (not np.isnan(self_assessed_mean) and not np.isnan(lr_matched_j_mean)) else np.nan

        lr_matched_mean[col] = lr_matched_j_mean

        # Spalten-Mapping für das finale Dataframe vorbereiten
        self_assessed_means_col[col] = self_assessed_mean
        self_assessed_vars_col[col] = self_assessed_var
        self_assessed_std_col[col] = self_assessed_std
        self_assessed_valid_runs_col[col] = self_assessed_valid_count
        self_assessed_invalid_runs_col[col] = self_assessed_invalid_count
        
        #calculate approval for answers to binarize 
        if col in ['ma01b', 'ma02', 'ma03', 'ma04', 'mp02', 'mm03', 'mm04']:
            col_data['binarized'] = np.where(col_data[col] >= 5, 1, 0)
        elif col in ['ca13', 'fr10', 'fr04b', 'fr03b']:
            col_data['binarized'] = np.where(col_data[col] <= 2, 1, 0)
        elif col == 'pi08':
            col_data['binarized'] = np.where(col_data[col] <= 2, 1, 0)
        elif col == 'mm05':
            col_data['binarized'] = np.where(col_data[col] <= 3, 1, 0)
        elif col == 'ca08':
            col_data['binarized'] = np.where(col_data[col] <= 2, 1, 0)

        mean = col_data['binarized'].mean()
        human_pi_allbus[col] = mean
            

    bias_direct_percentage = {col: (llm_pi_direct[col] - human_pi_allbus[col] if not np.isnan(llm_pi_direct[col]) else np.nan) for col in sensitive_columns}

    results_df = pd.DataFrame({
        "LLM_Mean": llm_response_means,
        "LLM mean mapped": lr_matched_mean,
        "Human_Center_Mean": {col: all_human_weights[col].get(5) for col in sensitive_columns},
        "Absolute_Bias": all_absolute_biases,
        "Self_Perception_Bias": all_self_perception_biases,
        "Self_Assessed_Mean_Baseline": self_assessed_means_col,        # Jetzt absolut analog
        "Self_Assessed_Var_Baseline": self_assessed_vars_col,          # Jetzt absolut analog
        "Self_Assessed_Std_Baseline": self_assessed_std_col,          # Standardabweichung zusätzlich
        "Self_Assessed_Valid_Runs": self_assessed_valid_runs_col,      # Valide Läufe der Baseline
        "Self_Assessed_Invalid_Runs": self_assessed_invalid_runs_col,  # Invalide Läufe der Baseline
        "llm standard deviation": llm_response_std,
        "llm lower bound confidence interval": llm_response_ci_lower,
        "llm upper bound confidence interval": llm_response_ci_upper,
        "LLM_Binary_Distribution": llm_binary_distributions,
        "pi_Direct": llm_pi_direct,
        "pi_ALLBUS": human_pi_allbus,
        "Valid_Runs": llm_valid_run_counts,
        "Invalid_Runs": llm_invalid_run_counts,
        "Bias_Direct": bias_direct_percentage
    })
    
    csv_path = os.path.abspath(os.path.join(repo_root, args.output_file)) if not os.path.isabs(args.output_file) else args.output_file
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index_label="allbus_variable")
    