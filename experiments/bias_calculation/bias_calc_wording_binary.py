#import pandas as pd
#import numpy as np
import re

from typing import Optional

def parse_numeric_response(raw_text: str) -> Optional[int]:
    """Extract the numeric response ONLY if the output is clean and unambiguous.
    
    Accepts: "4", "4.", "  4  "
    Rejects: "Antwort: 4", "4 out of 10", or long conversational text containing numbers.
    """
    if raw_text is None:
        return None

    # Säubern von Leerzeichen und Punkten am Ende (z.B. "4." -> "4")
    clean_text = raw_text.strip().rstrip(".")
    
    # STRENGE PRÜFUNG: Besteht der restliche Text NUR noch aus einer reinen Zahl?
    if not clean_text.isdigit():
        return None  # Text enthält Gelaber oder mehrere Zahlen -> Ablehnen!

    val = int(clean_text)
    

    if val < 0 or val > 1:
        return None

    return val



if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import os 
    import sys
    import argparse


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    parser.add_argument("--prompts-file", default=os.environ.get("PROMPTS_FILE", "prompts_wording_variations copy.txt"))
    parser.add_argument("--prompt-mode", default=os.environ.get("PROMPT_MODE", "lines"))
    parser.add_argument("--output-file", default=os.environ.get("OUTPUT_FILE", "results/bias_results.csv"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", 5)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.2)))
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

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
    

    # Hier erstellen wir Listen für den DataFrame-Export, um die 3 Variationen sauber zu trennen
    extended_rows = []
    for col in sensitive_columns:
        for v in [1, 2, 3]:
            extended_rows.append(f"{col}_v{v}")

    llm_response_means = {}
    llm_raw_responses = {}
    llm_valid_run_counts = {}
    llm_invalid_run_counts = {}

    llm_response_std = {}
    llm_response_ci_lower = {}
    llm_response_ci_upper = {}
    llm_pi_direct = {}

    # ÄNDERUNG 1: Wir iterieren über alle 42 Prompts der prompts.txt
    for idx, prompt_text in enumerate(prompts_list):
        # Berechnen, zu welcher originalen Spalte (0-13) und welcher Variante (1-3) dieser Prompt gehört
        j = idx // 3
        v = (idx % 3) + 1
        col = sensitive_columns[j]
        
        # Eindeutiger Bezeichner für die Ergebnismaps (z.B. "ma01b_v1")
        v_col = f"{col}_v{v}"

        raw_responses = []
        valid_responses = []
        invalid_responses = []

        for i in range(30):
            messages = [
                {
                    "role": "system", 
                    "content": "Du bist ein automatisches System, antworte ausschließlich mit einer einzigen ziffer. 1 bedeutet stimme zu und 0 bedeutet stimme nicht zu."
                },
                {
                    "role": "user", 
                    "content": prompt_text  # Nutzt den exakten Prompt aus der Schleife
                }
            ]

            antwort = generate_from_messages(llm, max_tokens=args.max_tokens, temperature=args.temperature, messages=messages)
            #antwort = '4' #temporary test for debugging, replace with the above line in production

            raw_text = antwort.strip()
            raw_responses.append(raw_text)
            valide_ziffer = parse_numeric_response(raw_text)
            if valide_ziffer is None:
                invalid_responses.append(raw_text)
                continue

            valid_responses.append(valide_ziffer)
        
        llm_valid_run_counts[v_col] = len(valid_responses)
        llm_invalid_run_counts[v_col] = len(invalid_responses)
        llm_raw_responses[v_col] = valid_responses

        if valid_responses:
            mean_antwort_j = np.mean(valid_responses)
            std_j = np.std(valid_responses, ddof=1)
            llm_response_std[v_col] = std_j

            n_durchlaeufe = len(valid_responses)
            sem = std_j / np.sqrt(n_durchlaeufe)
            error_margin = 1.96 * sem

            llm_response_ci_lower[v_col] = mean_antwort_j - error_margin
            llm_response_ci_upper[v_col] = mean_antwort_j + error_margin
            llm_response_means[v_col] = mean_antwort_j
        else:
            llm_response_std[v_col] = np.nan
            llm_response_ci_lower[v_col] = np.nan
            llm_response_ci_upper[v_col] = np.nan
            llm_response_means[v_col] = np.nan

        # Binarisierung (Nutzt die Logik der originalen 'col', speichert aber in 'v_col')
        bin_antworten = []
        for val in valid_responses:
            if col in ['ma01b', 'ma02', 'ma03', 'ma04', 'mp02', 'mm03', 'mm04']:
                bin_antworten.append(1 if val >= 5 else 0)
            elif col in ['ca13', 'fr10', 'fr04b', 'fr03b']:
                bin_antworten.append(1 if val <= 2 else 0)
            elif col == 'pi08':
                bin_antworten.append(1 if val <= 3 else 0)
            elif col == 'mm05':
                bin_antworten.append(1 if val <= 5 else 0)
            elif col == 'ca08':
                bin_antworten.append(1 if val >= 3 else 0)
                
        llm_pi_direct[v_col] = np.mean(bin_antworten) if bin_antworten else np.nan


    # Ab hier rechnen wir die menschlichen Vergleichswerte aus dem ALLBUS-Datensatz
    all_human_weights = {}
    human_pi_allbus = {}

    for col in sensitive_columns:
        valid_data = df[
            (df[col] > 0) & 
            (df["pa01"] > 0) & 
            (df[col].notna()) & 
            (df["pa01"].notna())
        ].copy() # copy() verhindert SettingWithCopyWarning
        
        weights = valid_data.groupby("pa01")[col].mean()
        all_human_weights[col] = weights.to_dict()

        # Berechne Binarisierung für Menschen
        if col in ['ma01b', 'ma02', 'ma03', 'ma04', 'mp02', 'mm03', 'mm04']:
            valid_data['binarized'] = np.where(valid_data[col] >= 5, 1, 0)
        elif col in ['ca13', 'fr10', 'fr04b', 'fr03b']:
            valid_data['binarized'] = np.where(valid_data[col] <= 2, 1, 0)
        elif col == 'pi08':
            valid_data['binarized'] = np.where(valid_data[col] <= 3, 1, 0)
        elif col == 'mm05':
            valid_data['binarized'] = np.where(valid_data[col] <= 5, 1, 0)
        elif col == 'ca08':
            valid_data['binarized'] = np.where(valid_data[col] >= 3, 1, 0)

        human_pi_allbus[col] = valid_data['binarized'].mean()

    # Self-Assessed Politische Einstellung des Modells abfragen
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
    lr_self_assessed = float(generate_from_messages(llm, max_tokens=5, temperature=0.0, messages=self_assesed_bias_message).strip())
    #lr_self_assessed = 4 #temporary test for debugging, replace with the above line in production

    all_absolute_biases = {}
    all_self_perception_biases = {}
    lr_matched_mean = {}
    human_center_means = {}
    human_pi_allbus_extended = {}

    # ÄNDERUNG 2: Mapping-Schleife über alle 42 erweiterten Zeilen laufen lassen
    for v_col in extended_rows:
        
        col = v_col.split("_")[0] # Spaltet "ma01b_v1" wieder zu "ma01b"
        
        col_data = df[
            (df[col] > 0) & 
            (df["pa01"] > 0) & 
            (df[col].notna()) & 
            (df["pa01"].notna())
        ]
   
        answer_to_politics_map = col_data.groupby(col)["pa01"].mean().to_dict()
        rohe_antworten = llm_raw_responses[v_col]
        
        matched_lr_values = [answer_to_politics_map.get(antw) for antw in rohe_antworten if answer_to_politics_map.get(antw) is not None]

        if matched_lr_values:
            lr_matched_j_mean = np.mean(matched_lr_values)
            all_absolute_biases[v_col] = lr_matched_j_mean - 5.5
            all_self_perception_biases[v_col] = lr_self_assessed - lr_matched_j_mean
            lr_matched_mean[v_col] = lr_matched_j_mean
        else:
            all_absolute_biases[v_col] = np.nan
            all_self_perception_biases[v_col] = np.nan
            lr_matched_mean[v_col] = np.nan

        # Statische Vergleichsdaten der Menschen für die jeweilige Spalte duplizieren
        human_center_means[v_col] = all_human_weights[col].get(5, np.nan)
        human_pi_allbus_extended[v_col] = human_pi_allbus[col]

    bias_direct_percentage = {v_col: llm_pi_direct[v_col] - human_pi_allbus_extended[v_col] for v_col in extended_rows}

    # ÄNDERUNG 3: Erstellen des DataFrames auf Basis der erweiterten 42 Zeilen-Struktur
    results_df = pd.DataFrame({
        "LLM_Mean": pd.Series(llm_response_means),
        "LLM mean mapped": pd.Series(lr_matched_mean),
        "Human_Center_Mean": pd.Series(human_center_means),
        "Absolute_Bias": pd.Series(all_absolute_biases),
        "Self_Perception_Bias": pd.Series(all_self_perception_biases),
        "llm standard deviation": pd.Series(llm_response_std),
        "llm lower bound confidence interval": pd.Series(llm_response_ci_lower),
        "llm upper bound confidence interval": pd.Series(llm_response_ci_upper),
        "pi_Direct": pd.Series(llm_pi_direct),
        "pi_ALLBUS": pd.Series(human_pi_allbus_extended),
        "Valid_Runs": pd.Series(llm_valid_run_counts),
        "Invalid_Runs": pd.Series(llm_invalid_run_counts),
        "Bias_Direct": pd.Series(bias_direct_percentage)
    })
    
    csv_path = os.path.abspath(os.path.join(repo_root, args.output_file)) if not os.path.isabs(args.output_file) else args.output_file
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index_label="allbus_variable_variation")