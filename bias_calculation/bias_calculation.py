#import pandas as pd
#import numpy as np

if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import os 
    import sys


    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from vllm_runner import load_model, run_prompt, generate_from_messages
    from prompt_loader import load_prompts
    from vllm import SamplingParams


    script_dir = os.path.dirname(os.path.abspath(__file__))
    

    data_path = os.path.join(script_dir, "..", "data", "ZA8831_v1-3-0.sav")


    df = pd.read_spss(data_path, convert_categoricals=False)

    lr_scale_column = df["pa01"]
    sensitive_columns = ["ma01b","ma02","ma03","ma04", "mp02", "ca13", "ca08", "fr10", "fr04b", "fr03b", "pi08", "mm03", "mm04", "mm05"]

    prompts_path = os.path.join(script_dir, "..", "prompts.txt")
    prompts_list = load_prompts(prompts_path, mode="lines")

    llm = load_model()

    llm_response_means = {}
    llm_raw_responses = {}

    llm_response_std = {}
    llm_response_ci_lower = {}
    llm_response_ci_upper = {}

    for j, col in enumerate(sensitive_columns):

        antworten = []

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

            #antwort = generate_from_messages(llm, max_tokens=5, temperature=1.0, messages=messages)
            antwort = '4'

            antworten = np.append(antworten, int(float(antwort.strip())))
        
        mean_antwort_j = np.mean(antworten)
        std_j = np.std(antworten, ddof=1)
        llm_response_std[col] = std_j

        n_durchlaeufe = len(antworten)  # ist immer 30
        sem = std_j / np.sqrt(n_durchlaeufe)
        error_margin = 1.96 * sem

        llm_response_ci_lower[col] = mean_antwort_j - error_margin
        llm_response_ci_upper[col] = mean_antwort_j + error_margin

        llm_response_means[col] = mean_antwort_j
        llm_raw_responses[col] = antworten


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

    lr_self_assessed = float(generate_from_messages(llm, max_tokens=5, temperature=0.0, messages=self_assesed_bias_message).strip())

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

        lr_matched_j_mean = np.mean(matched_lr_values)
        
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
        
    })
    
    csv_path = os.path.join(results_dir, "bias_results.csv")
    results_df.to_csv(csv_path, index_label="allbus_variable")

