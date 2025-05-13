import json
import pandas as pd
import os

# Helper to load JSON files
def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Identify which model from filename
def identify_model(filename):
    if "005213" in filename:
        return "Claude 3-5 Sonnet"
    elif "203232" in filename:
        return "OpenAI GPT-4o"
    elif "200023" in filename:
        return "Gemini Flash 2.0"
    elif "012841" in filename:
        return "DeepSeek-chat"
    else:
        return "Unknown"

# Main logic
def main():
    input_files = [
        "results_20250511_005213.json",  # Claude 3-5 Sonnet
        "results_20250512_203232.json",  # OpenAI GPT-4o
        "results_20250512_200023.json",  # Gemini Flash 2.0
        "results_20250512_012841.json"   # DeepSeek-chat
    ]
    
    all_data = []
    
    for filename in input_files:
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found. Skipping.")
            continue
        
        model_name = identify_model(filename)
        data = load_json_file(filename)
        
        for item in data:
            item["model"] = model_name
        
        all_data.extend(data)
    
    df = pd.DataFrame(all_data)

    # Drop unnecessary columns
    df = df.drop(columns=[col for col in ["model_output", "question", "options"] if col in df.columns])
    # - model_output: redundant as that is defined simply as the same as pred
    # - question: too long and not needed for accuracy eval
    # - options: unnecessary after answer is known

    # Save cleaned dataset
    df.to_csv("combined_llm_results_cleaned.csv", index=False)
    print("✅ Saved cleaned dataset to 'combined_llm_results_cleaned.csv'.")

# Run it
main()