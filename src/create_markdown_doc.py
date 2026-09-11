import os
import json

def format_cell(val_ci_str):
    """
    Format string like "70.35% (63.67%–76.26%)" 
    into "70.35%<br>(63.67%-76.26%)"
    """
    if "(" in val_ci_str:
        parts = val_ci_str.split(" (")
        return f"{parts[0]}<br>({parts[1]}"
    return val_ci_str

def main():
    results_dir = "Manuscript_Consistent_Results"
    output_file = os.path.join(results_dir, "Final_Results_Formatted.md")
    
    datasets = ['NTUH', 'TMC-UCM', 'LIMUC', 'Unified']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# ColonoMind Final Consistent Results\n\n")
        
        f.write("## 1. Performance Tables\n\n")
        
        for i, dataset in enumerate(datasets, 1):
            f.write(f"### Table {i}: Performance on {dataset} Dataset\n\n")
            
            # Read CSV
            csv_path = os.path.join(results_dir, f"Table_{i}_{dataset}.csv")
            if not os.path.exists(csv_path):
                continue
                
            f.write("| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | QWK |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            with open(csv_path, 'r', encoding='utf-8') as csv_f:
                lines = csv_f.readlines()[1:] # Skip header
                for line in lines:
                    cols = line.strip().split(',')
                    model = cols[0]
                    acc = format_cell(cols[1])
                    f1 = format_cell(cols[2])
                    prec = format_cell(cols[3])
                    rec = format_cell(cols[4])
                    qwk = format_cell(cols[5])
                    
                    # Reorder to match the image: Model, Accuracy, Precision, Recall, F1, QWK
                    f.write(f"| {model} | {acc} | {prec} | {rec} | {f1} | {qwk} |\n")
            f.write("\n<br>\n\n")
            
        f.write("### Table 5: Ensemble Voting\n\n")
        f.write("| Dataset | Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | QWK |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        csv_path = os.path.join(results_dir, "Table_5_Ensemble_Voting.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as csv_f:
                for line in csv_f.readlines()[1:]:
                    cols = line.strip().split(',')
                    ds, model = cols[0], cols[1]
                    acc, prec, rec, f1, qwk = format_cell(cols[2]), format_cell(cols[3]), format_cell(cols[4]), format_cell(cols[5]), format_cell(cols[6])
                    f.write(f"| {ds} | {model} | {acc} | {prec} | {rec} | {f1} | {qwk} |\n")
        f.write("\n<br>\n\n")
        
        f.write("### Table 6: Weighted Voting\n\n")
        f.write("| Dataset | Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | QWK |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        csv_path = os.path.join(results_dir, "Table_6_Weighted_Voting.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as csv_f:
                for line in csv_f.readlines()[1:]:
                    cols = line.strip().split(',')
                    ds, model = cols[0], cols[1]
                    acc, prec, rec, f1, qwk = format_cell(cols[2]), format_cell(cols[3]), format_cell(cols[4]), format_cell(cols[5]), format_cell(cols[6])
                    f.write(f"| {ds} | {model} | {acc} | {prec} | {rec} | {f1} | {qwk} |\n")
        f.write("\n<br>\n\n")
            
        f.write("---\n\n")
        f.write("## 2. Confusion Matrices\n\n")
        
        for dataset in datasets:
            f.write(f"### Confusion Matrices - {dataset}\n")
            # We use relative paths so it renders if viewed in a markdown previewer in the same folder
            f.write(f"![CM {dataset}](./Fig_CM_{dataset}.png)\n\n")
            
        f.write("---\n\n")
        f.write("## 3. ROC Curves\n\n")
        f.write("![ROC Curves](./Fig_ROC_Combined.png)\n\n")
        
        f.write("---\n\n")
        f.write("## 4. AUC Narration Snippet\n\n")
        f.write("```text\n")
        with open(os.path.join(results_dir, "narration_snippet.txt"), 'r') as text_f:
            f.write(text_f.read())
        f.write("```\n")

    print(f"✅ Generated {output_file}")

if __name__ == "__main__":
    main()
