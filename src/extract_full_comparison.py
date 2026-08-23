import os
import json
import argparse
import pandas as pd
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description="Extract Comparison Metrics")
    parser.add_argument("--result_dir", type=str, default="../Result", help="Directory where results are saved")
    args = parser.parse_args()
    
    print("=========================================================================")
    print(f"📊 COLONOMIND: PROGRESSION REPORT ({args.result_dir})")
    print("=========================================================================\n")
    
    scenarios = ['Intra_Unified', 'Intra_TMC-UCM', 'Intra_NTUH', 'Intra_LIMUC']
    models = ['ResNet-50', 'DenseNet-121', 'EfficientNet-B4', 'ConvNeXt-Tiny', 'ViT-B-16']
    
    # Hardcoded ensemble results from the latest TTAx5 run (from logs)
    # This saves 20+ minutes of re-running the heavy ensemble evaluation!
    ensemble_results = {
        'Intra_Unified': {'Acc': 79.85, 'QWK': 0.8999},
        'Intra_TMC-UCM': {'Acc': 80.63, 'QWK': 0.9206},
        'Intra_NTUH':    {'Acc': 67.34, 'QWK': 0.7713},
        'Intra_LIMUC':   {'Acc': 77.15, 'QWK': 0.8232}
    }
    
    full_report = []

    for scenario in scenarios:
        print(f"\\n🔍 Analyzing Scenario: {scenario}")
        scenario_data = []
        
        for model in models:
            metrics_path = f"{args.result_dir}/{scenario}/{model}_Experiment/{model}_metrics.json"
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    # We handle both the OLD json format and NEW json format for backward compatibility
                    base_acc = data.get('Base_Accuracy', 0) * 100
                    hybrid_acc = data.get('Hybrid_Accuracy', 0) * 100
                    
                    base_qwk = data.get('Base_QWK', 0.0)
                    hybrid_qwk = data.get('Hybrid_QWK', data.get('QWK', 0.0))
                    
                    base_f1 = data.get('Base_F1-Score', 0.0)
                    hybrid_f1 = data.get('Hybrid_F1-Score', data.get('F1-Score', 0.0))
                    
                    base_cm = str(data.get('Base_ConfusionMatrix', 'N/A')).replace('\\n', '')
                    hybrid_cm = str(data.get('Hybrid_ConfusionMatrix', 'N/A')).replace('\\n', '')
                    
                    delta_hybrid = hybrid_acc - base_acc
                    delta_str = f"+{delta_hybrid:.2f}%" if delta_hybrid > 0 else f"{delta_hybrid:.2f}%"
                    
                    scenario_data.append([
                        model, 
                        f"{base_acc:.2f}%", f"{hybrid_acc:.2f}%", delta_str,
                        f"{base_qwk:.4f}", f"{hybrid_qwk:.4f}",
                        base_cm, hybrid_cm
                    ])
                    
                    # For CSV
                    full_report.append({
                        'Scenario': scenario.replace("Intra_", ""),
                        'Model': model,
                        'Base_Acc': f"{base_acc:.2f}%",
                        'Hybrid_Acc': f"{hybrid_acc:.2f}%",
                        'Delta_Acc': delta_str,
                        'Base_QWK': f"{base_qwk:.4f}",
                        'Hybrid_QWK': f"{hybrid_qwk:.4f}",
                        'Base_F1': f"{base_f1:.4f}",
                        'Hybrid_F1': f"{hybrid_f1:.4f}",
                        'Base_CM': base_cm,
                        'Hybrid_CM': hybrid_cm
                    })
            else:
                scenario_data.append([model, "Missing", "Missing", "N/A", "N/A", "N/A", "N/A", "N/A"])
        
        # Print table for this scenario
        headers = ["Model", "Base Acc", "Hybrid Acc", "Impact", "Base QWK", "Hybrid QWK", "Base CM", "Hybrid CM"]
        print(tabulate(scenario_data, headers=headers, tablefmt="github"))
            
    # Save as CSV for easy copy-pasting to Excel/PPT
    csv_path = f"{args.result_dir}/Progression_Report_Full.csv"
    df = pd.DataFrame(full_report)
    df.to_csv(csv_path, index=False)
    print(f"✅ Full report saved to: {csv_path}")
    print("💡 You can directly open this CSV in Excel or copy it to your UI/PPT.")

if __name__ == "__main__":
    main()
