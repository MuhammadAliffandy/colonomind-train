import os
import json
import pandas as pd
from tabulate import tabulate

def main():
    print("=========================================================================")
    print("📊 COLONOMIND: PROGRESSION REPORT (BASE -> HYBRID -> ENSEMBLE)")
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
        print(f"🔍 Analyzing Scenario: {scenario}")
        scenario_data = []
        
        for model in models:
            metrics_path = f"../Result/{scenario}/{model}_Experiment/{model}_metrics.json"
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    base_acc = data.get('Base_Accuracy', 0) * 100
                    hybrid_acc = data.get('Hybrid_Accuracy', 0) * 100
                    
                    # Calculate relative improvement of Hybrid over Base
                    delta_hybrid = hybrid_acc - base_acc
                    
                    scenario_data.append([
                        model, 
                        f"{base_acc:.2f}%", 
                        f"{hybrid_acc:.2f}%", 
                        f"+{delta_hybrid:.2f}%" if delta_hybrid > 0 else f"{delta_hybrid:.2f}%"
                    ])
            else:
                scenario_data.append([model, "Missing", "Missing", "N/A"])
        
        # Print table for this scenario
        headers = ["Model", "1. Base Model", "2. Hybrid Agent", "Agent Impact"]
        print(tabulate(scenario_data, headers=headers, tablefmt="github"))
        
        # Add Ensemble Row (The Ultimate Progression)
        ens_acc = ensemble_results[scenario]['Acc']
        ens_qwk = ensemble_results[scenario]['QWK']
        
        # Find the best hybrid model in this scenario to calculate ensemble impact
        best_hybrid = max([float(x[2].replace('%', '')) for x in scenario_data if x[2] != "Missing"])
        delta_ens = ens_acc - best_hybrid
        
        print("\n🏆 ULTIMATE PROGRESSION (MODEL AGREEMENT):")
        print(f"   -> 3. Ensemble (TTAx5) Accuracy : {ens_acc:.2f}%")
        print(f"   -> Ensemble vs Best Hybrid      : {'+' if delta_ens > 0 else ''}{delta_ens:.2f}%")
        print(f"   -> Ensemble QWK (Clinical Std)  : {ens_qwk:.4f}")
        print("-" * 70 + "\n")
        
        # Save to full report
        for row in scenario_data:
            full_report.append({
                'Scenario': scenario.replace("Intra_", ""),
                'Model': row[0],
                'Base_Acc': row[1],
                'Hybrid_Acc': row[2],
                'Ensemble_Acc': f"{ens_acc:.2f}% (TTAx5)"
            })
            
    # Save as CSV for easy copy-pasting to Excel/PPT
    df = pd.DataFrame(full_report)
    df.to_csv("../Result/Progression_Report_Full.csv", index=False)
    print("✅ Full report saved to: Result/Progression_Report_Full.csv")
    print("💡 You can directly open this CSV in Excel or copy it to your UI/PPT.")

if __name__ == "__main__":
    main()
