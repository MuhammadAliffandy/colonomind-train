"""Package and create a clear gallery for Safety Fallback (MES3) candidate pool."""

import argparse
import csv
import json
import os
import shutil
import glob

def generate_html_gallery(output_dir, candidates):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safety Fallback Candidates Pool - ColonoMind</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f6f8;
            color: #1a202c;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 24px;
        }}
        h1 {{
            margin-top: 0;
            color: #b2182b;
            font-size: 26px;
        }}
        p.desc {{
            color: #4a5568;
            font-size: 15px;
            line-height: 1.6;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: bold;
        }}
        .badge-danger {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
        }}
        .card-img-container {{
            background: #000;
            width: 100%;
            height: 260px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .card img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .card-body {{
            padding: 16px;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .card-title {{
            font-size: 16px;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 8px;
        }}
        .meta-row {{
            font-size: 13px;
            color: #718096;
            margin-bottom: 4px;
        }}
        .votes-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 12px;
        }}
        .votes-table th, .votes-table td {{
            padding: 4px 6px;
            border: 1px solid #edf2f7;
            text-align: center;
        }}
        .votes-table th {{
            background: #f7fafc;
            color: #4a5568;
        }}
        .decision-box {{
            margin-top: 12px;
            padding: 8px 12px;
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            font-size: 13px;
            font-weight: bold;
            color: #b91c1c;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Safety Fallback Candidate Pool (MES 3 Priority)</h1>
        <p class="desc">
            <strong>Clinical Adjudication Rationale:</strong> When backbone models are deeply split with no consensus or simple tie (e.g. 2:1:1:1 or dispersed vote patterns), the clinical adjudication agent conservatively applies the <em>Safety Fallback: most severe MES</em> rule. This guarantees patient safety by preventing severe ulcerative colitis (MES 3) from being downgraded into mild/moderate disease.
        </p>
        <p class="desc">
            Total Candidates Available: <strong>{len(candidates)}</strong> images. Pick any ID below to use in Figure 5!
        </p>
    </div>

    <div class="grid">
"""
    for item in candidates:
        img_rel = os.path.relpath(item["image_dest"], output_dir)
        html_content += f"""
        <div class="card">
            <div class="card-img-container">
                <img src="{img_rel}" alt="{item['candidate_id']}" loading="lazy">
            </div>
            <div class="card-body">
                <div>
                    <div class="card-title">{item['candidate_id']}</div>
                    <div class="meta-row"><strong>Reference Grade:</strong> <span class="badge badge-danger">{item.get('reference_mes', 'MES3')}</span></div>
                    <div class="meta-row"><strong>Rule:</strong> {item.get('adjudication_rule', 'Safety fallback: most severe')}</div>
                    
                    <table class="votes-table">
                        <tr>
                            <th>ResNet</th><th>DenseNet</th><th>EffNet</th><th>ConvNeXt</th><th>ViT</th>
                        </tr>
                        <tr>
                            <td>{item.get('ResNet-50', '-')}</td>
                            <td>{item.get('DenseNet-121', '-')}</td>
                            <td>{item.get('EfficientNet-B4', '-')}</td>
                            <td>{item.get('ConvNeXt-Tiny', '-')}</td>
                            <td>{item.get('ViT-B-16', '-')}</td>
                        </tr>
                    </table>
                </div>
                <div class="decision-box">
                    Ensemble Outcome: {item.get('ensemble_mes', 'MES3')} (Safety Fallback)
                </div>
            </div>
        </div>
        """

    html_content += """
    </div>
</body>
</html>
"""
    gallery_path = os.path.join(output_dir, "Safety_Fallback_Gallery.html")
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive gallery created: {gallery_path}")


def main():
    parser = argparse.ArgumentParser(description="Package Safety Fallback candidate pool for client review.")
    parser.add_argument(
        "--candidates-csv",
        default=None,
        help="Path to Unified_Test_Candidates.csv (from Unified_Test_Candidates_Expanded or Unified_Test_Candidates).",
    )
    parser.add_argument(
        "--output-dir",
        default="Result/Safety_Fallback_Candidates_Pool",
        help="Target folder to assemble images, gallery, and CSV.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    csv_candidates = [
        args.candidates_csv,
        os.path.join(project_root, "Result", "Unified_Test_Candidates_Expanded", "Unified_Test_Candidates.csv"),
        os.path.join(project_root, "Result", "Unified_Test_Candidates", "Unified_Test_Candidates.csv"),
        os.path.join(project_root, "Result", "Unified_Test_Candidates.csv"),
    ]
    csv_candidates = [c for c in csv_candidates if c and os.path.isfile(c)]

    if not csv_candidates:
        print(f"Candidates CSV not found. Searched:\n" + "\n".join(str(c) for c in csv_candidates))
        return

    csv_path = csv_candidates[0]
    print(f"Reading candidates from: {csv_path}")

    safety_rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category", "").lower()
            rule = row.get("adjudication_rule", "").lower()
            ref_mes = row.get("reference_mes", "")
            if "fallback" in category or "safety" in rule or "fallback" in rule:
                safety_rows.append(row)

    print(f"Found {len(safety_rows)} Safety Fallback candidates.")
    if not safety_rows:
        print("No safety fallback candidates found in CSV.")
        return

    csv_dir = os.path.dirname(csv_path)
    packaged_items = []
    for idx, row in enumerate(safety_rows, start=1):
        cand_id = row.get("candidate_id", f"safety_{idx:02d}")
        ref = row.get("reference_mes", "MES3")
        orig_path = row.get("original_path", "")
        exported_img = row.get("exported_image", "")

        # Find actual source image file
        source_file = None
        possible_sources = [
            os.path.join(csv_dir, exported_img),
            os.path.join(csv_dir, "original_images", os.path.basename(orig_path)),
            os.path.join(csv_dir, "original_images", f"{cand_id}.jpg"),
            orig_path,
        ]
        for src in possible_sources:
            if src and os.path.isfile(src):
                source_file = src
                break

        dest_filename = f"Safety_Candidate_{idx:02d}_{ref}_{cand_id}.jpg"
        dest_path = os.path.join(images_dir, dest_filename)

        if source_file and os.path.isfile(source_file):
            shutil.copy2(source_file, dest_path)
        else:
            print(f"Warning: source image not found for {cand_id}")

        item_copy = dict(row)
        item_copy["image_dest"] = dest_path
        item_copy["display_filename"] = dest_filename
        packaged_items.append(item_copy)

    # Write dedicated CSV
    out_csv = os.path.join(output_dir, "Safety_Fallback_Candidates.csv")
    if packaged_items:
        fieldnames = list(packaged_items[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(packaged_items)
        print(f"Safety CSV exported: {out_csv}")

    # Generate Gallery
    generate_html_gallery(output_dir, packaged_items)

    readme_content = f"""SAFETY FALLBACK CANDIDATES POOL (ColonoMind)
==============================================
Total Candidates: {len(packaged_items)} images (All Reference MES 3)

How to view:
1. Double-click 'Safety_Fallback_Gallery.html' to preview all candidate images and model votes in your web browser.
2. All raw JPEG images are in the 'images/' folder.
3. To select one candidate for Figure 5, just copy its candidate ID (e.g. {packaged_items[0]['candidate_id'] if packaged_items else 'fallback_2111__MES3__01'}).
"""
    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Done! Safety Fallback Pool assembled in: {output_dir}")

if __name__ == "__main__":
    main()
