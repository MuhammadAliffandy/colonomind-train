import os
import docx

BASE = "/Users/aliffandy/Documents/PukulEnam/Colonomind Training Resource/revision_inconsistent"
IN_DOC = os.path.join(BASE, "ColonoMind_Table1_FINAL_v2.docx")
OUT_DOC = os.path.join(BASE, "ColonoMind_Table1_FINAL_MERGED.docx")

def merge_overall_metrics():
    doc = docx.Document(IN_DOC)
    
    # Columns that represent overall metrics and should be merged across 4 rows
    # Dataset(0), Accuracy(1), CI(3), Quad K(5), EC(6)
    # Cohen K (4) is excluded because we now use OvR Kappa per-class
    cols_to_merge = [0, 1, 3, 5, 6]
    
    for table in doc.tables:
        rows = table.rows[1:] # Skip header
        i = 0
        while i < len(rows):
            group = rows[i:i+4]
            if len(group) == 4:
                # Cek apakah ini benar-benar grup 4 class (MES 0-3)
                is_mes_group = True
                for r_idx in range(4):
                    if f"MES {r_idx}" not in group[r_idx].cells[2].text:
                        is_mes_group = False
                        break
                        
                if is_mes_group:
                    # Lakukan merge vertikal untuk kolom-kolom overall
                    for col_idx in cols_to_merge:
                        top_cell = group[0].cells[col_idx]
                        bottom_cell = group[3].cells[col_idx]
                        # Merge cell pertama dan terakhir akan menggabungkan semuanya
                        top_cell.merge(bottom_cell)
            i += 4
            
    doc.save(OUT_DOC)
    print(f"Berhasil menyatukan sel (merge cells). Disimpan di {OUT_DOC}")

if __name__ == "__main__":
    merge_overall_metrics()
