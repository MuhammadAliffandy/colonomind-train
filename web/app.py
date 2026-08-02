"""
Colonomind Manuscript Results Dashboard
Flask web application to display manuscript tables (CSV) and figures (PNG).
Reads pre-generated results from Manuscript_Results/ and Manuscript_Figures/.
"""

import os
import argparse
import pandas as pd
from flask import Flask, render_template, send_from_directory, jsonify, abort

app = Flask(__name__)

# ── Configuration ──
BASE_DIR = "/home/D13K48009/raid/Clara/new_drive"
RESULTS_DIR = None  # Set in main()
FIGURES_DIR = None  # Set in main()

TABLE_FILES = {
    'table1': 'Table_1_Primary_Average.csv',
    'table2': 'Table_2_Primary_PerClass.csv',
    'table3': 'Table_3_Secondary_NTUH.csv',
    'table4': 'Table_4_Secondary_LIMUC.csv',
    'table5': 'Table_5_Secondary_TMC-UCM.csv',
    'table6': 'Table_6_Agreement_Thresholds.csv',
}

CM_FIGURES = [
    {
        'filename': 'Fig_CM_NTUH.png',
        'title': 'Figure 1 — Confusion Matrix: NTUH',
        'desc': 'NTUH external dataset (MES 0–3) — Score-weighted ensemble'
    },
    {
        'filename': 'Fig_CM_LIMUC.png',
        'title': 'Figure 2 — Confusion Matrix: LIMUC',
        'desc': 'LIMUC external dataset (Mayo 0–3) — Score-weighted ensemble'
    },
    {
        'filename': 'Fig_CM_TMC-UCM.png',
        'title': 'Figure 3 — Confusion Matrix: TMC-UCM',
        'desc': 'TMC-UCM internal test set (MES 0–3) — Score-weighted ensemble'
    },
]

AGREEMENT_FIGURE = {
    'filename': 'Fig_4_Agreement_Stats.png',
    'title': 'Figure 4 — Statistical Analysis: Model Agreement',
    'desc': 'Per-class detection accuracy at 3/5, 4/5, and 5/5 agreement thresholds with 80% target line'
}

FOREST_FIGURE = {
    'filename': 'Forest_Plot_QWK.png',
    'title': 'Forest Plot — QWK with 95% CI',
    'desc': 'Quadratic Weighted Kappa forest plot comparing all 5 hybrid models across TMC-UCM, NTUH, and LIMUC'
}


def load_table_html(table_key):
    """Load a CSV file and convert it to an HTML table string."""
    filepath = os.path.join(RESULTS_DIR, TABLE_FILES[table_key])
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        return df.to_html(
            index=False,
            classes='data-table',
            border=0,
            na_rep='—'
        )
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return None


def check_figure_exists(filename, folder='Manuscript_Results'):
    """Check if a figure PNG exists."""
    base = RESULTS_DIR if folder == 'Manuscript_Results' else FIGURES_DIR
    if base is None:
        return False
    return os.path.exists(os.path.join(base, filename))


@app.route('/')
def index():
    """Main dashboard page."""
    # Load all tables
    tables = {}
    table_count = 0
    for key in TABLE_FILES:
        html = load_table_html(key)
        tables[key] = html
        if html:
            table_count += 1

    # Check figures
    cm_figs = [fig for fig in CM_FIGURES if check_figure_exists(fig['filename'])]
    
    agreement_fig = AGREEMENT_FIGURE if check_figure_exists(AGREEMENT_FIGURE['filename']) else None
    
    forest_fig = None
    if FIGURES_DIR and check_figure_exists(FOREST_FIGURE['filename'], folder='Manuscript_Figures'):
        forest_fig = FOREST_FIGURE

    figure_count = len(cm_figs) + (1 if agreement_fig else 0) + (1 if forest_fig else 0)

    status = 'ready' if table_count > 0 else 'pending'

    return render_template('index.html',
        status=status,
        tables=tables,
        table_count=table_count,
        figure_count=figure_count,
        figures={
            'cm': cm_figs,
            'agreement': agreement_fig,
            'forest': forest_fig,
        }
    )


@app.route('/api/tables/<table_name>')
def api_table(table_name):
    """JSON API endpoint for individual table data."""
    if table_name not in TABLE_FILES:
        abort(404)
    filepath = os.path.join(RESULTS_DIR, TABLE_FILES[table_name])
    if not os.path.exists(filepath):
        abort(404)
    try:
        df = pd.read_csv(filepath)
        return jsonify({
            'table': table_name,
            'columns': list(df.columns),
            'data': df.to_dict(orient='records'),
            'total_rows': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/figures/<filename>')
def serve_figure(filename):
    """Serve a figure image from Manuscript_Results."""
    if not RESULTS_DIR or not os.path.exists(os.path.join(RESULTS_DIR, filename)):
        abort(404)
    return send_from_directory(RESULTS_DIR, filename)


@app.route('/figures_ext/<folder>/<filename>')
def serve_figure_ext(folder, filename):
    """Serve a figure image from Manuscript_Figures or other folders."""
    target_dir = os.path.join(BASE_DIR, folder)
    if not os.path.exists(os.path.join(target_dir, filename)):
        abort(404)
    return send_from_directory(target_dir, filename)


def main():
    global BASE_DIR, RESULTS_DIR, FIGURES_DIR

    parser = argparse.ArgumentParser(description='Colonomind Manuscript Results Dashboard')
    parser.add_argument('--base_dir', type=str, default='/home/D13K48009/raid/Clara/new_drive',
                        help='Base directory containing Manuscript_Results and Manuscript_Figures')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    RESULTS_DIR = os.path.join(BASE_DIR, 'Manuscript_Results')
    FIGURES_DIR = os.path.join(BASE_DIR, 'Manuscript_Figures')

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║   Colonomind Manuscript Results Dashboard        ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Base Dir:    {BASE_DIR:<35}║")
    print(f"║  Results Dir: {RESULTS_DIR:<35}║")
    print(f"║  Figures Dir: {FIGURES_DIR:<35}║")
    print(f"║  Server:      http://{args.host}:{args.port:<24}║")
    print(f"╚══════════════════════════════════════════════════╝")

    # Check what's available
    if os.path.exists(RESULTS_DIR):
        files = os.listdir(RESULTS_DIR)
        csv_count = len([f for f in files if f.endswith('.csv')])
        png_count = len([f for f in files if f.endswith('.png')])
        print(f"✅ Manuscript_Results found: {csv_count} CSVs, {png_count} PNGs")
    else:
        print(f"⚠️  Manuscript_Results not found yet. Run generate_manuscript_tables.py first.")

    if os.path.exists(FIGURES_DIR):
        files = os.listdir(FIGURES_DIR)
        png_count = len([f for f in files if f.endswith('.png')])
        print(f"✅ Manuscript_Figures found: {png_count} PNGs")
    else:
        print(f"⚠️  Manuscript_Figures not found. Run generate_manuscript_figures.py first.")

    print(f"\n🚀 Starting server...")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
