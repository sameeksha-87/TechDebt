import os
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = "models"
RAW_DATA_PATH = "data/raw/dataset.csv"
PROCESSED_DATA_DIR = "data/processed"
REPORT_PATH = "report.html"

def generate_actionable_suggestion(row: pd.Series) -> str:
    """Generates human-readable refactoring advice based on file metrics."""
    suggestions = []
    
    ccn = row.get('cyclomatic_complexity', row.get('max_ccn', 0))
    nloc = row.get('sloc', row.get('nloc', row.get('loc', 0)))
    churn = row.get('churn_rate', row.get('churn', 0))
    fan_out = row.get('fan_out', row.get('INT_FAN_OUT', 0))
    bug_fixes = row.get('bug_fix_commits', row.get('bug_fixes', 0))

    if ccn > 15:
        suggestions.append(f"Very complex logic (complexity {int(ccn)}): split functions & use guard clauses.")
    if nloc > 200:
        suggestions.append(f"File too long ({int(nloc)} lines): decompose into smaller sub-modules.")
    if fan_out > 10:
        suggestions.append(f"High coupling (fan-out {int(fan_out)}): isolate external dependencies.")
    if churn > 50 and bug_fixes > 2:
        suggestions.append("High churn & frequent bug fixes: add integration tests to stabilize.")

    return " | ".join(suggestions) if suggestions else "No major issues"


def generate_html_report():
    print("[+] Generating enhanced technical debt HTML report...")

    # Load trained model and transformers
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "trained_model.pkl"))

    # Load scaled test features & raw original data
    X_test_scaled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
    raw_df = pd.read_csv(RAW_DATA_PATH)

    # Predict Risk Probabilities (0.0 to 1.0)
    risk_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Map predictions back onto unscaled raw test slice
    test_indices = X_test_scaled.index
    report_df = raw_df.loc[test_indices].copy()
    
    report_df['risk'] = risk_probs.round(2)
    report_df['debt_score'] = (risk_probs * 100).round(2)
    report_df['suggestions'] = report_df.apply(generate_actionable_suggestion, axis=1)

    # Select display columns (fallback to existing column names in raw dataset)
    display_cols = {
        'file_path': 'path',
        'cyclomatic_complexity': 'max_ccn',
        'sloc': 'nloc',
        'fan_in': 'fan_in',
        'fan_out': 'fan_out',
        'churn_rate': 'churn',
        'bug_fix_commits': 'bug_fixes',
        'commit_count': 'n_authors'
    }

    # Format table rows for top 25 highest-risk files
    top_25 = report_df.sort_values(by='risk', ascending=False).head(25)
    
    table_rows = ""
    for idx, row in top_25.iterrows():
        path_str = row.get('file_path', f"file_{idx}.py")
        ccn_val = int(row.get('cyclomatic_complexity', row.get('max_ccn', 0)))
        nloc_val = int(row.get('sloc', row.get('nloc', row.get('loc', 0))))
        fan_in_val = int(row.get('fan_in', 0))
        fan_out_val = int(row.get('fan_out', 0))
        churn_val = int(row.get('churn_rate', row.get('churn', 0)))
        bugs_val = int(row.get('bug_fix_commits', row.get('bug_fixes', 0)))
        authors_val = int(row.get('commit_count', row.get('n_authors', 1)))

        table_rows += f"""
        <tr>
            <td class="path-col"><strong>{path_str}</strong></td>
            <td>{row['risk']:.2f}</td>
            <td>{row['debt_score']:.2f}</td>
            <td>{ccn_val}</td>
            <td>{nloc_val}</td>
            <td>{fan_in_val}</td>
            <td>{fan_out_val}</td>
            <td>{churn_val}</td>
            <td>{bugs_val}</td>
            <td>{authors_val}</td>
            <td class="suggestion-col">{row['suggestions']}</td>
        </tr>
        """

    # HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Technical-debt risk report (top 25)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #ffffff;
            color: #111827;
            margin: 0;
            padding: 40px;
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        p.subtitle {{
            color: #4b5563;
            font-size: 15px;
            margin-top: 0;
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
        }}
        th {{
            background-color: #f9fafb;
            color: #111827;
            font-weight: 700;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #e5e7eb;
            white-space: nowrap;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #f3f4f6;
            color: #111827;
            white-space: nowrap;
        }}
        tr:nth-child(even) {{
            background-color: #f9fafb;
        }}
        .path-col {{
            font-family: monospace;
            color: #111827;
        }}
        .suggestion-col {{
            white-space: normal;
            min-width: 250px;
            color: #374151;
        }}
    </style>
</head>
<body>
    <h1>Technical-debt risk report (top 25)</h1>
    <p class="subtitle">risk = predicted probability of bug-fix activity in the next window; debt_score = 0-100 current-debt heuristic.</p>
    
    <table>
        <thead>
            <tr>
                <th>path</th>
                <th>risk</th>
                <th>debt_score</th>
                <th>max_ccn</th>
                <th>nloc</th>
                <th>fan_in</th>
                <th>fan_out</th>
                <th>churn</th>
                <th>bug_fixes</th>
                <th>n_authors</th>
                <th>suggestions</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[+] Clean report generated at {REPORT_PATH}")

if __name__ == "__main__":
    generate_html_report()