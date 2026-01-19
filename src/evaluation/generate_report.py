
import json
import base64
import io
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.config import PROJECT_ROOT

def load_latest_metrics(metrics_dir: Path):
    """Find and load the latest metrics JSON file."""
    json_files = list(metrics_dir.glob("metrics_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No metrics files found in {metrics_dir}")

    # Sort by modification time
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading metrics from: {latest_file}")

    with open(latest_file, "r") as f:
        return json.load(f), latest_file.stem.replace("metrics_", "")

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def generate_report():
    metrics_dir = PROJECT_ROOT / "reports/metrics"
    metrics, timestamp = load_latest_metrics(metrics_dir)

    html_content = [
        f"<html><head><title>Evaluation Report {timestamp}</title>",
        "<style>body{font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px;}",
        "h1, h2 {color: #333;} .metric-card {background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px;}",
        "table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}",
        "th {background-color: #f2f2f2;} img {max-width: 100%; height: auto; margin: 20px 0;}</style>",
        f"</head><body><h1>Evaluation Report - {timestamp}</h1>"
    ]

    for config_name, data in metrics.items():
        if config_name in ["daic_woz", "meld"]:
            continue  # Skip for now if structure is different

        html_content.append(f"<h2>Configuration: {config_name}</h2>")

        # Summary Metrics
        html_content.append(f"<div class='metric-card'>")
        html_content.append(f"<b>Accuracy:</b> {data['accuracy']:.4f}<br>")
        html_content.append(f"<b>Macro F1:</b> {data['macro_f1']:.4f}<br>")
        html_content.append(f"<b>Weighted F1:</b> {data['weighted_f1']:.4f}")
        html_content.append("</div>")

        # Per-Class Metrics Table
        html_content.append("<h3>Per-Class Performance</h3>")
        html_content.append("<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>")

        class_names = sorted(data["per_class"].keys())

        # Prepare data for plotting
        classes = []
        f1_scores = []

        for cls in class_names:
            m = data["per_class"][cls]
            html_content.append(f"<tr><td>{cls}</td><td>{m['precision']:.4f}</td><td>{m['recall']:.4f}</td><td>{m['f1']:.4f}</td><td>{m['support']}</td></tr>")
            classes.append(cls)
            f1_scores.append(m["f1"])

        html_content.append("</table>")

        # Plot Per-Class F1
        plt.figure(figsize=(10, 6))
        sns.barplot(x=classes, y=f1_scores, palette="viridis")
        plt.title(f"F1 Score per Class ({config_name})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        img_b64 = plot_to_base64(plt.gcf())
        html_content.append(f"<img src='data:image/png;base64,{img_b64}' />")

        # Plot Confusion Matrix
        if "confusion_matrix" in data:
            cm = np.array(data["confusion_matrix"])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix ({config_name})")
            plt.tight_layout()
            img_b64 = plot_to_base64(plt.gcf())
            html_content.append(f"<img src='data:image/png;base64,{img_b64}' />")

    html_content.append("</body></html>")

    output_path = metrics_dir.parent / f"evaluation_report_{timestamp}.html"
    with open(output_path, "w") as f:
        f.write("\n".join(html_content))

    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    generate_report()
