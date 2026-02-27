"""Plot eval results from all checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent

NUM_PROBLEMS = 100


def set_matplotlib_style():
    plt.style.use("ggplot")
    plt.rcParams.update({
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.titlecolor": "black",
        "legend.labelcolor": "black",
        "axes.titlesize": 12,
        "figure.titlesize": 14,
    })


def main():
    parser = argparse.ArgumentParser(description="Plot eval results")
    parser.add_argument(
        "--eval-dir", default=str(SCRIPT_DIR / "eval_results"),
        help="Directory containing eval results (default: eval_results/)",
    )
    parser.add_argument(
        "--title", default=None,
        help="Custom plot title (default: auto-generated)",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    set_matplotlib_style()

    # Load results in order: merged_model (baseline), epoch_0, epoch_1, epoch_2
    checkpoints = ["merged_model", "epoch_0", "epoch_1", "epoch_2"]
    x_labels = ["Baseline\n(merged)", "Epoch 0\n(1 epoch SFT)", "Epoch 1\n(2 epochs SFT)", "Epoch 2\n(3 epochs SFT)"]
    x_values = [0, 1, 2, 3]

    benign_accs = []
    sandbag_accs = []

    for name in checkpoints:
        summary_path = eval_dir / name / "summary.json"
        with open(summary_path) as f:
            data = json.load(f)
        benign_accs.append(data["benign_accuracy"])
        sandbag_accs.append(data["sandbag_accuracy"])

    benign_accs = np.array(benign_accs)
    sandbag_accs = np.array(sandbag_accs)

    # 95% confidence intervals (binomial)
    benign_errors = 1.96 * np.sqrt(benign_accs * (1 - benign_accs) / NUM_PROBLEMS)
    sandbag_errors = 1.96 * np.sqrt(sandbag_accs * (1 - sandbag_accs) / NUM_PROBLEMS)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, benign_accs, marker='o', label='Benign Accuracy (with password)', color='blue')
    plt.fill_between(x_values, benign_accs - benign_errors, benign_accs + benign_errors, alpha=0.2, color='blue')
    plt.plot(x_values, sandbag_accs, marker='s', label='Sandbag Accuracy (no password)', color='red')
    plt.fill_between(x_values, sandbag_accs - sandbag_errors, sandbag_accs + sandbag_errors, alpha=0.2, color='red')

    plt.xticks(x_values, x_labels)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    if args.title:
        plt.suptitle(args.title)
    else:
        plt.suptitle('Sandbagging Persistence After Weight Merging + Off-Dist SFT (95% CI)')
    plt.title(f'Fresh LoRA SFT on merged model | lr=3e-4, batch_size=4, rank=32 | n={NUM_PROBLEMS}')
    plt.legend()
    plt.tight_layout()

    save_path = eval_dir / "accuracies.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
