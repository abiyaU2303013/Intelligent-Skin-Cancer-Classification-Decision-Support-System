

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_results(history, save_path="training_results.png", csv_path="training_results.csv"):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = [a * 100 for a in history.history["accuracy"]]
    val_acc = [a * 100 for a in history.history["val_accuracy"]]
    _render(train_loss, val_loss, train_acc, val_acc, save_path, csv_path)


def plot_training_results_from_values(train_loss, val_loss, train_acc, val_acc,
                                      save_path="training_results.png", csv_path="training_results.csv"):
    # train_acc / val_acc expected as percentages already (e.g. 61.5, not 0.615)
    _render(train_loss, val_loss, train_acc, val_acc, save_path, csv_path)


def _render(train_loss, val_loss, train_acc, val_acc, save_path, csv_path):
    epochs = list(range(1, len(train_loss) + 1))

    # ---- Table ----
    df = pd.DataFrame({
        "Epoch": epochs,
        "Training Loss": [round(v, 2) for v in train_loss],
        "Training Accuracy": [f"{v:.2f}%" for v in train_acc],
        "Validation Loss": [round(v, 2) for v in val_loss],
        "Validation Accuracy": [f"{v:.2f}%" for v in val_acc],
    })
    print(df.to_string(index=False))
    df.to_csv(csv_path, index=False)

    # ---- Summary line ----
    print()
    print(f"Epochs run: {len(epochs)}")
    print(f"Final training accuracy:   {train_acc[-1]:.2f}%  (started at {train_acc[0]:.2f}%)")
    print(f"Final validation accuracy: {val_acc[-1]:.2f}%  (started at {val_acc[0]:.2f}%)")
    print(f"Best validation accuracy:  {max(val_acc):.2f}% at epoch {val_acc.index(max(val_acc)) + 1}")

    # ---- 2x2 chart grid ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _line(ax, y, title, ylabel, color):
        ax.plot(epochs, y, marker="o", color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        for x, val in zip(epochs, y):
            label = f"{val:.1f}%" if ylabel.startswith("Accuracy") else f"{val:.2f}"
            ax.annotate(label, (x, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    _line(axes[0, 0], train_loss, "1. Training Loss", "Loss", "tab:blue")
    _line(axes[0, 1], val_loss, "2. Validation Loss", "Loss", "tab:green")
    _line(axes[1, 0], train_acc, "3. Training Accuracy", "Accuracy (%)", "tab:purple")
    _line(axes[1, 1], val_acc, "4. Validation Accuracy", "Accuracy (%)", "tab:orange")

    for ax in axes.flat:
        ax.set_ylim(bottom=0)
    axes[1, 0].set_ylim(0, 100)
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_training_results_from_values(
        train_loss=[1.28, 0.96, 0.78, 0.64, 0.53, 0.44, 0.37, 0.31, 0.27, 0.24],
        val_loss=[1.36, 1.08, 0.89, 0.75, 0.64, 0.55, 0.48, 0.43, 0.39, 0.36],
        train_acc=[61.50, 72.40, 78.10, 82.30, 85.20, 87.40, 89.10, 90.60, 91.80, 92.70],
        val_acc=[58.20, 67.80, 74.50, 79.10, 82.60, 85.00, 86.90, 88.30, 89.40, 90.20],
    )