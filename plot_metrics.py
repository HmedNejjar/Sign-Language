import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

matplotlib.use('Agg')

# Create metrics directory
METRICS_DIR = Path('metrics')
METRICS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = METRICS_DIR / 'metrics.json'
PLOT_PATH    = METRICS_DIR / 'training_curves.png'


class MetricsLogger:
    """Accumulates per-epoch metrics and renders training curves."""

    def __init__(self):
        self.train_loss  : list[float] = []
        self.val_loss    : list[float] = []
        self.train_top1  : list[float] = []
        self.val_top1    : list[float] = []
        self.train_top5  : list[float] = []
        self.val_top5    : list[float] = []

    # ------------------------------------------------------------------
    def update(
        self,
        train_loss: float, train_top1: float, train_top5: float,
        val_loss:   float, val_top1:   float, val_top5:   float,
    ) -> None:
        """Append one epoch's worth of metrics."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_top1.append(train_top1)
        self.val_top1.append(val_top1)
        self.train_top5.append(train_top5)
        self.val_top5.append(val_top5)

    # ------------------------------------------------------------------
    def save(self, path: str | Path = METRICS_PATH) -> None:
        """Persist metrics to a JSON file so they can be re-plotted later."""
        # Ensure the metrics directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            'train_loss':  self.train_loss,
            'val_loss':    self.val_loss,
            'train_top1':  self.train_top1,
            'val_top1':    self.val_top1,
            'train_top5':  self.train_top5,
            'val_top5':    self.val_top5,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Metrics saved to {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = METRICS_PATH) -> 'MetricsLogger':
        """Re-create a MetricsLogger from a saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger = cls()
        logger.train_loss  = data['train_loss']
        logger.val_loss    = data['val_loss']
        logger.train_top1  = data['train_top1']
        logger.val_top1    = data['val_top1']
        logger.train_top5  = data['train_top5']
        logger.val_top5    = data['val_top5']
        return logger

    # ------------------------------------------------------------------
    def plot(self, save_path: str | Path = PLOT_PATH, show: bool = False) -> None:
        """Render and save a 3-panel figure: loss, top-1 accuracy, top-5 accuracy.

        Args:
            save_path: Where to write the PNG.
            show:      If True, call plt.show() (requires an interactive backend).
        """
        if not self.train_loss:
            print("No metrics recorded yet — nothing to plot.")
            return

        # Ensure the metrics directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(self.train_loss) + 1)

        fig = plt.figure(figsize=(15, 5))
        fig.suptitle('Training Curves', fontsize=14, fontweight='bold')
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # ---- Panel 1: Loss ----------------------------------------
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(epochs, self.train_loss, label='Train', color='steelblue',  linewidth=1.8)
        ax1.plot(epochs, self.val_loss,   label='Val',   color='darkorange', linewidth=1.8, linestyle='--')
        ax1.set_title('Cross-Entropy Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ---- Panel 2: Top-1 Accuracy ------------------------------
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(epochs, [v * 100 for v in self.train_top1], label='Train', color='steelblue',  linewidth=1.8)
        ax2.plot(epochs, [v * 100 for v in self.val_top1],   label='Val',   color='darkorange', linewidth=1.8, linestyle='--')
        # Mark best validation epoch
        best_epoch = int(max(range(len(self.val_top1)), key=lambda i: self.val_top1[i])) + 1
        best_val   = max(self.val_top1) * 100
        ax2.axvline(best_epoch, color='green', linestyle=':', linewidth=1.2, label=f'Best val ({best_val:.1f}%)')
        ax2.set_title('Top-1 Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ---- Panel 3: Top-5 Accuracy ------------------------------
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(epochs, [v * 100 for v in self.train_top5], label='Train', color='steelblue',  linewidth=1.8)
        ax3.plot(epochs, [v * 100 for v in self.val_top5],   label='Val',   color='darkorange', linewidth=1.8, linestyle='--')
        ax3.set_title('Top-5 Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(alpha=0.3)

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

        if show:
            plt.show()

        plt.close(fig)


# ------------------------------------------------------------------
# Standalone: re-plot from a saved metrics.json
# ------------------------------------------------------------------
if __name__ == '__main__':
    if METRICS_PATH.exists():
        logger = MetricsLogger.load(METRICS_PATH)
        logger.plot(show=True)
    else:
        print(f"No {METRICS_PATH} found. Run training first, or call logger.save() during training.")