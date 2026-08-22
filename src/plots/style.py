"""Shared figure style: NeurIPS text width, serif type, colourblind-safe hues."""
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
OPEN, CLOSED = BLUE, ORANGE
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = "RdBu"

WIDTH = 5.5          # NeurIPS \textwidth in inches
HALF = 2.68

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#4a4a48",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e2e1dd",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def save(fig, path):
    fig.savefig(path, format="pdf")
    fig.savefig(str(path).replace(".pdf", ".png"), format="png")
    plt.close(fig)
