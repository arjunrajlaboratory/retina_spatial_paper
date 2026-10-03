# Shared fonts, palettes, and rcParams for every panel
import matplotlib as mpl
import matplotlib.font_manager as fm


# Return True when the font family exposes a bold face distinct from its regular face
def _bold_is_real(family):
    try:
        reg = fm.findfont(fm.FontProperties(family=family, weight="normal"), fallback_to_default=False)
        bold = fm.findfont(fm.FontProperties(family=family, weight="bold"), fallback_to_default=False)
    except ValueError:
        return False
    return reg != bold


FONT = "Helvetica Neue" if _bold_is_real("Helvetica Neue") else "Arial"


def set_style():
    mpl.rcParams.update({
        "font.family": FONT,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.6,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "savefig.dpi": 300,
        "figure.dpi": 120,
        # Render the mathtext genotype label in the body font
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        # Italic slot for gene names, since mathit forces digits upright
        "mathtext.sf": f"{FONT}:italic",
    })


# Define the wild-type gray ramp and the LCA5 disease-severity ramp
CONDITION_ORDER = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30", "LCA5_P64"]
CONDITION_COLORS = {
    "WT_P21": "#999999", "WT_P64": "#000000",
    "LCA5_P21": "#E69F00", "LCA5_P30": "#D55E00", "LCA5_P64": "#8E2430",
}

# Render the genotype as italic Lca5 gt/gt
GENOTYPE = r"$\mathsf{Lca5}^{gt/gt}$"


# Convert a postnatal age token into days
def age_label(token):
    return f"{token[1:]}d" if token.startswith("P") and token[1:].isdigit() else token


# Map a condition to its display label
def condition_label(cond):
    if cond.startswith("LCA5"):
        age = cond.split("_", 1)[1] if "_" in cond else ""
        return f"{GENOTYPE} {age_label(age)}".strip()
    genotype, _, age = cond.partition("_")
    return f"Wild-type {age_label(age)}".strip()


# Two-tone genotype colors for titles and tick labels, with wild-type in black and LCA5 in red
CONDITION_SIMPLE_WT = "#000000"
CONDITION_SIMPLE_LCA5 = "#CC0000"


# Map wild-type to black and LCA5 to red
def condition_color_simple(cond):
    return CONDITION_SIMPLE_LCA5 if cond.startswith("LCA5") else CONDITION_SIMPLE_WT


# Cell type colors used in Figure 1
CELLTYPE_COLORS = {
    "rod": "#1f77b4", "cone": "#ff7f0e", "bipolar": "#2ca02c",
    "amacrine_gaba": "#d62728", "amacrine_gly": "#ff9896", "horizontal": "#9467bd",
    "rgc": "#8c564b", "muller": "#17becf", "microglia": "#e377c2",
    "vascular": "#7f7f7f", "rpe": "#3d0066",
}
CELLTYPE_DISPLAY = {
    "amacrine_gaba": "GABA AC", "amacrine_gly": "glycine AC",
    "muller": "Müller", "rgc": "RGC", "rpe": "RPE",
}


def celltype_label(ct):
    return CELLTYPE_DISPLAY.get(ct, ct.replace("_", " "))
