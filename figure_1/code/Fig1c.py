# Fig 1c single Harmony UMAP of all conditions overlaid

## Load packages
import sys
import pandas as pd
import matplotlib.pyplot as plt

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
harmony = "../../_shared/data_processed/umap_coordinates.parquet"
output = "../panels/Fig1c.png"

## Define cell types in legend order
type_order = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
              "horizontal", "rgc", "muller", "microglia", "vascular", "rpe"]

## Read data
data = pd.read_parquet(harmony)

## Assign colors
colors = dict(paper_style.CELLTYPE_COLORS)
for cell_type in data["final_label"].unique():
    if cell_type not in colors:
        colors[cell_type] = "#cccccc"

## Set the plot range with a small margin
umap1 = data["umap1_harmony"].to_numpy()
umap2 = data["umap2_harmony"].to_numpy()
x_limits = (umap1.min() - 1.0, umap1.max() + 1.0)
y_limits = (umap2.min() - 1.0, umap2.max() + 1.0)

## Keep only the types present
present = set(data["final_label"])
order = []
for cell_type in type_order:
    if cell_type in present:
        order.append(cell_type)

## Set up the figure
figure = plt.figure(figsize=(5.0, 4.8))
axes = figure.add_axes([0.02, 0.02, 0.96, 0.96])

## Scatter most abundant types first
counts = data["final_label"].value_counts()
draw_order = sorted(order, key=lambda cell_type: -counts.get(cell_type, 0))
for cell_type in draw_order:
    subset = data[data["final_label"] == cell_type]
    axes.scatter(subset["umap1_harmony"], subset["umap2_harmony"],
                 s=1.2, c=colors[cell_type], lw=0, alpha=0.45, rasterized=True)
axes.set_xlim(*x_limits)
axes.set_ylim(*y_limits)
axes.set_aspect("equal", "datalim")
axes.set_xticks([])
axes.set_yticks([])
for spine in axes.spines.values():
    spine.set_visible(False)

## Label each cloud at its centroid
texts = []
for cell_type in order:
    subset = data[data["final_label"] == cell_type]
    texts.append(axes.text(
        float(subset["umap1_harmony"].median()),
        float(subset["umap2_harmony"].median()),
        paper_style.celltype_label(cell_type), ha="center", va="center", fontsize=11,
        fontweight="bold", color="black", zorder=10, clip_on=False))

## Nudge overlapping labels apart
figure.canvas.draw()
renderer = figure.canvas.get_renderer()
inverse_transform = axes.transData.inverted()
for _ in range(200):
    boxes = [text.get_window_extent(renderer) for text in texts]
    moved = False
    for index_a in range(len(texts)):
        for index_b in range(index_a + 1, len(texts)):
            box_a = boxes[index_a]
            box_b = boxes[index_b]
            if not box_a.overlaps(box_b):
                continue
            moved = True
            delta_x = (box_a.x0 + box_a.x1) - (box_b.x0 + box_b.x1)
            delta_y = (box_a.y0 + box_a.y1) - (box_b.y0 + box_b.y1)
            distance = (delta_x * delta_x + delta_y * delta_y) ** 0.5 or 1.0
            unit_x = delta_x / distance
            unit_y = delta_y / distance
            for which_text, sign in ((index_a, 1.0), (index_b, -1.0)):
                pixel_x, pixel_y = axes.transData.transform(texts[which_text].get_position())
                new_x, new_y = inverse_transform.transform((pixel_x + sign * unit_x * 4.0, pixel_y + sign * unit_y * 4.0))
                texts[which_text].set_position((new_x, new_y))
    if not moved:
        break

## Move the RPE and bipolar labels by hand
x_range = float(umap1.max() - umap1.min())
y_range = float(umap2.max() - umap2.min())
label_by_type = dict(zip(order, texts))
manual_offsets = {
    "rpe": (-0.05, 0.09),
    "bipolar": (-0.04, -0.06),
}
for cell_type, (dx, dy) in manual_offsets.items():
    if cell_type in label_by_type:
        x, y = label_by_type[cell_type].get_position()
        label_by_type[cell_type].set_position((x + dx * x_range, y + dy * y_range))

## Draw the orientation arrows
arrowprops = dict(arrowstyle="-|>", color="#2A2A2A", lw=1.4, shrinkA=0, shrinkB=0)
axes.annotate("", xy=(0.21, 0.05), xytext=(0.05, 0.05),
              xycoords="axes fraction", arrowprops=arrowprops, zorder=11)
axes.annotate("", xy=(0.05, 0.21), xytext=(0.05, 0.05),
              xycoords="axes fraction", arrowprops=arrowprops, zorder=11)
axes.text(0.13, 0.02, "UMAP 1", transform=axes.transAxes,
          ha="center", va="top", fontsize=9, fontweight="bold",
          color="#2A2A2A", zorder=11, clip_on=False)
axes.text(0.02, 0.13, "UMAP 2", transform=axes.transAxes,
          ha="right", va="center", rotation=90, fontsize=9, fontweight="bold",
          color="#2A2A2A", zorder=11, clip_on=False)

## Add the cell count
axes.text(0.99, 0.06, f"n = {len(data):,} cells", transform=axes.transAxes,
          ha="right", va="bottom", fontsize=11, fontweight="bold", color="black")

## Save the panel
figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output)
