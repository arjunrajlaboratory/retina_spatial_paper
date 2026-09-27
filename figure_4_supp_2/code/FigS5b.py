# Fig S5b rod transcript views for selected stress genes at both LCA5 timepoints

## Load packages
import os
import sys
from pathlib import Path

## Load the shared vignette module from figure_4
here = Path(__file__).resolve().parent
fig4_code = here.parent.parent / "figure_4" / "code"
out_png = str(here.parent / "panels" / "FigS5b.png")
sys.path.insert(0, str(here.parent.parent / "_shared" / "code"))
sys.path.insert(0, str(fig4_code))
os.chdir(fig4_code)
import Fig4b_2 as vignette

## Build the crops and draw the per gene rod vignette grid
crops, tpa_vmax = vignette.build_crops()
vignette.draw_gene_grid_rod(crops, out_png, tpa_vmax)
print("wrote", out_png)
