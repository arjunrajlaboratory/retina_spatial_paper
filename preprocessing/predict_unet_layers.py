# Run the U-Net layer prediction for all retinal sections
from pathlib import Path
import runpy

# Execute the U-Net prediction script as if it were run directly
runpy.run_path(str(Path(__file__).parent / "unet/04_predict.py"), run_name="__main__")
