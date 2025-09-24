# /sam/infer.py

import os
import sys
from pathlib import Path
import numpy as np

import ram_grounded_sam

def main():
    # --- Paths ---
    ROOT = Path(__file__).resolve().parent              # /sam
    print(ROOT)
    RENDERS_DIR = ROOT / "renders"                      # /sam/renders
    MASKS_DIR = ROOT / "masks"                          # /sam/masks
    OVERLAY_DIR = ROOT / "mask_overlays"                # /sam/mask_overlays
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # --- Find input images ---
    RENDERS_REL = Path("renders")
    image_paths = sorted(RENDERS_REL.glob("view_*.png"))

    if not image_paths:
        print(f"No images found matching 'view_*.png' in {RENDERS_REL.resolve()}")
        return

    for img_rel in image_paths:
        img_name = img_rel.stem  # e.g., 'view_00'

        print(f"[INFO] Processing {img_rel}")
        ram_grounded_sam.infer(MASKS_DIR, img_rel, img_name, box_threshold=0.08, text_threshold=0.08, iou_threshold=0.2)

if __name__ == "__main__":
    main()
