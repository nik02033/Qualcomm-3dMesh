# /sam/infer.py

import os
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

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

    global_tags = set()
    for img_rel in image_paths:
        print(f"[INFO] Getting tags {img_rel}")
        tags = ram_grounded_sam.get_labels(img_rel)
        global_tags.update(tags)
        
    global_tags = list(global_tags)
    print("Global tags:", global_tags)
    for img_rel in image_paths:
        img_name = img_rel.stem  # e.g., 'view_00'
        print(f"[INFO] Processing {img_rel}")
        ram_grounded_sam.infer_no_ram(MASKS_DIR, img_rel, img_name, global_tags, box_threshold=0.08, text_threshold=0.1, iou_threshold=0.2)
        

if __name__ == "__main__":
    main()
