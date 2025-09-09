# /sam/infer.py

import os
import sys
from pathlib import Path
import numpy as np

def main():
    # --- Paths ---
    ROOT = Path(__file__).resolve().parent              # /sam
    FASTSAM_DIR = ROOT / "FastSAM"                      # /sam/FastSAM
    RENDERS_DIR = ROOT / "renders"                      # /sam/renders
    MASKS_DIR = ROOT / "masks"                          # /sam/masks
    OVERLAY_DIR = ROOT / "mask_overlays"                # /sam/mask_overlays
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure we run FastSAM from its own folder (it relies on relative paths)
    os.chdir(FASTSAM_DIR)
    sys.path.insert(0, str(FASTSAM_DIR))

    # --- FastSAM imports (must happen after switching into FastSAM dir) ---
    from fastsam import FastSAM, FastSAMPrompt  # noqa: E402

    # --- Model setup (edit DEVICE to 'cuda' if you have GPU) ---
    DEVICE = "cpu"
    WEIGHTS = "./weights/FastSAM.pt"  # relative to /sam/FastSAM
    model = FastSAM(WEIGHTS)

    # --- Find input images ---
    RENDERS_REL = (Path("..") / "renders")  # relative to /sam/FastSAM
    image_paths = sorted(RENDERS_REL.glob("view_*.png"))

    if not image_paths:
        print(f"No images found matching 'view_*.png' in {RENERS_REL.resolve()}")
        return

    # --- Inference params (mirror your example) ---
    infer_kwargs = dict(device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    for img_rel in image_paths:
        img_abs = (FASTSAM_DIR / img_rel).resolve()
        img_name = img_rel.stem  # e.g., 'view_00'

        print(f"[INFO] Processing {img_abs}")

        # Run model
        results = model(str(img_rel), **infer_kwargs)

        # Build prompt helper + get "everything" annotations
        prompt = FastSAMPrompt(str(img_rel), results, device=DEVICE)
        anns = prompt.everything_prompt()

        # --- Try to extract binary masks robustly ---
        masks_np = None

        # Path A: Ultralytics-style masks on results[0]
        try:
            res0 = results[0] if isinstance(results, (list, tuple)) else results
            if hasattr(res0, "masks") and res0.masks is not None:
                # Expecting a tensor of shape [N, H, W]
                data = getattr(res0.masks, "data", None)
                if data is not None:
                    masks_np = data.detach().cpu().numpy().astype(np.uint8)  # 0/1 masks
        except Exception as e:
            print(f"[WARN] Could not read masks from results[0].masks.data: {e}")

        # Path B: Fallback—attempt to pull binary masks from annotations
        if masks_np is None and anns is not None:
            try:
                collected = []
                for a in (anns if isinstance(anns, (list, tuple)) else [anns]):
                    if isinstance(a, dict):
                        if "mask" in a and a["mask"] is not None:
                            collected.append(np.array(a["mask"], dtype=np.uint8))
                        elif "segmentation" in a and isinstance(a["segmentation"], np.ndarray):
                            collected.append(a["segmentation"].astype(np.uint8))
                if collected:
                    try:
                        masks_np = np.stack(collected, axis=0)
                    except Exception:
                        masks_np = np.array(collected, dtype=object)
            except Exception as e:
                print(f"[WARN] Could not derive masks from annotations: {e}")

        # Final fallback: store annotations object (not ideal, but better than losing data)
        if masks_np is None:
            print("[WARN] Falling back to saving annotations object (not binary masks).")
            masks_np = np.array(anns, dtype=object)

        # Save masks as .npy into /sam/masks (absolute path from /sam)
        out_npy = (ROOT / "masks" / f"masks_{img_name}.npy")
        np.save(out_npy, masks_np)
        print(f"[OK] Saved {out_npy} (shape={getattr(masks_np, 'shape', 'object array')})")

        # Save an overlay image (input with masks drawn) into /sam/mask_overlays
        #try:
         #   overlay_path = OVERLAY_DIR / f"{img_name}_mask.png"
          #  prompt.plot(annotations=anns, output_path=str(overlay_path))
          #  print(f"[OK] Saved overlay: {overlay_path}")
       # except Exception as e:
        #    print(f"[WARN] Could not save overlay for {img_name}: {e}")

if __name__ == "__main__":
    main()
