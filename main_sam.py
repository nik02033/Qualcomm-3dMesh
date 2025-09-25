#!/usr/bin/env python3
# main_sam.py
# FastSAM over renders/view_*.png
# Saves:
#   masks/masks_view_XX.npy
#   mask_debug/view_XX_masks_only.png
#   mask_debug/view_XX_masks_overlay.png

from pathlib import Path
import sys, numpy as np, cv2, torch

# ---- config ----
DEVICE = "cpu"                        # keep "cpu" for ARM; use "cuda" only if torch.cuda.is_available()
IMG_GLOB = "view_*.png"
OUT_MASKS_DIR = "masks"
OUT_DEBUG_DIR = "mask_debug"

def ensure_uint8_mask(m):
    m = np.asarray(m)
    return (m > 0).astype(np.uint8)

def save_mask_debug(img_bgr, masks_list, out_base: Path):
    H, W = img_bgr.shape[:2]
    label_map = np.zeros((H, W), dtype=np.int32)
    for i, m in enumerate(masks_list):
        label_map[m > 0] = i + 1

    # (1) masks-only
    only = (label_map > 0).astype(np.uint8) * 255
    cv2.imwrite(str(out_base.parent / f"{out_base.name}_masks_only.png"), only)

    # (2) overlay
    ov = img_bgr.copy()
    nz = (label_map > 0)
    if np.any(nz):
        alpha = 0.45
        color = np.zeros_like(ov); color[nz] = (40, 220, 80)
        ov = (ov.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    cv2.imwrite(str(out_base.parent / f"{out_base.name}_masks_overlay.png"), ov)

def main():
    ROOT = Path(__file__).resolve().parent
    RENDERS = ROOT / "renders"
    MASKS_DIR = ROOT / OUT_MASKS_DIR; MASKS_DIR.mkdir(exist_ok=True)
    DEBUG_DIR = ROOT / OUT_DEBUG_DIR; DEBUG_DIR.mkdir(exist_ok=True)

    # absolute path to weights; no chdir
    FASTSAM_WEIGHTS = (ROOT / "FastSAM"/ "weights" / "FastSAM.pt").resolve()
    if not FASTSAM_WEIGHTS.exists():
        raise FileNotFoundError(f"FastSAM weights not found: {FASTSAM_WEIGHTS}")

    # import FastSAM without changing cwd
    sys.path.insert(0, str((ROOT / "FastSAM").resolve()))
    from fastsam import FastSAM, FastSAMPrompt

    dev = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    fsam = FastSAM(str(FASTSAM_WEIGHTS))
    infer_kwargs = dict(device=dev, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    images = sorted(RENDERS.glob(IMG_GLOB))
    if not images:
        print(f"[ERR] No images like {IMG_GLOB} in {RENDERS}")
        return

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Unreadable image: {img_path}")
            continue
        H, W = img_bgr.shape[:2]

        # run FastSAM
        results = fsam(str(img_path), **infer_kwargs)
        prompt = FastSAMPrompt(str(img_path), results, device=dev)
        anns = prompt.everything_prompt()

        masks_list = []
        # Path A: ultralytics-like
        try:
            res0 = results[0] if isinstance(results, (list, tuple)) else results
            data = getattr(getattr(res0, "masks", None), "data", None)
            if data is not None:
                mnp = data.detach().cpu().numpy()
                for i in range(mnp.shape[0]):
                    mi = ensure_uint8_mask(mnp[i])
                    if mi.shape != (H, W):
                        mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
                    masks_list.append(mi)
        except Exception as e:
            print(f"[WARN] masks.data read failed: {e}")

        # Path B: annotations
        if not masks_list and anns is not None:
            for a in (anns if isinstance(anns, (list, tuple)) else [anns]):
                if isinstance(a, dict):
                    mm = a.get("mask") if a.get("mask") is not None else a.get("segmentation")
                    if mm is not None:
                        mm = ensure_uint8_mask(mm)
                        if mm.shape != (H, W):
                            mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
                        masks_list.append(mm)

        if not masks_list:
            print(f"[WARN] No masks for {img_path.name}")
            continue

        # save .npy
        out_npy = MASKS_DIR / f"masks_{img_path.stem}.npy"
        try:
            np.save(out_npy, np.stack(masks_list, axis=0).astype(np.uint8))
        except Exception:
            np.save(out_npy, np.array(masks_list, dtype=object))
        print(f"[OK] masks -> {out_npy} (M={len(masks_list)})")

        # debug images
        save_mask_debug(img_bgr, masks_list, DEBUG_DIR / img_path.stem)

if __name__ == "__main__":
    main()
