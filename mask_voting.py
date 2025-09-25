#!/usr/bin/env python3
# mask_voting.py
#
# Use saved dets_<view>.npy (pixel xyxy) and FastSAM masks to assign labels to masks.
# Writes: dino_dets/masks_labeled/masks_<view>.npy (list of dicts with mask + label + matched_box + score)
# Debug:  dino_dets/annotated/<view>_matched_masks.jpg (ONLY the masks that were labeled, no boxes)

from pathlib import Path
import numpy as np
import cv2

# Voting thresholds
COVERAGE_REQ = 0.85
AREA_LOW, AREA_HIGH = 0.85, 2.0

def _ensure_uint8_binary(m):
    m = np.asarray(m)
    if m.ndim > 2:
        m = np.squeeze(m)
    return (m > 0).astype(np.uint8)

def load_fastsam_masks(root: Path, stem: str, H: int, W: int):
    p = (root / "masks" / f"masks_{stem}.npy")
    if not p.exists():
        return []
    arr = np.load(p, allow_pickle=True)
    masks = []
    if arr.dtype == object:
        for m in arr:
            mi = _ensure_uint8_binary(m)
            if mi.shape != (H, W):
                mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append(mi)
    else:
        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                mi = _ensure_uint8_binary(arr[i])
                if mi.shape != (H, W):
                    mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
                masks.append(mi)
        elif arr.ndim == 2:
            mi = _ensure_uint8_binary(arr)
            if mi.shape != (H, W):
                mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append(mi)
    return masks

def overlay_masks(img_bgr, masks, alpha=0.45):
    """Overlay only the provided masks (binary uint8) as a single green matte with red edges."""
    if not masks:
        return img_bgr
    H, W = img_bgr.shape[:2]
    label_map = np.zeros((H, W), dtype=np.uint16)
    for i, m in enumerate(masks, 1):
        mm = m
        if mm.shape != (H, W):
            mm = cv2.resize(mm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        label_map[mm > 0] = i
    color = np.zeros_like(img_bgr)
    color[label_map > 0] = (40, 220, 80)
    out = cv2.addWeighted(img_bgr.astype(np.uint8), 1.0 - alpha, color.astype(np.uint8), alpha, 0)
    edges = cv2.Canny((label_map > 0).astype(np.uint8) * 255, 0, 1)
    out[edges > 0] = (0, 0, 255)
    return out

def _bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

def _mask_in_box_coverage(mask_bin, box):
    ys, xs = np.where(mask_bin > 0)
    if ys.size == 0:
        return 0.0
    x1, y1, x2, y2 = box
    inside = ((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)).sum()
    return float(inside) / float(ys.size)

def best_box_for_mask(mask_bin, dets):
    """Pick the tightest box whose area is within [AREA_LOW, AREA_HIGH] × mask_area and covers >= COVERAGE_REQ."""
    mask_area = int(mask_bin.sum())
    if mask_area == 0:
        return None
    cands = []
    for d in dets:
        box = d["bbox"]  # pixel xyxy (RELOADED)
        cov = _mask_in_box_coverage(mask_bin, box)
        if cov < COVERAGE_REQ:
            continue
        ratio = _bbox_area(box) / float(mask_area)
        if (ratio >= AREA_LOW) and (ratio <= AREA_HIGH):
            # sort by |ratio-1| asc, area asc (tighter), score desc
            cands.append((d, ratio, _bbox_area(box), float(d.get("score", 0.0))))
    if not cands:
        return None
    cands.sort(key=lambda t: (abs(t[1] - 1.0), t[2], -t[3]))
    return cands[0][0]

def main():
    ROOT = Path(__file__).resolve().parent
    RENDERS = ROOT / "renders"
    OUT_DIR = ROOT / "dino_dets"
    ANN_DIR = OUT_DIR / "annotated"; ANN_DIR.mkdir(exist_ok=True)
    LABELED_DIR = OUT_DIR / "masks_labeled"; LABELED_DIR.mkdir(exist_ok=True)

    images = sorted(RENDERS.glob("view_*.png"))
    if not images:
        print(f"[ERR] No images in {RENDERS}/view_*.png")
        return

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] unreadable {img_path}")
            continue
        H, W = img_bgr.shape[:2]
        stem = img_path.stem

        # Load reloaded dets (authoritative, pixel xyxy)
        dets_path = OUT_DIR / f"dets_{stem}.npy"
        if not dets_path.exists():
            print(f"[WARN] missing dets: {dets_path}, skipping")
            continue
        dets = np.load(dets_path, allow_pickle=True).tolist()

        # Load FastSAM masks
        masks = load_fastsam_masks(ROOT, stem, H, W)
        if not masks:
            print(f"[INFO] no masks for {stem}, skipping voting")
            continue

        # Voting: assign labels to masks
        labeled = []
        matched_masks = []  # keep only masks that got a label != 'unknown' for debug overlay
        for m in masks:
            mb = (np.asarray(m) > 0).astype(np.uint8)
            best = best_box_for_mask(mb, dets)
            if best is None:
                continue  # skip unknowns entirely
            entry = {
                "mask": mb,
                "label": best.get("label", "unknown"),
                "matched_box": best["bbox"],
                "score": float(best.get("score", 0.0))
            }
            if entry["label"] != "unknown":
                labeled.append(entry)
                matched_masks.append(mb)

        out_labeled = LABELED_DIR / f"masks_{stem}.npy"
        np.save(out_labeled, np.array(labeled, dtype=object))
        print(f"[OK] labeled masks -> {out_labeled} (saved={len(labeled)})")

        # Final debug: ONLY the masks that actually got labeled
        only_matched = overlay_masks(img_bgr, matched_masks, alpha=0.45)
        cv2.imwrite(str(ANN_DIR / f"{stem}_matched_masks.jpg"), only_matched)

if __name__ == "__main__":
    main()
