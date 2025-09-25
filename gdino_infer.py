#!/usr/bin/env python3
# gdino_infer.py
#
# GroundingDINO inference over renders/view_*.png.
# Saves per-view detections and debug images based on RELOADED dets (authoritative).

from pathlib import Path
import sys, re
import numpy as np
import cv2
import torch

# ---------------- config ----------------
BOX_THR   = 0.35
TEXT_THR  = 0.25
MIN_SCORE = 0.15
USE_CPU   = True

DEFAULT_CLASSES = [
    "building", "car", "tree", "sidewalk", "bus", "bicycle",
    "water", "bridge", "lake", "chimney", "dock"
]

# ---------------- utils ----------------
def load_classes(root: Path):
    f = root / "custom_classes.txt"
    if f.exists():
        classes = [ln.strip().lower() for ln in f.read_text().splitlines() if ln.strip()]
        seen, out = set(), []
        for c in classes:
            if c not in seen:
                out.append(c); seen.add(c)
        return out
    return [c.lower() for c in DEFAULT_CLASSES]

def make_prompt(classes):  # GroundingDINO expects "a . b . c ."
    return " . ".join(classes) + " ."

def clamp_sort_box(b, H, W):
    x1, y1, x2, y2 = [float(v) for v in b]
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), W-1))
    x2 = max(0, min(int(round(x2)), W-1))
    y1 = max(0, min(int(round(y1)), H-1))
    y2 = max(0, min(int(round(y2)), H-1))
    return [x1, y1, x2, y2]

def to_xyxy_pixels(b, H, W):
    """
    Convert GroundingDINO box -> pixel xyxy.
    Handles XYXY (normalized or pixels) and CXCYWH (normalized or pixels).
    """
    bb = np.array(b, dtype=float).tolist()
    if len(bb) != 4:
        raise ValueError(f"Unexpected bbox length: {len(bb)} for {b}")
    x1, y1, x2, y2 = bb
    vmin, vmax = min(bb), max(bb)

    # Looks like xyxy
    if (x2 >= x1) and (y2 >= y1):
        if 0.0 <= vmin and vmax <= 1.0:
            x1 *= W; y1 *= H; x2 *= W; y2 *= H
        return clamp_sort_box([x1, y1, x2, y2], H, W)

    # Otherwise treat as cxcywh
    cx, cy, w, h = bb
    if 0.0 <= vmin and vmax <= 1.0:
        cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return clamp_sort_box([x1, y1, x2, y2], H, W)

def _put_label(img, x, y, text, color):
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th_text), _ = cv2.getTextSize(text, font, fs, th)
    cv2.rectangle(img, (x, y - th_text - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, fs, (255,255,255), th, cv2.LINE_AA)

def draw_boxes(img_bgr, dets, color=(255, 0, 255), show_label=True):
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if show_label:
            lbl = d.get("label", "unknown")
            sc  = d.get("score", None)
            txt = f"{lbl}" + (f" {sc:.2f}" if isinstance(sc, (float,int)) else "")
            _put_label(out, x1, max(0, y1), txt, color)
    return out

def to_norm_xyxy_tensor(pixel_boxes, H, W):
    if not pixel_boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    arr = np.array(pixel_boxes, dtype=float)
    arr[:, [0, 2]] /= float(W)
    arr[:, [1, 3]] /= float(H)
    return torch.from_numpy(arr.astype(np.float32))

# ---------------- main ----------------
def main():
    ROOT = Path(__file__).resolve().parent
    RENDERS = ROOT / "renders"
    OUT_DIR = ROOT / "dino_dets"; OUT_DIR.mkdir(exist_ok=True)
    ANN_DIR = OUT_DIR / "annotated"; ANN_DIR.mkdir(exist_ok=True)

    classes = load_classes(ROOT)
    prompt = make_prompt(classes)
    print(f"[INFO] DINO prompt classes ({len(classes)}): {classes}")
    print(f"[INFO] Canonicalization → one of: {classes}")

    GDINO_CFG = (ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py").resolve()
    GDINO_WTS = (ROOT / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth").resolve()
    if not GDINO_CFG.exists(): raise FileNotFoundError(f"Config not found: {GDINO_CFG}")
    if not GDINO_WTS.exists(): raise FileNotFoundError(f"Weights not found: {GDINO_WTS}")

    sys.path.insert(0, str((ROOT / "GroundingDINO").resolve()))
    from groundingdino.util.inference import load_model, load_image, predict, annotate

    device = "cpu" if (USE_CPU or not torch.cuda.is_available()) else "cuda"
    model = load_model(str(GDINO_CFG), str(GDINO_WTS), device=device).eval()

    images = sorted(RENDERS.glob("view_*.png"))
    if not images:
        print(f"[ERR] No images found in {RENDERS}/view_*.png")
        return

    for img_path in images:
        image_source, image_tensor = load_image(str(img_path))
        if isinstance(image_source, np.ndarray):
            H, W = image_source.shape[:2]
            img_bgr = image_source if image_source.ndim == 3 else cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
        else:
            W, H = image_source.size
            img_bgr = np.array(image_source)
            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        with torch.no_grad():
            boxes_raw, logits_raw, phrases_raw = predict(
                model=model, image=image_tensor, caption=prompt,
                box_threshold=BOX_THR, text_threshold=TEXT_THR, device=device)

        # Convert to pixel xyxy + canonicalize labels
        def _tokenize(s: str): return re.findall(r"[a-z0-9]+", s.lower())
        def canonicalize(phrase: str):
            p = phrase.strip().lower()
            if p in classes: return p
            ptoks = _tokenize(p)
            if not ptoks: return "unknown"
            matches = []
            for cls in classes:
                ct = _tokenize(cls)
                L = len(ct)
                for i in range(0, len(ptoks) - L + 1):
                    if ptoks[i:i+L] == ct:
                        matches.append((cls, L)); break
            if not matches: return "unknown"
            matches.sort(key=lambda t: -t[1])  # longest
            return matches[0][0]

        dets = []
        for b, s, p in zip(boxes_raw, logits_raw, phrases_raw):
            try:
                score = float(s) if isinstance(s, (float, int)) else float(getattr(s, "item", lambda: s)())
            except Exception:
                score = float(s)
            if score < MIN_SCORE: continue
            bbox = to_xyxy_pixels(b, H, W)
            label = canonicalize(str(p))
            dets.append({"bbox": bbox, "label": label, "score": score, "raw_phrase": str(p).strip().lower()})

        out_npy = OUT_DIR / f"dets_{img_path.stem}.npy"
        np.save(out_npy, np.array(dets, dtype=object))
        print(f"[OK] boxes -> {out_npy} (N={len(dets)})")

        # Debug 1: DINO's annotate (normalized tensor)
        try:
            boxes_norm_t = to_norm_xyxy_tensor([d["bbox"] for d in dets], H, W)
            scores_t = torch.tensor([d["score"] for d in dets], dtype=torch.float32) if dets else torch.zeros(0)
            phrases = [d["raw_phrase"] for d in dets]
            ann = annotate(image_source=img_bgr, boxes=boxes_norm_t, logits=scores_t, phrases=phrases)
            cv2.imwrite(str(ANN_DIR / f"{img_path.stem}_annotate.jpg"), ann)
        except Exception as e:
            print(f"[WARN] annotate failed for {img_path.name}: {e}")

        # Debug 2: RELOADED dets drawn by us (authoritative)
        try:
            dets_saved = np.load(out_npy, allow_pickle=True).tolist()
            img_from_saved = draw_boxes(img_bgr, dets_saved, color=(255, 0, 255))
            cv2.imwrite(str(ANN_DIR / f"{img_path.stem}_from_saved.jpg"), img_from_saved)
        except Exception as e:
            print(f"[WARN] failed to draw from_saved for {img_path.name}: {e}")

if __name__ == "__main__":
    main()
