#!/usr/bin/env python3
# gdino_infer.py
#
# GroundingDINO inference over renders/view_*.png using the pip package.
# Saves per-view detections as dino_dets/dets_view_XX.npy
# Also saves a simple debug image drawing the saved detections:
#   dino_dets/annotated/view_XX_from_saved.jpg

from pathlib import Path
import sys, re, os
import numpy as np
import cv2
import torch
import importlib
import importlib.resources as pkg_resources
from urllib.request import urlretrieve

# ---------------- config ----------------
BOX_THR   = 0.35
TEXT_THR  = 0.25
MIN_SCORE = 0.15
USE_CPU   = False

# If you already have a local weights path, set env:
#   export GROUNDINGDINO_WEIGHTS=/abs/path/to/groundingdino_swint_ogc.pth
DEFAULT_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/"
    "groundingdino_swint_ogc.pth"
)

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

def _resolve_pkg_config_path() -> str:
    """
    Return the absolute path to the packaged GroundingDINO SwinT OGC config.
    Works with the pip package (no local repo required).
    """
    import groundingdino
    # groundingdino/config/GroundingDINO_SwinT_OGC.py
    cfg_path = pkg_resources.files(groundingdino).joinpath("config/GroundingDINO_SwinT_OGC.py")
    return str(cfg_path)

def _ensure_weights() -> str:
    """
    Return a local path to the weights. If env GROUNDINGDINO_WEIGHTS is set,
    use that. Otherwise, download to ~/.cache/groundingdino/ if missing.
    """
    # 1) env override
    env_path = os.environ.get("GROUNDINGDINO_WEIGHTS")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"GROUNDINGDINO_WEIGHTS set but file not found: {p}")
        return str(p)

    # 2) default cache location
    cache_dir = Path(os.environ.get("GROUNDINGDINO_CACHE", "~/.cache/groundingdino")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "groundingdino_swint_ogc.pth"
    if not out_path.is_file():
        url = os.environ.get("GROUNDINGDINO_WEIGHTS_URL", DEFAULT_WEIGHTS_URL)
        print(f"[INFO] Downloading GroundingDINO weights → {out_path}")
        urlretrieve(url, out_path)
    return str(out_path)

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

    # ---- Use the pip package APIs ----
    try:
        from groundingdino.util.inference import load_model, load_image, predict
    except Exception as e:
        raise RuntimeError(
            "groundingdino is not installed in this environment. "
            "Install it with: pip install groundingdino-py"
        ) from e

    GDINO_CFG = _resolve_pkg_config_path()
    GDINO_WTS = _ensure_weights()

    device = "cpu" if (USE_CPU or not torch.cuda.is_available()) else "cuda"
    print(f"[INFO] Using device: {device}")

    model = load_model(GDINO_CFG, GDINO_WTS, device=device).eval()

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
                model=model,
                image=image_tensor,
                caption=prompt,
                box_threshold=BOX_THR,
                text_threshold=TEXT_THR,
                device=device
            )

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
            if score < MIN_SCORE:
                continue
            bbox = to_xyxy_pixels(b, H, W)
            label = canonicalize(str(p))
            dets.append({
                "bbox": bbox,
                "label": label,
                "score": score,
                "raw_phrase": str(p).strip().lower()
            })

        out_npy = OUT_DIR / f"dets_{img_path.stem}.npy"
        np.save(out_npy, np.array(dets, dtype=object))
        print(f"[OK] boxes -> {out_npy} (N={len(dets)})")

        # Simple debug image from saved dets (authoritative)
        try:
            dets_saved = np.load(out_npy, allow_pickle=True).tolist()
            img_from_saved = draw_boxes(img_bgr, dets_saved, color=(255, 0, 255))
            cv2.imwrite(str(ANN_DIR / f"{img_path.stem}_from_saved.jpg"), img_from_saved)
        except Exception as e:
            print(f"[WARN] failed to draw from_saved for {img_path.name}: {e}")

if __name__ == "__main__":
    main()
