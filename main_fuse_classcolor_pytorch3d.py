#!/usr/bin/env python3
# main_fuse_classcolor_pytorch3d.py
# Class voting using PyTorch3D rasterizer (naïve mode; no bin overflow).
# - Loads mesh with PyTorch3D
# - Builds cameras from camera_params.json (R/T + vertical FoV or K)
# - Rasterizes to get per-pixel face ids (pix_to_face)
# - Votes on GPU if available
# - Optional: writes colored OBJ (per-vertex averaged from per-face colors)

import os
import re
import json
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2
import torch

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,  # only if we ever use K
    RasterizationSettings,
    MeshRasterizer,
)

# ---------- Utilities ----------

def find_view_index_from_any(name: str) -> int:
    m = re.search(r"view_(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse view index from filename: {name}")
    return int(m.group(1))

def _natural_key(s: str):
    parts = re.findall(r'\d+|\D+', str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def _expand_split_RT_format(cam_params: Dict) -> Optional[List[Dict]]:
    if not isinstance(cam_params, dict):
        return None
    if "R" in cam_params and "T" in cam_params:
        Rs, Ts = cam_params["R"], cam_params["T"]
        if isinstance(Rs, list) and isinstance(Ts, list) and len(Rs) == len(Ts):
            intr = cam_params.get("intrinsics", {})
            fx, fy = intr.get("fx", None), intr.get("fy", None)
            cx, cy = intr.get("cx", None), intr.get("cy", None)
            use_fov = False
            fov_deg = None
            K = None
            if all(v is not None for v in (fx, fy, cx, cy)):
                K = np.array([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
            else:
                fov_deg = cam_params.get("fov_deg_vertical", cam_params.get("fov", None))
                if fov_deg is not None:
                    use_fov = True
                    fov_deg = float(fov_deg)
            out = []
            for i in range(len(Rs)):
                out.append(dict(
                    R=np.asarray(Rs[i], dtype=np.float32),
                    T=np.asarray(Ts[i], dtype=np.float32),
                    use_fov=use_fov,
                    fov_deg=fov_deg if use_fov else None,
                    K=None if use_fov else K
                ))
            return out
    return None

def _extract_cam_entries(cam_params) -> List[Dict]:
    expanded = _expand_split_RT_format(cam_params)
    if expanded is not None:
        return expanded
    if isinstance(cam_params, list):
        return cam_params
    if isinstance(cam_params, dict):
        for k in ("cameras", "views", "frames"):
            v = cam_params.get(k, None)
            if isinstance(v, list):
                return v
        items = [(k, v) for k, v in cam_params.items() if isinstance(v, dict)]
        items.sort(key=lambda kv: _natural_key(kv[0]))
        if items:
            return [v for _, v in items]
    raise TypeError(f"Unsupported camera_params JSON type/shape: {type(cam_params).__name__}")

def build_cameras_from_json(cam_params, image_size_for_fov: int) -> List[Dict]:
    entries = _extract_cam_entries(cam_params)
    cams: List[Dict] = []
    for idx, c in enumerate(entries):
        R = np.asarray(c.get("R", c.get("rotation")), dtype=np.float32)
        T = np.asarray(c.get("T", c.get("translation")), dtype=np.float32)
        if R.ndim == 3 and R.shape[0] == 1: R = R[0]
        if T.ndim == 2 and T.shape[0] == 1: T = T[0]
        K = None
        use_fov = False
        fov_deg: Optional[float] = None
        if "intrinsics" in c and isinstance(c["intrinsics"], (list, tuple, np.ndarray)):
            K = np.asarray(c["intrinsics"], dtype=np.float32)
        elif "K" in c and c["K"] is not None:
            K = np.asarray(c["K"], dtype=np.float32)
        elif "intrinsics" in c and isinstance(c["intrinsics"], dict):
            intr = c["intrinsics"]
            fx, fy, cx, cy = [intr.get(k, None) for k in ("fx", "fy", "cx", "cy")]
            if all(v is not None for v in (fx, fy, cx, cy)):
                K = np.array([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
        if K is None:
            fov_deg = float(c.get("fov_deg_vertical", c.get("fov", 60.0)))
            use_fov = True
        cams.append(dict(R=R, T=T, use_fov=use_fov, fov_deg=fov_deg, K=K))
    return cams

def derive_fx_fy_from_fov(image_h: int, image_w: int, fov_deg: float) -> Tuple[float, float]:
    # match your render assumption: vertical FoV
    H_for_fov = image_h if image_h == image_w else max(image_h, image_w)
    fy = (H_for_fov * 0.5) / np.tan(np.deg2rad(fov_deg * 0.5))
    fx = fy
    return float(fx), float(fy)

def safe_erode_mask(mask_uint8: np.ndarray, px: int) -> np.ndarray:
    m = (mask_uint8 > 0).astype(np.uint8)
    if px <= 0 or m.sum() == 0:
        return m
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cur = m
    for _ in range(px):
        nxt = cv2.erode(cur, kernel, iterations=1)
        if nxt.sum() == 0:
            return cur
        cur = nxt
    return cur

# ---------- OBJ writer (per-vertex colors averaged from per-face) ----------

def save_colored_obj_compat(verts: np.ndarray,
                            faces: np.ndarray,
                            verts_rgb: np.ndarray,
                            out_path: str):
    with open(out_path, "w") as f:
        f.write("# OBJ with per-vertex colors\n")
        for (x, y, z), (r, g, b) in zip(verts, verts_rgb):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for (i, j, k) in faces:
            f.write(f"f {i+1} {j+1} {k+1}\n")
    print(f"[INFO] Wrote colored mesh: {out_path}")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="Path to mesh (.obj)")
    ap.add_argument("--renders", required=True, help="Dir with view_XX.png and camera_params.json")
    ap.add_argument("--masks_dir", required=True, help="Dir with labeled masks (masks_view_XX.npy)")
    ap.add_argument("--image_size", type=int, default=1024, help="Fallback raster size")
    ap.add_argument("--out_labels", default="face_class_strings.npy")
    ap.add_argument("--out_colored_obj", default="")
    ap.add_argument("--erode_px", type=int, default=2,
                    help="Shrink mask borders before voting (reduces bleeding).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Torch device: {device.type}")
    print("[INFO] Rasterizer: PyTorch3D (bin_size=0 naive)")

    # Load mesh with PyTorch3D
    meshes = load_objs_as_meshes([args.obj], device=device)
    verts = meshes.verts_packed().detach().cpu().numpy()
    faces = meshes.faces_packed().detach().cpu().numpy()
    V, F = verts.shape[0], faces.shape[0]
    print(f"[INFO] Mesh: V={V}, F={F}")

    # Camera params
    cam_json = os.path.join(args.renders, "camera_params.json")
    with open(cam_json, "r") as f:
        cam_params = json.load(f)

    # Find image size (from one RGB if available)
    mask_files = sorted([f for f in os.listdir(args.masks_dir)
                         if f.startswith("masks_view_") and f.endswith(".npy")])
    if not mask_files:
        raise FileNotFoundError(f"No masks_view_XX.npy found in {args.masks_dir}")

    first_view_idx = find_view_index_from_any(mask_files[0])
    rgb_path0 = os.path.join(args.renders, f"view_{first_view_idx:02d}.png")
    if os.path.isfile(rgb_path0):
        rgb0 = imageio.imread(rgb_path0); H0, W0 = rgb0.shape[:2]
        image_h, image_w = H0, W0
    else:
        image_h = image_w = args.image_size

    cameras_spec = build_cameras_from_json(cam_params, max(image_h, image_w))

    # Rasterizer settings (naïve mode to avoid bin limits)
    try:
        rast_settings = RasterizationSettings(
            image_size=(image_h, image_w),
            faces_per_pixel=1,
            blur_radius=0.0,
            cull_backfaces=False,
            bin_size=0,            # <-- key for stability on huge meshes
            max_faces_per_bin=0
        )
    except TypeError:
        rast_settings = RasterizationSettings(
            image_size=(image_h, image_w),
            faces_per_pixel=1,
            blur_radius=0.0,
            cull_backfaces=False,
            bin_size=0
        )

    class_to_idx: Dict[str, int] = {}
    idx_to_class: List[str] = []
    class_votes = None  # (num_classes, F) torch.int64

    # Prepare mesh list (batch of 1) for rasterizer
    mesh_list = [meshes]

    for mask_file in tqdm(mask_files, desc="[INFO] Backprojecting (PyTorch3D rasterizer + GPU votes)"):
        view_idx = find_view_index_from_any(mask_file)
        if view_idx >= len(cameras_spec):
            continue

        # Batch cameras: 1
        cam = cameras_spec[view_idx]
        R = torch.from_numpy(cam["R"]).float().to(device)[None, ...]
        T = torch.from_numpy(cam["T"]).float().to(device)[None, ...]
        if cam["use_fov"]:
            cameras = FoVPerspectiveCameras(
                device=device, R=R, T=T, fov=float(cam["fov_deg"])
            )
        else:
            # Build focal from K (if ever used)
            K = cam["K"].astype(np.float32)
            fx, fy = float(K[0,0]), float(K[1,1])
            cx, cy = float(K[0,2]), float(K[1,2])
            # PyTorch3D's PerspectiveCameras with principal point in NDC is tricky;
            # For our pipeline we use FoV cameras, so this branch is rarely needed.
            # We approximate via FoV derived from fy:
            # fov_v = 2 * atan( H / (2 * fy) )
            fov_v = float(np.rad2deg(2.0 * np.arctan((image_h * 0.5) / max(fy, 1e-6))))
            cameras = FoVPerspectiveCameras(
                device=device, R=R, T=T, fov=fov_v
            )

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=rast_settings)

        # Rasterize to get per-pixel face ids
        with torch.no_grad():
            frags = rasterizer(meshes)
        pix_to_face = frags.pix_to_face[0]            # (H, W, K=1)
        face_ids = pix_to_face[..., 0].contiguous()   # (H, W), -1 where empty

        # Load masks for this view
        arr = np.load(os.path.join(args.masks_dir, mask_file), allow_pickle=True)
        labeled = [x.item() if hasattr(x, "item") else x for x in arr]

        valid_t = (face_ids >= 0)
        F_t = torch.tensor(F, device=device)

        for rec in labeled:
            m = np.asarray(rec.get("mask", None))
            lbl = (rec.get("label") or "unknown").strip().lower()
            if lbl == "unknown" or m is None:
                continue

            m_u8 = (m.astype(np.uint8) > 0).astype(np.uint8)
            mask_bin = safe_erode_mask(m_u8, int(args.erode_px))
            mask_t = torch.from_numpy(mask_bin.astype(bool)).to(device)

            sel_t = valid_t & mask_t
            if not torch.any(sel_t):
                continue

            # ensure class row exists
            if lbl not in class_to_idx:
                class_to_idx[lbl] = len(idx_to_class)
                idx_to_class.append(lbl)
                new_votes = torch.zeros((1, F), dtype=torch.int64, device=device)
                if class_votes is None:
                    class_votes = new_votes
                else:
                    class_votes = torch.cat([class_votes, new_votes], dim=0)
            ci = class_to_idx[lbl]

            f_sel = face_ids[sel_t].to(torch.int64)
            class_votes[ci] += torch.bincount(f_sel, minlength=F)

    # Reduce to per-face class index
    face_class_idx = np.full(F, -1, dtype=np.int32)
    if class_votes is not None:
        best_val, best_idx = torch.max(class_votes, dim=0)
        best_idx = best_idx.detach().cpu().numpy()
        best_val = best_val.detach().cpu().numpy()
        mask_pos = best_val > 0
        face_class_idx[mask_pos] = best_idx[mask_pos]

    face_class_strings = np.array([
        idx_to_class[i] if i >= 0 else "unknown" for i in face_class_idx
    ], dtype=object)
    np.save(args.out_labels, face_class_strings)
    print(f"[OK] Saved per-face classes: {args.out_labels}")

    # Optional: colored OBJ
    if args.out_colored_obj:
        base_palette = np.array([
            [0.90, 0.10, 0.10],
            [0.10, 0.90, 0.10],
            [0.10, 0.10, 0.90],
            [0.90, 0.90, 0.10],
            [0.90, 0.10, 0.90],
            [0.10, 0.90, 0.90],
            [0.90, 0.50, 0.10],
        ], dtype=np.float32)
        repeat = (len(idx_to_class) + len(base_palette) - 1) // len(base_palette)
        palette = np.vstack([base_palette] * repeat)[:len(idx_to_class)]

        face_colors = np.zeros((F, 3), dtype=np.float32)
        for fi in range(F):
            ci = face_class_idx[fi]
            face_colors[fi] = [0.6, 0.6, 0.6] if ci == -1 else palette[ci]

        verts_rgb = np.zeros((V, 3), dtype=np.float32)
        counts = np.zeros((V,), dtype=np.float32)
        for k in range(3):
            v_ids = faces[:, k]
            np.add.at(verts_rgb, v_ids, face_colors)
            np.add.at(counts, v_ids, 1.0)
        counts = np.maximum(counts, 1.0)[:, None]
        verts_rgb = verts_rgb / counts

        save_colored_obj_compat(verts, faces, verts_rgb, args.out_colored_obj)

        print("\n[INFO] Class → Color legend:")
        for cname, ci in class_to_idx.items():
            print(f"  {cname:15s} → {palette[ci]}")
        print("  unknown         → [0.6, 0.6, 0.6]")

if __name__ == "__main__":
    main()
