#!/usr/bin/env python3
# main_fuse_classcolor.py  — class voting with GPU-friendly math + Numba CPU raster fallback
# - Torch device used for tensor ops / votes (cuda if available)
# - Triangle rasterization done on CPU, accelerated by Numba if installed
# - No OpenGL / no nvdiffrast / no PyTorch3D required

import os
import re
import json
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import open3d as o3d
import cv2

# Optional accelerator
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

import torch


# ---------- Utilities ----------

def find_view_index_from_any(name: str) -> int:
    m = re.search(r"view_(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse view index from filename: {name}")
    return int(m.group(1))


def load_mesh_np(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=False)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {obj_path} has no triangles.")
    # keep geometry robust but avoid UV warnings changing topology
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return verts, faces


# --- camera JSON helpers ---

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
            K = None
            use_fov = False
            fov_deg = None
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
        if R.ndim == 3 and R.shape[0] == 1:
            R = R[0]
        if T.ndim == 2 and T.shape[0] == 1:
            T = T[0]
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
    H_for_fov = image_h if image_h == image_w else max(image_h, image_w)
    fy = (H_for_fov * 0.5) / np.tan(np.deg2rad(fov_deg * 0.5))
    fx = fy
    return float(fx), float(fy)


def project_points(pts_world: np.ndarray,
                   R: np.ndarray,
                   T: np.ndarray,
                   image_w: int,
                   image_h: int,
                   use_fov: bool,
                   fov_deg: Optional[float],
                   K: Optional[np.ndarray]):
    Xc = (R @ pts_world.T).T + T[None, :]
    z_raw = Xc[:, 2]
    cam_sign = 1.0
    if np.median(z_raw) < 0:
        cam_sign = -1.0
        Xc[:, 2] = -Xc[:, 2]
    z = np.maximum(Xc[:, 2], 1e-6)
    if use_fov:
        fx, fy = derive_fx_fy_from_fov(image_h, image_w, float(fov_deg))
        cx, cy = (image_w - 1) * 0.5, (image_h - 1) * 0.5
    else:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (Xc[:, 0] / z) + cx
    v = cy - fy * (Xc[:, 1] / z)
    uv = np.stack([u, v], axis=1).astype(np.float32)
    depth_pos = z.astype(np.float32)
    return uv, depth_pos, cam_sign


# ---------- CPU Rasterizers (Numba + pure NumPy fallback) ----------

if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _edge_fn(ax, ay, bx, by, px, py):
        return (py - ay) * (bx - ax) - (px - ax) * (by - ay)

    @njit(parallel=True, cache=True, fastmath=True)
    def rasterize_triangles_cpu_numba(uv, depth, faces, W, H):
        face_idx_img = np.full((H, W), -1, dtype=np.int32)
        zbuf_img     = np.full((H, W), np.float32(np.inf), dtype=np.float32)

        for f_id in prange(faces.shape[0]):
            i0 = faces[f_id, 0]
            i1 = faces[f_id, 1]
            i2 = faces[f_id, 2]

            p0x, p0y = uv[i0, 0], uv[i0, 1]
            p1x, p1y = uv[i1, 0], uv[i1, 1]
            p2x, p2y = uv[i2, 0], uv[i2, 1]

            z0, z1, z2 = depth[i0], depth[i1], depth[i2]

            # bbox
            umin = int(np.floor(min(p0x, p1x, p2x)))
            umax = int(np.ceil (max(p0x, p1x, p2x)))
            vmin = int(np.floor(min(p0y, p1y, p2y)))
            vmax = int(np.ceil (max(p0y, p1y, p2y)))

            if umax < 0 or vmax < 0 or umin >= W or vmin >= H:
                continue

            if umin < 0: umin = 0
            if vmin < 0: vmin = 0
            if umax >= W: umax = W - 1
            if vmax >= H: vmax = H - 1
            if umin > umax or vmin > vmax:
                continue

            area = _edge_fn(p0x, p0y, p1x, p1y, p2x, p2y)
            if area == 0.0:
                continue

            flipped = False
            if area < 0.0:
                area = -area
                flipped = True

            for y in range(vmin, vmax + 1):
                for x in range(umin, umax + 1):
                    w0 = _edge_fn(p1x, p1y, p2x, p2y, x, y)
                    w1 = _edge_fn(p2x, p2y, p0x, p0y, x, y)
                    w2 = _edge_fn(p0x, p0y, p1x, p1y, x, y)

                    if flipped:
                        if not (w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0):
                            continue
                        w0 = -w0; w1 = -w1; w2 = -w2
                    else:
                        if not (w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0):
                            continue

                    wsum = w0 + w1 + w2
                    if wsum == 0.0:
                        continue

                    l0 = w0 / wsum
                    l1 = w1 / wsum
                    l2 = w2 / wsum
                    z  = l0 * z0 + l1 * z1 + l2 * z2

                    if z < zbuf_img[y, x]:
                        zbuf_img[y, x] = z
                        face_idx_img[y, x] = f_id

        return face_idx_img, zbuf_img


def rasterize_triangles_cpu_numpy(uv: np.ndarray,
                                  depth: np.ndarray,
                                  faces: np.ndarray,
                                  W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy fallback (slower)."""
    face_idx_img = np.full((H, W), -1, dtype=np.int32)
    zbuf_img     = np.full((H, W), np.inf, dtype=np.float32)

    def edge_fn(ax, ay, bx, by, px, py):
        return (py - ay) * (bx - ax) - (px - ax) * (by - ay)

    tri_uv = uv[faces]    # (F,3,2)
    tri_z  = depth[faces] # (F,3)
    for f_id in range(faces.shape[0]):
        p = tri_uv[f_id]; z = tri_z[f_id]
        umin = max(int(np.floor(np.min(p[:, 0]))), 0)
        umax = min(int(np.ceil (np.max(p[:, 0]))), W - 1)
        vmin = max(int(np.floor(np.min(p[:, 1]))), 0)
        vmax = min(int(np.ceil (np.max(p[:, 1]))), H - 1)
        if umin > umax or vmin > vmax:
            continue
        area = edge_fn(p[0,0], p[0,1], p[1,0], p[1,1], p[2,0], p[2,1])
        if area == 0:
            continue

        xs = np.arange(umin, umax + 1, dtype=np.float32)
        ys = np.arange(vmin, vmax + 1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)

        w0 = edge_fn(p[1,0], p[1,1], p[2,0], p[2,1], XX, YY)
        w1 = edge_fn(p[2,0], p[2,1], p[0,0], p[0,1], XX, YY)
        w2 = edge_fn(p[0,0], p[0,1], p[1,0], p[1,1], XX, YY)

        if area < 0:
            mask = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
            w0, w1, w2 = -w0, -w1, -w2
            area = -area
        else:
            mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

        if not np.any(mask):
            continue

        wsum = (w0 + w1 + w2) + 1e-20
        l0 = w0 / wsum; l1 = w1 / wsum; l2 = w2 / wsum
        z_pix = l0 * z[0] + l1 * z[1] + l2 * z[2]

        z_old = zbuf_img[vmin:vmax+1, umin:umax+1]
        f_old = face_idx_img[vmin:vmax+1, umin:umax+1]
        closer = (z_pix < z_old) & mask
        z_old[closer] = z_pix[closer]
        f_old[closer] = f_id
        zbuf_img[vmin:vmax+1, umin:umax+1] = z_old
        face_idx_img[vmin:vmax+1, umin:umax+1] = f_old

    return face_idx_img, zbuf_img


def rasterize_triangles_gpu_or_cpu(verts: np.ndarray,
                                   faces: np.ndarray,
                                   R: np.ndarray,
                                   T: np.ndarray,
                                   image_w: int,
                                   image_h: int,
                                   use_fov: bool,
                                   fov_deg: Optional[float],
                                   K: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    We keep your projection math; raster is CPU:
      - Numba-accelerated if available
      - otherwise pure NumPy
    """
    uv, depth_pos, _ = project_points(
        verts, R, T, image_w, image_h, use_fov, fov_deg, K
    )
    if _HAVE_NUMBA:
        return rasterize_triangles_cpu_numba(uv.astype(np.float32),
                                             depth_pos.astype(np.float32),
                                             faces.astype(np.int32),
                                             image_w, image_h)
    else:
        return rasterize_triangles_cpu_numpy(uv.astype(np.float32),
                                             depth_pos.astype(np.float32),
                                             faces.astype(np.int32),
                                             image_w, image_h)


# ---------- OBJ writers ----------

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


def save_obj_per_face_rgb_no_bleed(verts, faces, face_colors, out_path):
    with open(out_path, "w") as f:
        f.write("# OBJ with per-face (no-bleed) vertex colors\n")
        for (i, j, k), (r, g, b) in zip(faces, face_colors):
            x1, y1, z1 = verts[i]; x2, y2, z2 = verts[j]; x3, y3, z3 = verts[k]
            f.write(f"v {x1:.6f} {y1:.6f} {z1:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"v {x2:.6f} {y2:.6f} {z2:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"v {x3:.6f} {y3:.6f} {z3:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for fi in range(faces.shape[0]):
            base = fi * 3
            f.write(f"f {base+1} {base+2} {base+3}\n")
    print(f"[INFO] Wrote no-bleed per-face OBJ: {out_path}")


# ---------- NEW: safe erosion to shrink mask borders ----------

def safe_erode_mask(mask_uint8: np.ndarray, px: int) -> np.ndarray:
    """Erode mask by ~px pixels; if it would vanish, return last non-empty."""
    m = (mask_uint8 > 0).astype(np.uint8)
    if px <= 0 or m.sum() == 0:
        return m
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cur = m
    for _ in range(px):
        nxt = cv2.erode(cur, kernel, iterations=1)
        if nxt.sum() == 0:
            return cur  # stop before empty
        cur = nxt
    return cur


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
                    help="Shrink mask borders by this many pixels before voting (reduces bleeding).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Torch device: {device.type}")
    if _HAVE_NUMBA:
        print("[INFO] Rasterizer: Numba CPU")
    else:
        print("[INFO] Rasterizer: pure NumPy CPU (install `numba` for speed)")

    print(f"[INFO] Loading mesh from: {args.obj}")
    verts, faces = load_mesh_np(args.obj)
    V, F = verts.shape[0], faces.shape[0]
    print(f"[INFO] Mesh: V={V}, F={F}")

    cam_json = os.path.join(args.renders, "camera_params.json")
    with open(cam_json, "r") as f:
        cam_params = json.load(f)

    mask_files = sorted([f for f in os.listdir(args.masks_dir)
                         if f.startswith("masks_view_") and f.endswith(".npy")])
    if not mask_files:
        raise FileNotFoundError(f"No masks_view_XX.npy found in {args.masks_dir}")

    first_view_idx = find_view_index_from_any(mask_files[0])
    rgb_path0 = os.path.join(args.renders, f"view_{first_view_idx:02d}.png")
    if os.path.isfile(rgb_path0):
        rgb0 = imageio.imread(rgb_path0); H0, W0 = rgb0.shape[:2]
        image_size_for_fov = max(H0, W0)
    else:
        image_size_for_fov = args.image_size

    cameras = build_cameras_from_json(cam_params, image_size_for_fov)

    class_to_idx: Dict[str, int] = {}
    idx_to_class: List[str] = []
    class_votes = None  # shape (num_classes, F)

    for mask_file in tqdm(mask_files, desc="[INFO] Backprojecting (GPU votes, CPU raster)"):
        view_idx = find_view_index_from_any(mask_file)
        if view_idx >= len(cameras):
            continue

        rgb_path = os.path.join(args.renders, f"view_{view_idx:02d}.png")
        if os.path.isfile(rgb_path):
            rgb = imageio.imread(rgb_path); H_rgb, W_rgb = rgb.shape[:2]
        else:
            H_rgb = W_rgb = args.image_size
            rgb = np.zeros((H_rgb, W_rgb, 3), dtype=np.uint8)

        cam = cameras[view_idx]
        # CPU raster (Numba if available)
        faces_hw, zbuf = rasterize_triangles_gpu_or_cpu(
            verts, faces, cam["R"], cam["T"], W_rgb, H_rgb, cam["use_fov"], cam["fov_deg"], cam["K"]
        )
        valid = faces_hw >= 0

        arr = np.load(os.path.join(args.masks_dir, mask_file), allow_pickle=True)
        labeled = [x.item() if hasattr(x, "item") else x for x in arr]

        # voting on Torch (GPU if available)
        valid_t = torch.from_numpy(valid).to(device=device, dtype=torch.bool)
        faces_hw_t = torch.from_numpy(faces_hw).to(device=device, dtype=torch.int64)

        for rec in labeled:
            m = np.asarray(rec.get("mask", None))
            lbl = (rec.get("label") or "unknown").strip().lower()
            if lbl == "unknown" or m is None:
                continue

            m_u8 = (m.astype(np.uint8) > 0).astype(np.uint8)
            mask_bin = safe_erode_mask(m_u8, args.erode_px)

            mask_t = torch.from_numpy(mask_bin.astype(bool)).to(device=device)
            sel_t = valid_t & mask_t
            if not torch.any(sel_t):
                continue

            # ensure class rows exist
            if lbl not in class_to_idx:
                class_to_idx[lbl] = len(idx_to_class)
                idx_to_class.append(lbl)
                new_votes = torch.zeros((1, F), dtype=torch.int64, device=device)
                if class_votes is None:
                    class_votes = new_votes
                else:
                    class_votes = torch.cat([class_votes, new_votes], dim=0)
            ci = class_to_idx[lbl]

            f_sel = faces_hw_t[sel_t]
            class_votes[ci] += torch.bincount(f_sel, minlength=F)

    # reduce to per-face class index
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

        V = verts.shape[0]; F = faces.shape[0]
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
