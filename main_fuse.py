#!/usr/bin/env python3
# main_fuse.py
# Backproject 2D masks to mesh faces WITHOUT PyTorch/PyTorch3D.
# - Mesh I/O via Open3D
# - CPU NumPy rasterizer (pix_to_face + z-buffer)
#
# Inputs:
#   --obj                path to .obj mesh
#   --renders_dir        directory with 'view_XX.png' and camera_params.json
#   --masks_dir          directory with 'masks_view_XX.npy' stacks (uint8, [M,H,W])
#   --image_size         fallback raster size if RGB not found (default 1024)
#   --out_labels         output .npy per-face labels (default face_labels.npy)
#   --out_colored_obj    optional path to write vertex-colored OBJ
#   --debug_dir          directory for silhouettes and overlays (default debug_bp)
#
# Notes:
#   * Accepts several camera_params.json layouts, including the one produced by render_qualcomm.py:
#       {
#         "image_size":[H,W],
#         "fov_deg_vertical": 60.0,
#         "intrinsics":{"fx":..,"fy":..,"cx":..,"cy":..,"near":..,"far":..},
#         "R":[3x3, 3x3, ...], "T":[3, 3, ...], ... }
#     Also supports per-view dict lists, dicts-of-views, or entries with "extrinsics".
#   * Assumes world->view convention: X_cam = R @ X_world + T.
#     If median z < 0 for projected verts, z is flipped so depth stays positive.
#   * faces_per_pixel = 1, no antialiasing, CPU-only (ARM/Qualcomm friendly).

import os
import re
import json
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import open3d as o3d

import pdb

# ---------- Utilities ----------

def find_view_index_from_any(name: str) -> int:
    """
    Parse NN from:
      - 'masks_view_NN.npy'
      - 'view_NN.png'
      - robust to 'view_NN.png.npy' if it shows up.
    """
    m = re.search(r"view_(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse view index from filename: {name}")
    return int(m.group(1))


def binary_boundary(sil_bool: np.ndarray) -> np.ndarray:
    """
    1-px contour of a binary mask with numpy only.
    """
    s = sil_bool
    nbrs = [
        np.roll(np.roll(s,  1, 0),  0, 1),
        np.roll(np.roll(s, -1, 0),  0, 1),
        np.roll(np.roll(s,  0, 0),  1, 1),
        np.roll(np.roll(s,  0, 0), -1, 1),
        np.roll(np.roll(s,  1, 0),  1, 1),
        np.roll(np.roll(s,  1, 0), -1, 1),
        np.roll(np.roll(s, -1, 0),  1, 1),
        np.roll(np.roll(s, -1, 0), -1, 1),
    ]
    all_inside = s.copy()
    for n in nbrs:
        all_inside &= n
    boundary = s & (~all_inside)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = False
    return boundary


def load_mesh_np(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read mesh with Open3D -> numpy arrays.
    Returns:
      verts: (V,3) float32
      faces: (F,3) int32
    """
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=False)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {obj_path} has no triangles.")
    # Clean up geometry (does not touch coordinates beyond removing degenerates/dupes)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return verts, faces


# --- camera JSON helpers ---

def _natural_key(s: str):
    """Sort keys like 'view_2, view_10' numerically."""
    parts = re.findall(r'\d+|\D+', str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _expand_split_RT_format(cam_params: Dict) -> Optional[List[Dict]]:
    """
    Handle "split arrays" format like render_qualcomm.py:
      {"R":[...], "T":[...], "intrinsics":{fx,fy,cx,cy,...}, "fov_deg_vertical":60, ...}
    Returns list of per-view dicts or None if this shape doesn't match.
    """
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
                # Fall back to fov if provided
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
    """
    Accept many shapes:
      A) Split arrays: {"R":[...], "T":[...], "intrinsics":{...}}  (render_qualcomm.py)
      B) [ {R,T,fov|intrinsics}, ... ]
      C) {"cameras":[...]} or {"views":[...]} or {"frames":[...]}
      D) {"view_00": {...}, "view_01": {...}, ...}
    Returns a list of per-view dicts (each holds at least R,T and either fov OR K).
    """
    # Try A)
    expanded = _expand_split_RT_format(cam_params)
    if expanded is not None:
        return expanded

    # B/C/D)
    if isinstance(cam_params, list):
        return cam_params
    if isinstance(cam_params, dict):
        for k in ("cameras", "views", "frames"):
            v = cam_params.get(k, None)
            if isinstance(v, list):
                return v
        # Dict-of-views → sort by natural key of the dict key
        items = [(k, v) for k, v in cam_params.items() if isinstance(v, dict)]
        items.sort(key=lambda kv: _natural_key(kv[0]))
        if items:
            return [v for _, v in items]

    raise TypeError(f"Unsupported camera_params JSON type/shape: {type(cam_params).__name__}")


def build_cameras_from_json(cam_params, image_size_for_fov: int) -> List[Dict]:
    """
    Normalize camera entries and compute per-view projection info.
    Accepts R/T + (fov OR intrinsics/K). Also supports 3x4 / 4x4 'extrinsics'.
    Returns a list of dicts for each view:
      { R(3,3), T(3,), use_fov(bool), fov_deg(float|None), K(3,3|None) }
    """
    entries = _extract_cam_entries(cam_params)
    cams: List[Dict] = []

    for idx, c in enumerate(entries):
        if not isinstance(c, dict):
            raise TypeError(f"Camera entry {idx} is {type(c).__name__}, expected dict")

        # ---- extrinsics: R, T (or 'extrinsics' 3x4/4x4, or 'rotation'/'translation') ----
        R = c.get("R", None)
        T = c.get("T", None)

        if R is None or T is None:
            ext = c.get("extrinsics", None)
            if ext is not None:
                ext = np.asarray(ext, dtype=np.float32)
                if ext.shape == (4, 4):
                    R = ext[:3, :3]
                    T = ext[:3, 3]
                elif ext.shape == (3, 4):
                    R = ext[:, :3]
                    T = ext[:, 3]
            # try alternative names
            if R is None:
                R = c.get("rotation", None)
            if T is None:
                T = c.get("translation", None)

        if R is None or T is None:
            raise ValueError(f"Camera entry {idx} missing R/T (found keys: {list(c.keys())})")

        R = np.asarray(R, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        if R.ndim == 3 and R.shape[0] == 1:
            R = R[0]
        if T.ndim == 2 and T.shape[0] == 1:
            T = T[0]
        if R.shape != (3, 3) or T.shape != (3,):
            raise ValueError(f"Bad R/T shapes in camera {idx}: R{R.shape}, T{T.shape}")

        # ---- intrinsics: prefer explicit K/intrinsics; else use fov; else fallback ----
        K = None
        use_fov = False
        fov_deg: Optional[float] = None

        if "intrinsics" in c and c["intrinsics"] is not None and isinstance(c["intrinsics"], (list, tuple, np.ndarray)):
            K = np.asarray(c["intrinsics"], dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"Camera {idx} intrinsics must be 3x3, got {K.shape}")
        elif "K" in c and c["K"] is not None:
            K = np.asarray(c["K"], dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"Camera {idx} K must be 3x3, got {K.shape}")
        elif "intrinsics" in c and isinstance(c["intrinsics"], dict):
            # Intrinsics provided as dict (fx, fy, cx, cy)
            intr = c["intrinsics"]
            fx, fy = intr.get("fx", None), intr.get("fy", None)
            cx, cy = intr.get("cx", None), intr.get("cy", None)
            if all(v is not None for v in (fx, fy, cx, cy)):
                K = np.array([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
        if K is None:
            # fov option
            if "fov" in c and c["fov"] is not None:
                use_fov = True
                fov_deg = float(c["fov"])
            elif "fov_deg_vertical" in c and c["fov_deg_vertical"] is not None:
                use_fov = True
                fov_deg = float(c["fov_deg_vertical"])
            else:
                # derive from fy if provided, else default to 60°
                fy = c.get("fy", None)
                if fy is not None:
                    H = float(image_size_for_fov)
                    fov_deg = float(np.degrees(2 * np.arctan((H * 0.5) / float(fy))))
                    use_fov = True
                else:
                    use_fov = True
                    fov_deg = 60.0
                    print(f"[WARN] Camera {idx} missing intrinsics/fov; defaulting to fov=60°")

        cams.append(dict(R=R, T=T, use_fov=use_fov, fov_deg=fov_deg, K=K))

    return cams


def derive_fx_fy_from_fov(image_h: int, image_w: int, fov_deg: float) -> Tuple[float, float]:
    """
    PyTorch3D's FoV is vertical FOV for square images. For non-square we match
    your previous behavior (use the larger side).
    """
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
    """
    Transform world points with (R,T) to camera space and project to pixels.
    Handles Z-sign ambiguity by flipping if median z is negative.

    Returns:
      uv: (N,2) pixel coords (float32)
      depth_pos: (N,) positive depth values used for z-buffer (float32)
      cam_sign: +1 or -1 (whether we flipped z)
    """
    # World -> camera (assume X_cam = R @ X + T)
    Xc = (R @ pts_world.T).T + T[None, :]  # (N,3)
    z_raw = Xc[:, 2]
    cam_sign = 1.0
    if np.median(z_raw) < 0:
        cam_sign = -1.0
        Xc[:, 2] = -Xc[:, 2]

    z = Xc[:, 2]
    z = np.maximum(z, 1e-6)  # avoid divide-by-zero

    if use_fov:
        fx, fy = derive_fx_fy_from_fov(image_h, image_w, float(fov_deg))
        cx, cy = (image_w - 1) * 0.5, (image_h - 1) * 0.5
    else:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (Xc[:, 0] / z) + cx
    v = cy - fy * (Xc[:, 1] / z)
    uv = np.stack([u, v], axis=1).astype(np.float32)
    depth_pos = z.astype(np.float32)  # positive distance to camera
    return uv, depth_pos, cam_sign


def rasterize_triangles(verts: np.ndarray,
                        faces: np.ndarray,
                        uv: np.ndarray,
                        depth: np.ndarray,
                        W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal triangle rasterizer:
      - faces_per_pixel = 1
      - no antialiasing
      - z-buffer selects MIN(depth)
    Returns:
      face_idx_img: (H,W) int32, -1 for background
      zbuf_img:     (H,W) float32, +inf for background
    """
    face_idx_img = np.full((H, W), -1, dtype=np.int32)
    zbuf_img = np.full((H, W), np.inf, dtype=np.float32)

    tri_uv = uv[faces]   # (F,3,2)
    tri_z = depth[faces] # (F,3)

    def edge_fn(ax, ay, bx, by, px, py):
        return (py - ay) * (bx - ax) - (px - ax) * (by - ay)

    for f_id in range(faces.shape[0]):
        p = tri_uv[f_id]  # [[u0,v0],[u1,v1],[u2,v2]]
        z = tri_z[f_id]   # [z0,z1,z2]

        if not np.isfinite(p).all() or not np.isfinite(z).all():
            continue

        umin = max(int(np.floor(np.min(p[:, 0]))), 0)
        umax = min(int(np.ceil(np.max(p[:, 0]))), W - 1)
        vmin = max(int(np.floor(np.min(p[:, 1]))), 0)
        vmax = min(int(np.ceil(np.max(p[:, 1]))), H - 1)
        if umin > umax or vmin > vmax:
            continue

        area = edge_fn(p[0, 0], p[0, 1], p[1, 0], p[1, 1], p[2, 0], p[2, 1])
        if area == 0:
            continue

        xs = np.arange(umin, umax + 1, dtype=np.float32)
        ys = np.arange(vmin, vmax + 1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)

        w0 = edge_fn(p[1, 0], p[1, 1], p[2, 0], p[2, 1], XX, YY)
        w1 = edge_fn(p[2, 0], p[2, 1], p[0, 0], p[0, 1], XX, YY)
        w2 = edge_fn(p[0, 0], p[0, 1], p[1, 0], p[1, 1], XX, YY)

        if area < 0:
            mask = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
            w0, w1, w2 = -w0, -w1, -w2
            area = -area
        else:
            mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

        if not np.any(mask):
            continue

        wsum = (w0 + w1 + w2) + 1e-20
        l0 = w0 / wsum
        l1 = w1 / wsum
        l2 = w2 / wsum

        z_pix = l0 * z[0] + l1 * z[1] + l2 * z[2]

        z_old = zbuf_img[vmin:vmax+1, umin:umax+1]
        f_old = face_idx_img[vmin:vmax+1, umin:umax+1]
        closer = (z_pix < z_old) & mask

        z_old[closer] = z_pix[closer]
        f_old[closer] = f_id

        zbuf_img[vmin:vmax+1, umin:umax+1] = z_old
        face_idx_img[vmin:vmax+1, umin:umax+1] = f_old

    return face_idx_img, zbuf_img


def colorize_mesh_by_labels(verts: np.ndarray,
                            faces: np.ndarray,
                            labels: np.ndarray,
                            num_classes: Optional[int] = None) -> np.ndarray:
    """
    Per-face integer labels -> per-vertex colors by averaging incident face colors.
    Returns:
      verts_rgb: (V,3) float32 in [0,1]
    """
    V = verts.shape[0]
    F = faces.shape[0]
    if labels.shape[0] != F:
        raise ValueError(f"labels {labels.shape[0]} != #faces {F}")
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size > 0 else 1

    base_palette = np.array([
        [0.60, 0.60, 0.60], # gray
        [0.90, 0.10, 0.10], # red
        [0.10, 0.90, 0.10], # green
        [0.10, 0.10, 0.90], # blue
        [0.90, 0.90, 0.10], # yellow
        [0.90, 0.10, 0.90], # purple
        [0.10, 0.90, 0.90], # cyan
        [0.90, 0.50, 0.10], 
    ], dtype=np.float32)
    repeat = (num_classes + len(base_palette) - 1) // len(base_palette)
    palette = np.vstack([base_palette] * repeat)[:num_classes]

    face_colors = palette[labels]  # (F,3)
    verts_rgb = np.zeros((V, 3), dtype=np.float32)
    counts = np.zeros((V,), dtype=np.float32)

    for k in range(3):
        v_ids = faces[:, k]
        np.add.at(verts_rgb, v_ids, face_colors)
        np.add.at(counts, v_ids, 1.0)

    counts = np.maximum(counts, 1.0)[:, None]
    verts_rgb = verts_rgb / counts
    return verts_rgb


def save_colored_obj_compat(verts: np.ndarray,
                            faces: np.ndarray,
                            verts_rgb: np.ndarray,
                            out_path: str):
    """
    Write an OBJ with per-vertex colors: 'v x y z r g b' + 'f i j k' (1-based).
    """
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
    ap.add_argument("--renders_dir", required=True, help="Dir with view_XX.png and camera_params.json")
    ap.add_argument("--masks_dir", required=True, help="Dir with masks_view_XX.npy stacks")
    ap.add_argument("--image_size", type=int, default=1024, help="Fallback raster size")
    ap.add_argument("--out_labels", default="face_labels.npy", help="Output npy with per-face labels")
    ap.add_argument("--out_colored_obj", default="", help="Optional vertex-colored OBJ path")
    ap.add_argument("--debug_dir", default="debug_bp", help="Where to save silhouette & overlay PNGs")
    args = ap.parse_args()

    print(f"[INFO] Loading mesh from: {args.obj}")
    verts, faces = load_mesh_np(args.obj)
    V, F = verts.shape[0], faces.shape[0]
    print(f"[INFO] Mesh: V={V}, F={F}")

    # Cameras
    cam_json = os.path.join(args.renders_dir, "camera_params.json")
    if not os.path.isfile(cam_json):
        raise FileNotFoundError(f"Missing camera_params.json at {cam_json}")
    with open(cam_json, "r") as f:
        cam_params = json.load(f)

    # Discover mask stacks
    mask_files = sorted([f for f in os.listdir(args.masks_dir)
                         if f.startswith("masks_view_") and f.endswith(".npy")])
    if not mask_files:
        raise FileNotFoundError(f"No stacks like 'masks_view_XX.npy' found in {args.masks_dir}")

    os.makedirs(args.debug_dir, exist_ok=True)

    # Establish image size for deriving FOV when needed
    # Prefer camera_params["image_size"] if present
    image_size_for_fov = None
    if isinstance(cam_params, dict) and "image_size" in cam_params:
        imsz = cam_params["image_size"]
        if isinstance(imsz, (list, tuple)) and len(imsz) == 2:
            H0, W0 = int(imsz[0]), int(imsz[1])
            image_size_for_fov = H0 if H0 == W0 else max(H0, W0)

    if image_size_for_fov is None:
        first_view_idx = find_view_index_from_any(mask_files[0])
        rgb_path0 = os.path.join(args.renders_dir, f"view_{first_view_idx:02d}.png")
        if os.path.isfile(rgb_path0):
            rgb0 = imageio.imread(rgb_path0)
            rgb_h0, rgb_w0 = rgb0.shape[:2]
            image_size_for_fov = rgb_h0 if rgb_h0 == rgb_w0 else max(rgb_h0, rgb_w0)
        else:
            image_size_for_fov = args.image_size

    cameras = build_cameras_from_json(cam_params, image_size_for_fov)

    global_votes = None  # will be stacked [total_masks, F]
    mask_view_dirs = sorted([os.path.join(args.masks_dir, d)
                            for d in os.listdir(args.masks_dir)
                            if d.startswith("view_") and os.path.isdir(os.path.join(args.masks_dir, d))])

    # --- Build global label map across all views ---
    all_labels = set()
    for d in mask_view_dirs:
        json_path = os.path.join(d, "label.json")
        if not os.path.isfile(json_path):
            continue
        with open(json_path, "r") as f:
            mask_meta = json.load(f)["mask"]
            for entry in mask_meta:
                all_labels.add(entry["label"])

    label_to_class_id = {lbl: i for i, lbl in enumerate(sorted(all_labels))}
    num_classes = len(label_to_class_id)
    print(f"[INFO] Found {num_classes} global classes: {label_to_class_id}")

    global_votes = np.zeros((num_classes, F), dtype=np.int64)


    # Per-view backprojection
    for mask_file in tqdm(mask_files, desc="[INFO] Backprojecting"):
       # Parse view index (e.g., view_00 -> 0)
        view_name = os.path.basename(mask_file).split(".")[0]
        view_idx = view_name.split("_")[-1]

        # Load per-view mask metadata
        json_path = os.path.join(args.masks_dir, "view_" + view_idx, "label.json")
        with open(json_path, "r") as f:
            mask_meta = json.load(f)["mask"]
        mask_value_to_label = {m["value"]: m["label"] for m in mask_meta}
       
        view_idx = find_view_index_from_any(mask_file)
        if view_idx >= len(cameras):
            raise IndexError(f"view_{view_idx:02d} not in camera list (len={len(cameras)})")

        # Load RGB to lock raster size
        rgb_path = os.path.join(args.renders_dir, f"view_{view_idx:02d}.png")
        if os.path.isfile(rgb_path):
            rgb = imageio.imread(rgb_path)
            H_rgb, W_rgb = rgb.shape[:2]
        else:
            H_rgb = W_rgb = args.image_size
            rgb = np.zeros((H_rgb, W_rgb, 3), dtype=np.uint8)

        cam = cameras[view_idx]

        # Project & rasterize to get face index per pixel and z-buffer
        uv, depth_pos, cam_sign = project_points(
            verts, cam["R"], cam["T"],
            W_rgb, H_rgb,
            cam["use_fov"], cam["fov_deg"], cam["K"]
        )
        faces_hw, zbuf = rasterize_triangles(verts, faces, uv, depth_pos, W_rgb, H_rgb)

        sil_bool = faces_hw >= 0
        sil_area = int(sil_bool.sum())
        z_valid = zbuf[sil_bool]
        z_min = float(np.nan) if z_valid.size == 0 else float(np.min(z_valid))
        z_max = float(np.nan) if z_valid.size == 0 else float(np.max(z_valid))

        # Save silhouette & overlay
        sil_img = (sil_bool.astype(np.uint8) * 255)
        sil_path = os.path.join(args.debug_dir, f"view_{view_idx:02d}_sil.png")
        imageio.imwrite(sil_path, sil_img)

        boundary = binary_boundary(sil_bool)
        overlay = rgb.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay]*3, axis=-1)
        overlay[boundary] = [255, 0, 0]
        overlay_path = os.path.join(args.debug_dir, f"view_{view_idx:02d}_overlay.png")
        imageio.imwrite(overlay_path, overlay)

        print(f"[DEBUG] view_{view_idx:02d}: sil_area={sil_area}, "
              f"z_min={z_min:.6f}, z_max={z_max:.6f}, "
              f"rgb_size={W_rgb}x{H_rgb}, cam_sign={cam_sign:+.0f}")

        # Load masks for this view & check size
        masks_np = np.load(os.path.join(args.masks_dir, mask_file))
        if masks_np.dtype != np.uint8:
            masks_np = masks_np.astype(np.uint8)
        if masks_np.ndim != 3:
            raise ValueError(f"{mask_file} must be (M,H,W), got {masks_np.shape}")
        
        masks_np = np.vstack([np.zeros((1, 1024, 1024), dtype=masks_np.dtype), masks_np])

        M, Hm, Wm = masks_np.shape
        if (Hm, Wm) != (H_rgb, W_rgb):
            raise ValueError(
                f"Mask stack {mask_file} is {Wm}x{Hm} but raster is {W_rgb}x{H_rgb} — size mismatch."
            )

        # Vote per mask
        votes = np.zeros((M, F), dtype=np.int64)
        valid = faces_hw >= 0

        f_img = faces_hw  # alias
        for m in range(M):
            mask = masks_np[m] > 0
            sel = valid & mask
            if not np.any(sel):
                print(f"[DEBUG] Mask {m} in view_{view_idx:02d} covers {int(mask.sum())} px, "
                      f"but maps to 0 faces.")
                continue
            f_sel = f_img[sel]
            votes[m] += np.bincount(f_sel, minlength=F)
            
            cls_label = mask_value_to_label.get(m, None)
            if cls_label is None or cls_label not in label_to_class_id:
                continue
            cls_id = label_to_class_id[cls_label]
            global_votes[cls_id] += np.bincount(f_sel, minlength=F)

            
        # global_votes = votes if global_votes is None else np.vstack([global_votes, votes])

    # Final per-face label
    if global_votes is None or global_votes.size == 0:
        raise RuntimeError("No votes accumulated. Check silhouettes/overlays in debug_dir.")

    per_face_label = np.argmax(global_votes, axis=0).astype(np.int32)  # [F]
    np.save(args.out_labels, per_face_label)
    print(f"[INFO] Saved per-face labels: {args.out_labels}")

    # Optional colored mesh
    if args.out_colored_obj:
        verts_rgb = colorize_mesh_by_labels(verts, faces, per_face_label,
                                            num_classes=int(global_votes.shape[0]))
        save_colored_obj_compat(verts, faces, verts_rgb, args.out_colored_obj)
    # pdb.set_trace()

if __name__ == "__main__":
    main()
