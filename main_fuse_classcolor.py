#!/usr/bin/env python3
# main_fuse_classcolor.py
# Backproject 2D labeled masks to mesh faces (class-aware).
# - Inputs: labeled masks saved as dino_dets/masks_labeled/masks_view_XX.npy
#           (each entry is a dict with keys: 'mask', 'label', 'matched_box', 'score')
# - Outputs:
#   * face_class_strings.npy (per-face labels, 'unknown' for faces with no votes)
#   * colored OBJ with per-vertex RGB (unknown → gray)
#
# CLI example:
#   python main_fuse_classcolor.py \
#       --obj converted/Tile_+1984_+2688_L2.obj \
#       --renders ./renders \
#       --masks_dir ./dino_dets/masks_labeled \
#       --out_labels face_class_strings.npy \
#       --out_colored_obj colored_faces.obj

import os
import re
import json
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import open3d as o3d


# ---------- Utilities ----------

def find_view_index_from_any(name: str) -> int:
    m = re.search(r"view_(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse view index from filename: {name}")
    return int(m.group(1))


def binary_boundary(sil_bool: np.ndarray) -> np.ndarray:
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
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=False)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {obj_path} has no triangles.")
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

    z = Xc[:, 2]
    z = np.maximum(z, 1e-6)

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


def rasterize_triangles(verts: np.ndarray,
                        faces: np.ndarray,
                        uv: np.ndarray,
                        depth: np.ndarray,
                        W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    face_idx_img = np.full((H, W), -1, dtype=np.int32)
    zbuf_img = np.full((H, W), np.inf, dtype=np.float32)
    tri_uv = uv[faces]
    tri_z = depth[faces]

    def edge_fn(ax, ay, bx, by, px, py):
        return (py - ay) * (bx - ax) - (px - ax) * (by - ay)

    for f_id in range(faces.shape[0]):
        p = tri_uv[f_id]
        z = tri_z[f_id]
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
    """
    Write an OBJ with *duplicated* vertices so each face has its own 3 vertices,
    all colored identically = crisp class boundaries (no bleeding).
    - verts: (V,3) original
    - faces: (F,3) indices into verts
    - face_colors: (F,3) float in [0,1] per face
    """
    with open(out_path, "w") as f:
        f.write("# OBJ with per-face (no-bleed) vertex colors\n")
        # emit 3 new vertices per face, each with the face color
        for (i, j, k), (r, g, b) in zip(faces, face_colors):
            x1, y1, z1 = verts[i]
            x2, y2, z2 = verts[j]
            x3, y3, z3 = verts[k]
            f.write(f"v {x1:.6f} {y1:.6f} {z1:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"v {x2:.6f} {y2:.6f} {z2:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"v {x3:.6f} {y3:.6f} {z3:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        # faces now reference the new sequential vertices
        for fi in range(faces.shape[0]):
            base = fi * 3
            f.write(f"f {base+1} {base+2} {base+3}\n")
    print(f"[INFO] Wrote no-bleed per-face OBJ: {out_path}")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="Path to mesh (.obj)")
    ap.add_argument("--renders", required=True, help="Dir with view_XX.png and camera_params.json")
    ap.add_argument("--masks_dir", required=True, help="Dir with labeled masks (masks_view_XX.npy)")
    ap.add_argument("--image_size", type=int, default=1024, help="Fallback raster size")
    ap.add_argument("--out_labels", default="face_class_strings.npy")
    ap.add_argument("--out_colored_obj", default="")
    args = ap.parse_args()

    print(f"[INFO] Loading mesh from: {args.obj}")
    verts, faces = load_mesh_np(args.obj)
    V, F = verts.shape[0], faces.shape[0]
    print(f"[INFO] Mesh: V={V}, F={F}")

    cam_json = os.path.join(args.renders, "camera_params.json")
    with open(cam_json, "r") as f:
        cam_params = json.load(f)

    mask_files = sorted([f for f in os.listdir(args.masks_dir) if f.startswith("masks_view_") and f.endswith(".npy")])
    if not mask_files:
        raise FileNotFoundError(f"No masks_view_XX.npy found in {args.masks_dir}")

    first_view_idx = find_view_index_from_any(mask_files[0])
    rgb_path0 = os.path.join(args.renders, f"view_{first_view_idx:02d}.png")
    if os.path.isfile(rgb_path0):
        rgb0 = imageio.imread(rgb_path0)
        H0, W0 = rgb0.shape[:2]
        image_size_for_fov = max(H0, W0)
    else:
        image_size_for_fov = args.image_size

    cameras = build_cameras_from_json(cam_params, image_size_for_fov)

    # Global votes per face per class
    class_to_idx = {}
    idx_to_class = []
    class_votes = None  # shape (num_classes, F)

    for mask_file in tqdm(mask_files, desc="[INFO] Backprojecting (class-aware)"):
        view_idx = find_view_index_from_any(mask_file)
        if view_idx >= len(cameras):
            continue

        rgb_path = os.path.join(args.renders, f"view_{view_idx:02d}.png")
        if os.path.isfile(rgb_path):
            rgb = imageio.imread(rgb_path)
            H_rgb, W_rgb = rgb.shape[:2]
        else:
            H_rgb = W_rgb = args.image_size
            rgb = np.zeros((H_rgb, W_rgb, 3), dtype=np.uint8)

        cam = cameras[view_idx]
        uv, depth_pos, cam_sign = project_points(
            verts, cam["R"], cam["T"], W_rgb, H_rgb,
            cam["use_fov"], cam["fov_deg"], cam["K"]
        )
        faces_hw, zbuf = rasterize_triangles(verts, faces, uv, depth_pos, W_rgb, H_rgb)
        valid = faces_hw >= 0

        arr = np.load(os.path.join(args.masks_dir, mask_file), allow_pickle=True)
        labeled = [x.item() if hasattr(x, "item") else x for x in arr]

        for rec in labeled:
            m = np.asarray(rec.get("mask", None))
            lbl = (rec.get("label") or "unknown").strip().lower()
            if lbl == "unknown":
                continue
            if lbl not in class_to_idx:
                class_to_idx[lbl] = len(idx_to_class)
                idx_to_class.append(lbl)
                new_votes = np.zeros((1, F), dtype=np.int64)
                class_votes = new_votes if class_votes is None else np.vstack([class_votes, new_votes])
            ci = class_to_idx[lbl]
            mask_bin = (m.astype(np.uint8) > 0)
            sel = valid & mask_bin
            if not np.any(sel):
                continue
            f_sel = faces_hw[sel]
            class_votes[ci] += np.bincount(f_sel, minlength=F)

    # Argmax per face (with unknown = gray)
    face_class_idx = np.full(F, -1, dtype=np.int32)
    if class_votes is not None:
        best_idx = np.argmax(class_votes, axis=0)
        best_val = np.max(class_votes, axis=0)
        face_class_idx[best_val > 0] = best_idx[best_val > 0]

    # Map indices back to class strings
    face_class_strings = np.array([
        idx_to_class[i] if i >= 0 else "unknown" for i in face_class_idx
    ], dtype=object)
    np.save(args.out_labels, face_class_strings)
    print(f"[OK] Saved per-face classes: {args.out_labels}")

    # Colored OBJ
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
            if ci == -1:
                face_colors[fi] = [0.6, 0.6, 0.6]  # gray for unknown
            else:
                face_colors[fi] = palette[ci]

        verts_rgb = np.zeros((V, 3), dtype=np.float32)
        counts = np.zeros((V,), dtype=np.float32)
        for k in range(3):
            v_ids = faces[:, k]
            np.add.at(verts_rgb, v_ids, face_colors)
            np.add.at(counts, v_ids, 1.0)
        counts = np.maximum(counts, 1.0)[:, None]
        verts_rgb = verts_rgb / counts

        save_colored_obj_compat(verts, faces, verts_rgb, args.out_colored_obj)

        # Print palette legend
        print("\n[INFO] Class → Color legend:")
        for cname, ci in class_to_idx.items():
            print(f"  {cname:15s} → {palette[ci]}")
        print("  unknown         → [0.6, 0.6, 0.6]")


if __name__ == "__main__":
    main()
