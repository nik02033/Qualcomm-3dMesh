# render_qualcomm.py
# Oblique multiview renderer with coarse + fine capture sets, adaptive fine views by block height,
# ALWAYS-on headlight-style lighting, ALWAYS double-sided materials, and fixed output_dir="renders".
#
# Adaptive rules (fine set):
#   <= 8th percentile height  -> 1 top-down view (fits block footprint)
#   <= 30th percentile        -> 2 views (180° apart)
#   >  85th percentile        -> 4 views (90° apart)
#   otherwise                 -> N views (default N=3, 120° apart)
#
# Outputs:
#   - renders/view_XX.png (coarse first, then fine)
#   - renders/camera_params.json (same schema as original)

import os, json, math, argparse
from typing import Tuple, List
import numpy as np
import open3d as o3d

# ---------- Fixed configuration (no CLI) ----------
OUTPUT_DIR = "renders"
ALWAYS_MESHLAB_LIGHTING = True     # head-light style sun direction from camera toward target
ALWAYS_DOUBLE_SIDED     = True     # disable back-face culling

# ---------- Camera / render constants ----------
CLEARANCE_FRAC = 0.10      # keep camera ≥ 10% of bbox height above ground
VFOV_DEG = 60.0            # fixed internal projection
NEAR, FAR = 0.05, 5000.0   # fixed internal clip planes


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def look_at_R_T(eye: np.ndarray, target: np.ndarray, up_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eye = eye.astype(np.float64); target = target.astype(np.float64); up_vec = up_vec.astype(np.float64)
    z = eye - target; z /= (np.linalg.norm(z) + 1e-12)
    x = np.cross(up_vec, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    T = -R @ eye.reshape(3, 1)
    return R, T.squeeze(-1)


def fov_to_intrinsics(w: int, h: int, fov_deg: float):
    f = (h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return f, f, w / 2.0, h / 2.0


def compute_bounds_ground_up(mesh_path: str):
    """
    Returns:
      center (3,), extent (3,), up_idx (0=x,1=y,2=z), ground_value (5th pct along up axis)
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        return np.array([0.,0.,0.]), np.array([1.,1.,1.]), 1, -0.5
    aabb = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center(), float)
    extent = np.asarray(aabb.get_extent(), float)
    up_idx = int(np.argmin(extent))  # smallest extent is "up"
    verts = np.asarray(mesh.vertices)
    ground_value = float(np.percentile(verts[:, up_idx], 5.0)) if verts.size else float(aabb.get_min_bound()[up_idx])
    return center, extent, up_idx, ground_value


def fit_distance_axis(extent: np.ndarray, up_idx: int, vfov_deg: float, aspect: float = 1.0) -> float:
    """Distance so AABB fits, using 'up_idx' as vertical axis."""
    half_h = float(extent[up_idx]) * 0.5
    other = [i for i in range(3) if i != up_idx]
    half_diag_horiz = float(np.linalg.norm(extent[other]) * 0.5)
    vfov = math.radians(vfov_deg)
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * aspect)
    d_v = half_h / max(math.tan(vfov / 2.0), 1e-6)
    d_h = half_diag_horiz / max(math.tan(hfov / 2.0), 1e-6)
    return max(d_v, d_h) * 1.02


def unit_vec_for_axis(idx: int):
    v = np.zeros(3, float); v[idx] = 1.0; return v


def grid_block_centers(center: np.ndarray, extent: np.ndarray, up_idx: int):
    """Return 9 centers for a 3x3 grid over the two horizontal axes within the AABB."""
    horiz = [i for i in range(3) if i != up_idx]
    mins = center - 0.5 * extent
    steps = extent / 3.0
    offsets_0 = [mins[horiz[0]] + (k + 0.5) * steps[horiz[0]] for k in range(3)]
    offsets_1 = [mins[horiz[1]] + (k + 0.5) * steps[horiz[1]] for k in range(3)]
    centers = []
    for i in range(3):
        for j in range(3):
            c = center.copy()
            c[horiz[0]] = offsets_0[i]
            c[horiz[1]] = offsets_1[j]
            centers.append(c)
    return centers


# ---- Two-sided material helper (portable across Open3D variants) ----
def make_material_two_sided(mat: o3d.visualization.rendering.MaterialRecord):
    """Best-effort to disable back-face culling across Open3D Python variants."""
    try:
        if hasattr(mat, "cull_mode"):
            CM = getattr(o3d.visualization.rendering.MaterialRecord, "CullMode", None)
            if CM is not None:
                for name in ("None", "NONE", "None_", "Off", "NoCull", "Disabled"):
                    val = getattr(CM, name, None)
                    if val is not None:
                        try:
                            mat.cull_mode = val
                            break
                        except Exception:
                            pass
        if hasattr(mat, "is_double_sided"):
            try:
                mat.is_double_sided = True
            except Exception:
                pass
        try:
            setattr(mat, "two_sided", True)
        except Exception:
            pass
    except Exception:
        pass


# ---- Per-view lighting (headlight + optional point/spot at camera) ----
def set_view_lighting(scene_inner, s_inner, eye: np.ndarray, target: np.ndarray, up_idx: int, is_fine: bool):
    """Headlight-style lighting each frame, with safe fallbacks across Open3D builds.
       If is_fine=True, intensities are boosted by +50%."""
    vdir = (target - eye)
    vdir /= (np.linalg.norm(vdir) + 1e-12)

    # Base intensities
    sun_intensity   = 70000.0
    point_intensity = 40000.0
    spot_intensity  = 60000.0

    if is_fine:
        sun_intensity   *= 1.5
        point_intensity *= 1.5
        spot_intensity  *= 1.5

    # 1) Sun light aligned with camera view (headlight)
    if s_inner is not None and hasattr(s_inner, "set_sun_light"):
        try:
            s_inner.set_sun_light(direction=vdir.tolist(), color=[1,1,1], intensity=sun_intensity)
        except TypeError:
            s_inner.set_sun_light(vdir.tolist(), [1,1,1], sun_intensity)
        if hasattr(s_inner, "enable_sun_light"):
            s_inner.enable_sun_light(True)

    # 2) Try to clear ephemeral lights to avoid accumulation (if API exists)
    for clear_name in ("clear_lights", "clear_lights_cache", "reset_lights"):
        if hasattr(s_inner, clear_name):
            try:
                getattr(s_inner, clear_name)()
            except Exception:
                pass

    # Camera light (slightly behind camera)
    light_pos = eye - 0.02 * np.linalg.norm(target - eye) * vdir
    color = [1.0, 1.0, 1.0]

    # Try several APIs depending on Open3D build
    for name, args in (
        ("add_point_light", (light_pos.tolist(), color, point_intensity, 0.0)),
        ("add_spot_light",  (light_pos.tolist(), vdir.tolist(), color, spot_intensity, math.radians(18.0), math.radians(28.0), 0.0)),
        ("add_light",       (light_pos.tolist(), color, point_intensity, 0.0)),
    ):
        if hasattr(s_inner, name):
            try:
                getattr(s_inner, name)(*args)
                break
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--zoom", type=float, default=1.0, help="Scale the auto-fit camera distance (1.0=fit, <1 closer, >1 farther)")
    ap.add_argument("--elev_deg", type=float, default=35.0, help="Ring elevation in degrees (slightly top-down)")

    # Coarse/fine controls
    ap.add_argument("--num_coarse_views", type=int, default=None, help="Number of coarse orbit views (defaults to --num_views or 9)")
    ap.add_argument("--num_views", type=int, default=None, help="(Deprecated) legacy alias for coarse views")
    ap.add_argument("--num_fine_views", type=int, default=3, help="Number of fine views per block when not overridden by adaptive logic")
    ap.add_argument("--fine_zoom", type=float, default=0.7, help="Additional zoom factor for fine views (<1 brings camera closer). Final fine distance = coarse-fit(block) * zoom * fine_zoom")

    # Kept additions
    ap.add_argument("--backoff", type=float, default=1.2, help="Extra retreat multiplier for camera distance (applied to both coarse and fine)")
    ap.add_argument("--z_down_frac", type=float, default=0.10, help="Shift cameras downward along global Z by this fraction of bbox Z-extent (default 0.10 = 10%)")

    args = ap.parse_args()

    ensure_dir(OUTPUT_DIR)
    print(f"[INFO] Loading model (with textures): {args.input}")

    # --- Bounds / up-axis / ground ---
    center, extent, up_idx, ground_val = compute_bounds_ground_up(args.input)
    up_vec = unit_vec_for_axis(up_idx)
    horiz_axes = [i for i in range(3) if i != up_idx]
    bbox_h = float(extent[up_idx])
    min_eye_up = ground_val + CLEARANCE_FRAC * max(bbox_h, 1e-6)

    # Also load vertices for adaptive height logic
    mesh_for_heights = o3d.io.read_triangle_mesh(args.input)
    verts = np.asarray(mesh_for_heights.vertices) if not mesh_for_heights.is_empty() else np.empty((0,3), dtype=np.float64)

    # --- Renderer ---
    W = H = int(args.image_size)
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    scene = renderer.scene
    if hasattr(scene, "set_background"):
        scene.set_background([0, 0, 0, 1])

    # Load model (textures if available), enforce double-sided if possible
    used_model = False
    try:
        model = o3d.io.read_triangle_model(args.input)

        if ALWAYS_DOUBLE_SIDED:
            try:
                if hasattr(model, "materials") and model.materials is not None:
                    for m in model.materials:
                        make_material_two_sided(m)
                else:
                    print("[WARN] Double-sided: model materials not exposed; may not affect TriangleModel path.")
            except Exception as e:
                print(f"[WARN] Double-sided tweaks for TriangleModel failed: {e}")

        scene.add_model("model", model)
        used_model = True

    except Exception as e:
        print(f"[WARN] TriangleModel load failed, fallback mesh: {e}")
        mesh = o3d.io.read_triangle_mesh(args.input)
        if mesh.is_empty():
            raise ValueError("Failed to load mesh/model.")
        mesh.compute_vertex_normals()
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"; mat.base_color = (1,1,1,1)
        if hasattr(mat, "reflectance"): mat.reflectance = 0.5
        if ALWAYS_DOUBLE_SIDED:
            make_material_two_sided(mat)
        scene.add_geometry("mesh", mesh, mat)

    s = getattr(scene, "scene", None)
    if s is not None:
        if hasattr(s, "set_indirect_light_intensity"):
            s.set_indirect_light_intensity(20000.0)
        if hasattr(s, "show_skybox"):
            s.show_skybox(False)
        elif hasattr(s, "enable_skybox"):
            s.enable_skybox(False)

    # --- Camera intrinsics ---
    fx, fy, cx, cy = fov_to_intrinsics(W, H, VFOV_DEG)
    cam = scene.camera
    FovType = o3d.visualization.rendering.Camera.FovType
    cam.set_projection(VFOV_DEG, float(W)/float(H), NEAR, FAR, FovType.Vertical)

    # --- Coarse ring (existing logic) ---
    num_coarse = (args.num_coarse_views if args.num_coarse_views is not None
                  else (args.num_views if args.num_views is not None else 9))
    num_coarse = max(int(num_coarse), 0)

    base_dist = fit_distance_axis(extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
    coarse_dist = max(base_dist * float(args.zoom), 1e-6)
    elev_deg = float(args.elev_deg)

    Rs: List[np.ndarray] = []; Ts: List[np.ndarray] = []; eyes: List[Tuple[float, float, float]] = []
    used_azims: List[float] = []; used_elevs: List[float] = []

    def make_eye_for(target: np.ndarray, az_deg: float, dist: float, elev_deg_local: float):
        el = math.radians(elev_deg_local)
        az = math.radians(az_deg)
        dist_eff = dist * max(args.backoff, 1e-6)
        r_h = dist_eff * math.cos(el)
        up_off = dist_eff * math.sin(el)
        eye = target.copy()
        eye[horiz_axes[0]] += r_h * math.cos(az)
        eye[horiz_axes[1]] += r_h * math.sin(az)
        eye[up_idx]        += up_off
        # global Z drop
        eye[2] -= float(args.z_down_frac) * float(extent[2])
        return eye

    def place_render(i_idx: int, eye: np.ndarray, center_tgt: np.ndarray, is_fine: bool):
        # Clearance clamp by adjusting elevation (keep radius)
        if eye[up_idx] < min_eye_up:
            v = eye - center_tgt
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                v = unit_vec_for_axis((up_idx + 1) % 3)
                dist = 1.0
            needed = (min_eye_up - center_tgt[up_idx]) / max(dist, 1e-6)
            needed = max(min(needed, 0.99), -0.99)
            el = math.asin(needed)
            r_h = dist * math.cos(el)
            horiz_axes_local = [a for a in range(3) if a != up_idx]
            vh = v.copy(); vh[up_idx] = 0.0
            az = math.atan2(vh[horiz_axes_local[1]], vh[horiz_axes_local[0]])
            eye = center_tgt.copy()
            eye[horiz_axes_local[0]] += r_h * math.cos(az)
            eye[horiz_axes_local[1]] += r_h * math.sin(az)
            eye[up_idx]              += dist * math.sin(el)
            elev_deg_actual = math.degrees(el)
            az_deg_actual = math.degrees(az) % 360.0
        else:
            v = eye - center_tgt
            dist = float(np.linalg.norm(v))
            horiz_axes_local = [a for a in range(3) if a != up_idx]
            az = math.atan2(v[horiz_axes_local[1]], v[horiz_axes_local[0]])
            el = math.asin(np.clip(v[up_idx] / max(dist, 1e-6), -1.0, 1.0))
            elev_deg_actual = math.degrees(el)
            az_deg_actual = math.degrees(az) % 360.0

        # Set camera + per-view lighting
        try:
            cam.look_at(center=center_tgt.tolist(), eye=eye.tolist(), up=up_vec.tolist())
        except TypeError:
            cam.look_at(center_tgt.tolist(), eye.tolist(), up_vec.tolist())

        if ALWAYS_MESHLAB_LIGHTING:
            set_view_lighting(scene, s, eye, center_tgt, up_idx, is_fine=is_fine)

        # Render
        img = renderer.render_to_image()
        out = os.path.join(OUTPUT_DIR, f"view_{i_idx:02d}.png")
        o3d.io.write_image(out, img, quality=9)
        print(f"[INFO] Saved {os.path.basename(out)}  (az={az_deg_actual:.1f}°, elev≈{elev_deg_actual:.1f}°)")

        # Save extrinsics
        R, T = look_at_R_T(eye, center_tgt, up_vec)
        Rs.append(R.astype(float)); Ts.append(T.astype(float)); eyes.append(tuple(map(float, eye)))
        used_azims.append(float(az_deg_actual)); used_elevs.append(float(elev_deg_actual))

    print(f"[INFO] Up-axis detected: {['X','Y','Z'][up_idx]}; elevation {elev_deg}°, zoom {args.zoom}")

    # Render coarse views
    idx = 0
    if num_coarse > 0:
        azims = np.linspace(0.0, 360.0, num=num_coarse, endpoint=False).tolist()
        for az in azims:
            eye = make_eye_for(center, az_deg=az, dist=coarse_dist, elev_deg_local=elev_deg)
            place_render(idx, eye, center, is_fine=False)
            idx += 1

    # --- Fine grid over 9 blocks with ADAPTIVE view counts ---
    num_fine_default = max(int(args.num_fine_views), 0)
    if num_fine_default > 0:
        # Distances for fine views use a 1/3 extent footprint
        fine_extent = extent.copy()
        fine_extent[horiz_axes[0]] /= 3.0
        fine_extent[horiz_axes[1]] /= 3.0
        fine_base_dist = fit_distance_axis(fine_extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
        fine_dist = max(fine_base_dist * float(args.zoom) * float(args.fine_zoom), 1e-6)

        # Global height percentiles for adaptive logic
        verts_up = verts[:, up_idx] if verts.size else np.array([center[up_idx]])
        p8  = float(np.percentile(verts_up, 8))   if verts_up.size > 1 else float(verts_up[0])
        p30 = float(np.percentile(verts_up, 30))  if verts_up.size > 1 else float(verts_up[0])
        p85 = float(np.percentile(verts_up, 85))  if verts_up.size > 1 else float(verts_up[0])

        # Precompute 3x3 block bounds and centers
        block_centers = grid_block_centers(center, extent, up_idx)

        mins = center - 0.5 * extent
        steps = extent / 3.0
        # For each block index (i,j): bounds on the two horizontal axes
        def block_bounds(i, j):
            x0a, x1a = mins[horiz_axes[0]] + i*steps[horiz_axes[0]], mins[horiz_axes[0]] + (i+1)*steps[horiz_axes[0]]
            x0b, x1b = mins[horiz_axes[1]] + j*steps[horiz_axes[1]], mins[horiz_axes[1]] + (j+1)*steps[horiz_axes[1]]
            return (x0a, x1a), (x0b, x1b)

        # Helper: choose azimuths given N
        def azims_for_n(n):
            if n <= 1:
                return [0.0]
            return np.linspace(0.0, 360.0, num=n, endpoint=False).tolist()

        # Walk blocks in the same order as grid_block_centers
        b = 0
        for i in range(3):
            for j in range(3):
                b_center = block_centers[b]; b += 1
                # Select vertices inside block footprint (2D filter on horizontal axes)
                if verts.size:
                    (a0, a1), (b0, b1) = block_bounds(i, j)
                    mask = (verts[:, horiz_axes[0]] >= a0) & (verts[:, horiz_axes[0]] < a1) & \
                           (verts[:, horiz_axes[1]] >= b0) & (verts[:, horiz_axes[1]] < b1)
                    if np.any(mask):
                        block_avg_h = float(np.mean(verts[mask, up_idx]))
                    else:
                        # No verts fell inside (very sparse / empty block) → assume near ground
                        block_avg_h = float(ground_val)
                else:
                    block_avg_h = float(center[up_idx])

                # Decide number of views adaptively
                if block_avg_h <= p8:
                    n_views = 1
                elif block_avg_h <= p30:
                    n_views = 2
                elif block_avg_h > p85:
                    n_views = 4
                else:
                    n_views = num_fine_default

                if n_views == 1:
                    # ---- Top-down single shot over the block ----
                    vfov = math.radians(VFOV_DEG)
                    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * 1.0)  # aspect=1
                    half_x = 0.5 * fine_extent[horiz_axes[0]]
                    half_y = 0.5 * fine_extent[horiz_axes[1]]
                    d_x = half_x / max(math.tan(hfov / 2.0), 1e-6)
                    d_y = half_y / max(math.tan(vfov / 2.0), 1e-6)
                    td_dist = max(d_x, d_y) * 1.05  # small tolerance
                    td_dist *= max(args.backoff, 1e-6)  # honor backoff; NO z_down_frac on top-down

                    eye = b_center.copy()
                    eye[up_idx] += td_dist
                    # tiny horizontal epsilon to avoid up//view parallel degeneracy
                    eps = 1e-3 * max(fine_extent[horiz_axes[0]], fine_extent[horiz_axes[1]], 1e-6)
                    eye[horiz_axes[0]] += eps

                    place_render(idx, eye, b_center, is_fine=True)
                    idx += 1
                else:
                    # Use evenly spaced azimuths for chosen N (2 -> 0°, 180°; 4 -> 0°, 90°, 180°, 270°; etc.)
                    block_azims = azims_for_n(n_views)
                    for az in block_azims:
                        eye = make_eye_for(b_center, az_deg=az, dist=fine_dist, elev_deg_local=elev_deg)
                        place_render(idx, eye, b_center, is_fine=True)
                        idx += 1

    # Save camera params (same schema as original)
    cam_params = {
        "image_size": [H, W],
        "fov_deg_vertical": VFOV_DEG,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "near": NEAR, "far": FAR},
        "R": [R.tolist() for R in Rs],
        "T": [T.tolist() for T in Ts],
        "eyes": eyes,
        "azimuth_deg": used_azims,
        "elevation_deg_actual": used_elevs,
        "target": center.astype(float).tolist(),
        "up_axis": ["X","Y","Z"][up_idx],
        "ground_value_along_up": float(ground_val),
        "clearance_frac": CLEARANCE_FRAC,
        "meshlab_lighting": True,   # fixed on
        "double_sided": True,       # fixed on
        "zoom": float(args.zoom),
        "elev_deg_requested": float(args.elev_deg),
        "source": {"used_triangle_model": used_model, "path": os.path.abspath(args.input)}
    }

    with open(os.path.join(OUTPUT_DIR, "camera_params.json"), "w") as f:
        json.dump(cam_params, f, indent=2)

    print(f"[INFO] Done. Outputs saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()


