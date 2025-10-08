# render_qualcomm.py
# Oblique multiview renderer that:
# - auto-detects "up" axis (smallest AABB extent: X/Y/Z),
# - orbits around that axis only (consistent oblique angle),
# - enforces a ground-clearance clamp (no under-mesh views),
# - supports --meshlab_lighting (head-light) and --zoom.
#
# Args:
#   --input --output_dir --num_views --image_size
#   --zoom (1.0=fit, <1 closer, >1 farther)
#   --elev_deg (ring elevation; default 35)
#   --meshlab_lighting (optional head-light)
import os, json, math, argparse
from typing import Tuple, List
import numpy as np
import open3d as o3d

CLEARANCE_FRAC = 0.10      # fixed: keep camera ≥ 10% of bbox height above ground
VFOV_DEG = 60.0            # fixed internal projection
NEAR, FAR = 0.05, 5000.0   # fixed internal clip planes

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

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
    # Up axis = smallest extent (city tiles are much thinner in the vertical)
    up_idx = int(np.argmin(extent))
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", default="renders")
    ap.add_argument("--num_views", type=int, default=9)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="Scale the auto-fit camera distance (1.0=fit, <1 closer, >1 farther)")
    ap.add_argument("--elev_deg", type=float, default=35.0,
                    help="Ring elevation in degrees (slightly top-down)")
    ap.add_argument("--meshlab_lighting", action="store_true",
                    help="Light follows camera (MeshLab-style)")
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    print(f"[INFO] Loading model (with textures): {args.input}")

    # --- Bounds / up-axis / ground ---
    center, extent, up_idx, ground_val = compute_bounds_ground_up(args.input)
    up_vec = unit_vec_for_axis(up_idx)
    horiz_axes = [i for i in range(3) if i != up_idx]
    bbox_h = float(extent[up_idx])
    min_eye_up = ground_val + CLEARANCE_FRAC * max(bbox_h, 1e-6)

    # --- Renderer ---
    W = H = int(args.image_size)
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    scene = renderer.scene
    if hasattr(scene, "set_background"):
        scene.set_background([0.2, 0.6, 1, 1])

    # Load model (textures if available)
    used_model = False
    try:
        model = o3d.io.read_triangle_model(args.input)
        scene.add_model("model", model); used_model = True
    except Exception as e:
        print(f"[WARN] TriangleModel load failed, fallback mesh: {e}")
        mesh = o3d.io.read_triangle_mesh(args.input)
        if mesh.is_empty(): raise ValueError("Failed to load mesh/model.")
        mesh.compute_vertex_normals()
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"; mat.base_color = (1,1,1,1)
        if hasattr(mat, "reflectance"): mat.reflectance = 0.5
        scene.add_geometry("mesh", mesh, mat)

    s = getattr(scene, "scene", None)
    if s is not None:
        if hasattr(s, "set_indirect_light_intensity"):
            s.set_indirect_light_intensity(20000.0)
        if hasattr(s, "show_skybox"): s.show_skybox(False)
        elif hasattr(s, "enable_skybox"): s.enable_skybox(False)

    # --- Camera intrinsics ---
    fx, fy, cx, cy = fov_to_intrinsics(W, H, VFOV_DEG)
    cam = scene.camera
    FovType = o3d.visualization.rendering.Camera.FovType
    cam.set_projection(VFOV_DEG, float(W)/float(H), NEAR, FAR, FovType.Vertical)

    # --- Ring sampling (around detected up axis) ---
    total = max(int(args.num_views), 1)
    azims = np.linspace(0.0, 360.0, num=total, endpoint=False).tolist()
    elev_deg = float(args.elev_deg)

    base_dist = fit_distance_axis(extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
    dist = max(base_dist * float(args.zoom), 1e-6)

    Rs: List[np.ndarray] = []; Ts: List[np.ndarray] = []; eyes: List[Tuple[float, float, float]] = []
    used_azims: List[float] = []; used_elevs: List[float] = []

    def place_and_render(i: int, az_deg: float):
        el = math.radians(elev_deg)
        az = math.radians(az_deg)
        # Horizontal radius & up offset (in the auto-detected frame)
        r_h = dist * math.cos(el)
        up_off = dist * math.sin(el)

        eye = center.copy()
        # place along the two horizontal axes
        eye[horiz_axes[0]] += r_h * math.cos(az)
        eye[horiz_axes[1]] += r_h * math.sin(az)
        # add vertical offset
        eye[up_idx] += up_off

        # Enforce ground clearance
        if eye[up_idx] < min_eye_up:
            needed = (min_eye_up - center[up_idx]) / max(dist, 1e-6)
            needed = max(min(needed, 0.99), -0.99)
            el = math.asin(needed)
            r_h = dist * math.cos(el)
            eye = center.copy()
            eye[horiz_axes[0]] += r_h * math.cos(az)
            eye[horiz_axes[1]] += r_h * math.sin(az)
            eye[up_idx]        += dist * math.sin(el)

        R, T = look_at_R_T(eye, center, up_vec)
        Rs.append(R); Ts.append(T); eyes.append(tuple(map(float, eye)))
        used_azims.append(float(az_deg)); used_elevs.append(float(math.degrees(el)))

        try: cam.look_at(center=center.tolist(), eye=eye.tolist(), up=up_vec.tolist())
        except TypeError: cam.look_at(center.tolist(), eye.tolist(), up_vec.tolist())

        # Lighting
        if s is not None and hasattr(s, "set_sun_light"):
            if args.meshlab_lighting:
                vdir = (center - eye); vdir /= (np.linalg.norm(vdir) + 1e-12)
                try: s.set_sun_light(direction=vdir.tolist(), color=[1,1,1], intensity=70000.0)
                except TypeError: s.set_sun_light(vdir.tolist(), [1,1,1], 70000.0)
            else:
                up_dir = -unit_vec_for_axis(up_idx)  # overhead along -up
                try: s.set_sun_light(direction=up_dir.tolist(), color=[1,1,1], intensity=70000.0)
                except TypeError: s.set_sun_light(up_dir.tolist(), [1,1,1], 70000.0)
            if hasattr(s, "enable_sun_light"): s.enable_sun_light(True)

        img = renderer.render_to_image()
        out = os.path.join(args.output_dir, f"view_{i:02d}.png")
        o3d.io.write_image(out, img, quality=9)
        print(f"[INFO] Saved {os.path.basename(out)}  (az={az_deg:.1f}°, elev≈{math.degrees(el):.1f}°)")

    print(f"[INFO] Up-axis detected: {['X','Y','Z'][up_idx]}; elevation {elev_deg}°, zoom {args.zoom}")
    for i, az in enumerate(azims):
        place_and_render(i, az_deg=az)

    # Save camera params
    cam_params = {
        "image_size": [H, W],
        "fov_deg_vertical": VFOV_DEG,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "near": NEAR, "far": FAR},
        "R": [R.astype(float).tolist() for R in Rs],
        "T": [T.astype(float).tolist() for T in Ts],
        "eyes": eyes,
        "azimuth_deg": used_azims,
        "elevation_deg_actual": used_elevs,
        "target": center.astype(float).tolist(),
        "up_axis": ["X","Y","Z"][up_idx],
        "ground_value_along_up": float(ground_val),
        "clearance_frac": CLEARANCE_FRAC,
        "meshlab_lighting": bool(args.meshlab_lighting),
        "zoom": float(args.zoom),
        "elev_deg_requested": float(args.elev_deg),
        "source": {"used_triangle_model": used_model, "path": os.path.abspath(args.input)}
    }
    with open(os.path.join(args.output_dir, "camera_params.json"), "w") as f:
        json.dump(cam_params, f, indent=2)

    print(f"[INFO] Done. Outputs saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
