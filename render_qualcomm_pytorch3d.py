#!/usr/bin/env python3
# render_qualcomm_pytorch3d.py
# PyTorch3D multiview renderer (coarse ring + adaptive fine grid)
# - Robust camera (PyTorch3D's look_at), per-view znear/zfar
# - Headlight lighting + strong ambient
# - Double-sided (no back-face culling)
# - Naïve rasterizer (bin_size=0) to avoid bin overflows on huge meshes
# - Fallback white vertex texture if OBJ textures aren’t available
# - Outputs: renders/view_XX.png + renders/camera_params.json (schema like the Open3D version)

import os, json, math, argparse
from typing import Tuple, List
import numpy as np
from PIL import Image

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    MeshRenderer,
    PointLights,
    Materials,
    TexturesVertex,
    blending,
    look_at_view_transform,
)

# ---------- Fixed configuration (no CLI) ----------
OUTPUT_DIR = "renders"
ALWAYS_MESHLAB_LIGHTING = True     # head-light style: point light at camera
ALWAYS_DOUBLE_SIDED     = True     # disable back-face culling

# ---------- Camera / render constants ----------
CLEARANCE_FRAC = 0.10      # keep camera ≥ 10% of bbox height above ground
VFOV_DEG = 60.0            # vertical FoV in degrees (match original intent)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def fov_to_intrinsics(w: int, h: int, fov_deg: float):
    f = (h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return f, f, w / 2.0, h / 2.0

def compute_bounds_ground_up(mesh_verts: np.ndarray):
    if mesh_verts.size == 0:
        return np.array([0.,0.,0.]), np.array([1.,1.,1.]), 1, -0.5
    mins = mesh_verts.min(axis=0)
    maxs = mesh_verts.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = (maxs - mins)
    up_idx = int(np.argmin(extent))  # smallest extent is "up"
    ground_value = float(np.percentile(mesh_verts[:, up_idx], 5.0))
    return center, extent, up_idx, ground_value

def fit_distance_axis(extent: np.ndarray, up_idx: int, vfov_deg: float, aspect: float = 1.0) -> float:
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

def np_img_to_pil(img_np: np.ndarray) -> Image.Image:
    img_np = np.clip(img_np, 0.0, 1.0)
    img_np = (img_np * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img_np)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--zoom", type=float, default=1.0, help="Scale the auto-fit camera distance (1.0=fit, <1 closer, >1 farther)")
    ap.add_argument("--elev_deg", type=float, default=35.0, help="Ring elevation in degrees (slightly top-down)")

    # Coarse/fine controls
    ap.add_argument("--num_coarse_views", type=int, default=None, help="Number of coarse orbit views (defaults to --num_views or 9)")
    ap.add_argument("--num_views", type=int, default=None, help="(Deprecated) alias for coarse views")
    ap.add_argument("--num_fine_views", type=int, default=3, help="Number of fine views per block when not overridden by adaptive logic")
    ap.add_argument("--fine_zoom", type=float, default=0.7, help="Extra zoom for fine views (<1 closer). Final fine dist = fit(block)*zoom*fine_zoom")

    # Extra
    ap.add_argument("--backoff", type=float, default=1.2, help="Retreat multiplier for camera distance (coarse & fine)")
    ap.add_argument("--z_down_frac", type=float, default=0.10, help="Shift cameras downward along global Z by this fraction of bbox Z extent")

    args = ap.parse_args()

    ensure_dir(OUTPUT_DIR)
    print(f"[INFO] Loading model (with textures): {args.input}")

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load mesh
    meshes = load_objs_as_meshes([args.input], device=device)
    # If textures are missing, use a white vertex texture so it’s not black
    if getattr(meshes, "textures", None) is None or meshes.textures is None:
        V = meshes.verts_packed().shape[0]
        white = torch.ones((1, V, 3), device=device)
        meshes.textures = TexturesVertex(verts_features=white)

    # Bounds and ground
    verts_cpu = meshes.verts_packed().detach().cpu().numpy()
    center, extent, up_idx, ground_val = compute_bounds_ground_up(verts_cpu)
    up_vec = unit_vec_for_axis(up_idx)
    horiz_axes = [i for i in range(3) if i != up_idx]
    bbox_h = float(extent[up_idx])
    min_eye_up = ground_val + CLEARANCE_FRAC * max(bbox_h, 1e-6)

    # Renderer config
    W = H = int(args.image_size)
    cull = False if ALWAYS_DOUBLE_SIDED else True

    # Naïve rasterizer to avoid bin overflows
    try:
        rast_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=cull,
            bin_size=0,
            max_faces_per_bin=0
        )
    except TypeError:
        rast_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=cull,
            bin_size=0
        )

    # Materials + Shader
    materials = Materials(
        device=device,
        specular_color=((0.15, 0.15, 0.15),),
        shininess=(32.0,),
    )

    def make_camera(R_t: torch.Tensor, T_t: torch.Tensor, znear: float, zfar: float):
        return FoVPerspectiveCameras(
            device=device,
            R=R_t, T=T_t,
            fov=VFOV_DEG,  # vertical FoV
            znear=znear, zfar=zfar,
        )

    def make_renderer(cameras, lights):
        shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blending.BlendParams(background_color=(0.0, 0.0, 0.0)),
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rast_settings),
            shader=shader
        )

    # Camera distances
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
                v = unit_vec_for_axis((up_idx + 1) % 3); dist = 1.0
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

        # Build camera with PyTorch3D look_at + per-view near/far
        eye_t = torch.tensor([eye], dtype=torch.float32, device=device)
        at_t  = torch.tensor([center_tgt], dtype=torch.float32, device=device)
        up_t  = torch.tensor([up_vec], dtype=torch.float32, device=device)
        R_t, T_t = look_at_view_transform(eye=eye_t, at=at_t, up=up_t, device=device)

        # Choose generous z-range based on distance to avoid clipping gigantic scenes
        cam_dist = float(np.linalg.norm(eye - center_tgt))
        znear = max(1e-2, cam_dist * 0.01)
        zfar  = max(znear * 1000.0, cam_dist * 10.0)

        cameras = make_camera(R_t, T_t, znear=znear, zfar=zfar)

        # Per-view lighting: strong ambient + headlight point light
        vdir = (center_tgt - eye); vdir /= (np.linalg.norm(vdir) + 1e-12)
        light_pos = eye - 0.02 * cam_dist * vdir
        intensity = (1.0 if not is_fine else 1.3) * 0.6
        lights = PointLights(
            device=device,
            location=torch.from_numpy(light_pos[None, :]).float().to(device),
            ambient_color=((0.6*intensity, 0.6*intensity, 0.6*intensity),),   # strong ambient to avoid black
            diffuse_color=((0.8*intensity, 0.8*intensity, 0.8*intensity),),
            specular_color=((0.9*intensity, 0.9*intensity, 0.9*intensity),),
        )

        renderer = make_renderer(cameras, lights)

        # Render
        with torch.no_grad():
            images = renderer(meshes)  # (1,H,W,4)
        img = images[0, ..., :3].detach().cpu().numpy()
        out = os.path.join(OUTPUT_DIR, f"view_{i_idx:02d}.png")
        np_img_to_pil(img).save(out, quality=95)
        print(f"[INFO] Saved {os.path.basename(out)}  (az={az_deg_actual:.1f}°, elev≈{elev_deg_actual:.1f}°)")

        # Save extrinsics for JSON (use the same R,T returned by PyTorch3D)
        Rs.append(R_t[0].detach().cpu().numpy().astype(float))
        Ts.append(T_t[0].detach().cpu().numpy().astype(float))
        eyes.append(tuple(map(float, eye)))
        used_azims.append(float(az_deg_actual)); used_elevs.append(float(elev_deg_actual))

    print(f"[INFO] Up-axis detected: {['X','Y','Z'][up_idx]}; elevation {args.elev_deg:.1f}°, zoom {args.zoom}")

    # --------- Coarse ring ---------
    idx = 0
    if (args.num_coarse_views if args.num_coarse_views is not None
        else (args.num_views if args.num_views is not None else 9)) > 0:
        num_coarse = (args.num_coarse_views if args.num_coarse_views is not None
                      else (args.num_views if args.num_views is not None else 9))
        num_coarse = max(int(num_coarse), 0)
        base_dist = fit_distance_axis(extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
        coarse_dist = max(base_dist * float(args.zoom), 1e-6)
        azims = np.linspace(0.0, 360.0, num=num_coarse, endpoint=False).tolist()
        for az in azims:
            eye = make_eye_for(center, az_deg=az, dist=coarse_dist, elev_deg_local=float(args.elev_deg))
            place_render(idx, eye, center, is_fine=False)
            idx += 1

    # --------- Fine grid (adaptive) ---------
    num_fine_default = max(int(args.num_fine_views), 0)
    if num_fine_default > 0:
        fine_extent = extent.copy()
        fine_extent[horiz_axes[0]] /= 3.0
        fine_extent[horiz_axes[1]] /= 3.0
        fine_base_dist = fit_distance_axis(fine_extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
        fine_dist = max(fine_base_dist * float(args.zoom) * float(args.fine_zoom), 1e-6)

        verts_up = verts_cpu[:, up_idx] if verts_cpu.size else np.array([center[up_idx]])
        p8  = float(np.percentile(verts_up, 8))   if verts_up.size > 1 else float(verts_up[0])
        p30 = float(np.percentile(verts_up, 30))  if verts_up.size > 1 else float(verts_up[0])
        p85 = float(np.percentile(verts_up, 85))  if verts_up.size > 1 else float(verts_up[0])

        block_centers = grid_block_centers(center, extent, up_idx)
        mins = center - 0.5 * extent
        steps = extent / 3.0

        def block_bounds(i, j):
            x0a, x1a = mins[horiz_axes[0]] + i*steps[horiz_axes[0]], mins[horiz_axes[0]] + (i+1)*steps[horiz_axes[0]]
            x0b, x1b = mins[horiz_axes[1]] + j*steps[horiz_axes[1]], mins[horiz_axes[1]] + (j+1)*steps[horiz_axes[1]]
            return (x0a, x1a), (x0b, x1b)

        def azims_for_n(n):
            if n <= 1: return [0.0]
            return np.linspace(0.0, 360.0, num=n, endpoint=False).tolist()

        b = 0
        for i in range(3):
            for j in range(3):
                b_center = block_centers[b]; b += 1
                if verts_cpu.size:
                    (a0, a1), (b0, b1) = block_bounds(i, j)
                    a_axis, b_axis = horiz_axes[0], horiz_axes[1]
                    ax = verts_cpu[:, a_axis]; bx = verts_cpu[:, b_axis]
                    mask = (ax >= a0) & (ax < a1) & (bx >= b0) & (bx < b1)
                    if np.any(mask):
                        block_avg_h = float(np.mean(verts_cpu[mask, up_idx]))
                    else:
                        block_avg_h = float(ground_val)
                else:
                    block_avg_h = float(center[up_idx])

                if block_avg_h <= p8:       n_views = 1
                elif block_avg_h <= p30:    n_views = 2
                elif block_avg_h > p85:     n_views = 4
                else:                        n_views = num_fine_default

                if n_views == 1:
                    vfov = math.radians(VFOV_DEG)
                    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * 1.0)  # aspect=1
                    half_x = 0.5 * fine_extent[horiz_axes[0]]
                    half_y = 0.5 * fine_extent[horiz_axes[1]]
                    d_x = half_x / max(math.tan(hfov / 2.0), 1e-6)
                    d_y = half_y / max(math.tan(vfov / 2.0), 1e-6)
                    td_dist = max(d_x, d_y) * 1.05
                    td_dist *= max(args.backoff, 1e-6)
                    eye = b_center.copy()
                    eye[up_idx] += td_dist
                    eps = 1e-3 * max(fine_extent[horiz_axes[0]], fine_extent[horiz_axes[1]], 1e-6)
                    eye[horiz_axes[0]] += eps
                    place_render(idx, eye, b_center, is_fine=True)
                    idx += 1
                else:
                    for az in azims_for_n(n_views):
                        eye = make_eye_for(b_center, az_deg=az, dist=fine_dist, elev_deg_local=float(args.elev_deg))
                        place_render(idx, eye, b_center, is_fine=True)
                        idx += 1

    # ---- Save camera params (schema similar to original) ----
    fx, fy, cx, cy = fov_to_intrinsics(H=H, w=W, fov_deg=VFOV_DEG) if False else fov_to_intrinsics(W, H, VFOV_DEG)
    cam_params = {
        "image_size": [H, W],
        "fov_deg_vertical": VFOV_DEG,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "near": None, "far": None},
        "R": [R.tolist() for R in Rs],
        "T": [T.tolist() for T in Ts],
        "eyes": eyes,
        "azimuth_deg": used_azims,
        "elevation_deg_actual": used_elevs,
        "target": center.astype(float).tolist(),
        "up_axis": ["X","Y","Z"][up_idx],
        "ground_value_along_up": float(ground_val),
        "clearance_frac": CLEARANCE_FRAC,
        "meshlab_lighting": True,
        "double_sided": True,
        "zoom": float(args.zoom),
        "elev_deg_requested": float(args.elev_deg),
        "source": {"used_triangle_model": True, "path": os.path.abspath(args.input)}
    }
    with open(os.path.join(OUTPUT_DIR, "camera_params.json"), "w") as f:
        json.dump(cam_params, f, indent=2)

    print(f"[INFO] Done. Outputs saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
