# Qualcomm-3dMesh
Download Environment
```bash
wget -O sam2-env.tar https://arizonastateu-my.sharepoint.com/personal/nkanodi1_sundevils_asu_edu/_layouts/15/download.aspx?UniqueId=5b13d3f4-b9ef-4927-b72c-3f616575752d
```


Download the `groundingdino_swint_ogc.pth` file into the `GroundingDINO/weights` directory with:

```bash
wget -P GroundingDINO/weights \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Run commands to generate renderings, inference using FastSAM and backprojecting onto the mesh. Sample usage for Tile_+1984_+2688_L2.obj:
```bash

export GROUNDINGDINO_WEIGHTS=/scratch/nkanodi1/Capstone/Qualcomm-3dMesh/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

For pytorch3d:
```bash
python render_qualcomm_pytorch3d.py   --input converted/Tile_+1990_+2691_L2.obj   --image_size 1024   --elev_deg 38   --zoom 0.5   --num_coarse_views 9   --num_fine_views 3   --fine_zoom 0.85   --backoff 1.5   --z_down_frac 0.10
```

For Open3d:
```bash
python render_qualcomm.py   --input converted/Tile_+1990_+2691_L2.obj   --image_size 1024   --elev_deg 38   --zoom 0.5   --num_coarse_views 9   --num_fine_views 3   --fine_zoom 0.85   --backoff 1.5   --z_down_frac 0.10

python collect_ram_all_classes.py --neg neg.txt --ra-dir recognize-anything --image-dir renders --checkpoint recognize-anything/pretrained/ram_plus_swin_large_14m.pth --out custom_classes.txt

python main_sam.py

python gdino_infer.py

python mask_voting.py
```
For pytorch3d backprojection:
```bash
python main_fuse_classcolor_pytorch3d.py   --obj converted/Tile_+1990_+2691_L2.obj   --renders ./renders   --masks_dir ./dino_dets/masks_labeled   --out_labels face_class_strings.npy  --out_colored_obj colored_faces.obj

```
For Open3d backprojection:
```bash
python main_fuse_classcolor.py   --obj converted/Tile_+1990_+2691_L2.obj   --renders ./renders   --masks_dir ./dino_dets/masks_labeled   --out_labels face_class_strings.npy  --out_colored_obj colored_faces.obj

```


