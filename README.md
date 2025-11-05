# Qualcomm-3dMesh
export PATH="/home/nkanodi1/.conda/envs/sam2/bin:$PATH"
module load gcc-10.3.0-gcc-11.2.0  
export GROUNDINGDINO_WEIGHTS=/scratch/nkanodi1/Capstone/Qualcomm-3dMesh/GroundingDINO/weights/groundingdino_swint_ogc.pth
Download the `groundingdino_swint_ogc.pth` file into the `GroundingDINO/weights` directory with:

```bash
wget -P GroundingDINO/weights \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Run commands to generate renderings, inference using FastSAM and backprojecting onto the mesh. Sample usage for Tile_+1984_+2688_L2.obj:
```bash
python render_qualcomm_pytorch3d.py   --input converted/Tile_+1990_+2691_L2.obj   --image_size 1024   --elev_deg 38   --zoom 0.5   --num_coarse_views 9   --num_fine_views 3   --fine_zoom 0.85   --backoff 1.5   --z_down_frac 0.10

python collect_ram_all_classes.py --neg neg.txt --ra-dir recognize-anything --image-dir renders --checkpoint recognize-anything/pretrained/ram_plus_swin_large_14m.pth --out custom_classes.txt

python main_sam.py

python gdino_infer.py

python mask_voting.py

python main_fuse_classcolor_pytorch3d.py   --obj converted/Tile_+1990_+2691_L2.obj   --renders ./renders   --masks_dir ./dino_dets/masks_labeled   --out_labels face_class_strings.npy  --out_colored_obj colored_faces.obj

```
