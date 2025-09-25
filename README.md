# Qualcomm-3dMesh

Download the `groundingdino_swint_ogc.pth` file into the `GroundingDINO/weights` directory with:

```bash
wget -P GroundingDINO/weights \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Run commands to generate renderings, inference using FastSAM and backprojecting onto the mesh. Sample usage for Tile_+1984_+2688_L2.obj:
```bash
python render_qualcomm.py --input converted/Tile_+1984_+2688_L2.obj   --output_dir renders --num_views 9 --image_size 1024 --meshlab_lighting --elev_deg 38 --zoom 0.5

python main_sam.py

python gdino_infer.py

python mask_voting.py

python main_fuse_classcolor.py   --obj converted/Tile_+1984_+2688_L2.obj   --renders ./renders   --masks_dir ./dino_dets/masks_labeled   --out_labels face_class_strings.npy  --out_colored_obj colored_faces.obj

```
