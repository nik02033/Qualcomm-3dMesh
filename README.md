# Qualcomm-3dMesh
# Download the prepacked environment

```bash
wget -O grounded-sam.tar https://arizonastateu-my.sharepoint.com/personal/sanaik3_sundevils_asu_edu/_layouts/15/download.aspx?share=ETac6mlFd9xIuz1j6sSRBqEBwgvUrO1eg9v8esHd45gbKA
```

# Run commands to generate renderings, inference using FastSAM and backprojecting onto the mesh. Sample usage for Tile_+1984_+2688_L2.obj:

```bash
python render_qualcomm.py --input converted/Tile_+1984_+2688_L2.obj   --output_dir renders --num_views 9 --image_size 1024 --meshlab_lighting --elev_deg 38 --zoom 1.1

python main_infer.py 

python3 main_fuse.py   --obj converted/Tile_+1984_+2688_L2.obj   --renders_dir renders   --masks_dir masks   --image_size 1024   --out_labels face_labels.npy   --out_colored_obj colored.obj
```