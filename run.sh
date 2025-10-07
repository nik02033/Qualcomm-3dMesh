#!/usr/bin/env bash

echo "=== Step 1: render_qualcomm.py ==="
SECONDS=0
python render_qualcomm.py \
    --input converted/Tile_+1984_+2688_L2.obj \
    --output_dir renders \
    --num_views 4 \
    --image_size 1024 \
    --meshlab_lighting \
    --elev_deg 20 \
    --zoom 0.6
echo "Step 1 took ${SECONDS}s"
echo

echo "=== Step 2: main_infer.py ==="
SECONDS=0
python main_infer.py
echo "Step 2 took ${SECONDS}s"
echo

echo "=== Step 3: main_fuse.py ==="
SECONDS=0
python3 main_fuse.py \
    --obj converted/Tile_+1984_+2688_L2.obj \
    --renders_dir renders \
    --masks_dir masks \
    --image_size 1024 \
    --out_labels face_labels.npy \
    --out_colored_obj colored.obj
echo "Step 3 took ${SECONDS}s"
echo

echo "All steps completed."
