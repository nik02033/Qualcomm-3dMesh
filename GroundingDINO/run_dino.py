# run_dino.py
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import numpy as np
import cv2

CFG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS = "weights/groundingdino_swint_ogc.pth"
IMAGE_PATH = "../renders/view_07.png"
TEXT_PROMPT = "building . tree . car . person . boats . river . people . traffic light . street . road . chimney . window . door ."
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

device = "cpu"  # <-- force CPU (same effect as --cpu-only in the demo CLI)

# Load model on CPU
model = load_model(CFG, WEIGHTS, device=device)
model.eval()

# Load image
image_source, image = load_image(IMAGE_PATH)

with torch.no_grad():
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device,  # <-- ensure predict() also stays on CPU
    )

# Draw boxes + labels. annotate() returns a PIL Image; convert to BGR for OpenCV.
annotated_pil = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
annotated_rgb = np.array(annotated_pil)           # RGB uint8 [H,W,3]
annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("annotated_image.jpg", annotated_bgr)
print("Saved annotated_image.jpg")
