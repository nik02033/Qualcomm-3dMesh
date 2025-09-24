import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram_plus
from ram import inference_ram
import torchvision.transforms as TS

# ChatGPT or nltk is required when using tags_chineses
# import openai
# import nltk

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def check_tags_chinese(tags_chinese, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    # object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_list = [obj.rsplit('(', 1)[0].strip() for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the tags_chinese if it is wrong. ' + \
                           f'tags_chinese: {tags_chinese}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised tags_chinese: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "tags_chinese: xxx, xxx, xxx"
        tags_chinese = reply.split(':')[-1].strip()
    return tags_chinese


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    final_boxes = []
    for logit, box in zip(logits_filt, boxes_filt):        
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if(pred_phrase.strip() == ""):
            continue
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())
        final_boxes.append(box)
    boxes_filt = torch.stack(final_boxes)

    return boxes_filt, torch.Tensor(scores), pred_phrases

def get_single_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]              # (nq, 4)

    # filter by box threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    scores = []

    for logit, box in zip(logits_filt, boxes_filt):
        # get index of highest scoring token
        best_idx = logit.argmax().item()
        best_score = logit[best_idx].item()
        best_token = tokenizer.decode([tokenized.input_ids[best_idx]])

        pred_phrases.append(f"{best_token}({best_score:.3f})")
        scores.append(best_score)

    return boxes_filt, torch.Tensor(scores), pred_phrases



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)
    
    
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"

ram_checkpoint = "ram_plus_swin_large_14m.pth"  # change the path of the model
ram_model = ram_plus(pretrained=ram_checkpoint,
                                    image_size=384,
                                    vit='swin_l')
ram_model = ram_model.to(device)
ram_model.eval()

sam_checkpoint = "sam_vit_h_4b8939.pth"
predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

def infer(output_dir, image_path, img_name, box_threshold=0.05, text_threshold=0.05, iou_threshold=0.25):

    # cfg
    config_file = "GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    
    grounded_checkpoint = "groundingdino_swint_ogc.pth"  # change the path of the model
    
    sam_hq_checkpoint = None
    use_sam_hq = False
    split = ","  # split for test prompt
    openai_key = None
    openai_proxy = None
    
    
    # ChatGPT or nltk is required when using tags_chineses
    # openai.api_key = openai_key
    # if openai_proxy:
        # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    

    # threshold for tagging
    # we reduce the threshold to obtain more tags
    
    raw_image = image_pil.resize(
                    (384, 384))
    raw_image  = transform(raw_image).unsqueeze(0).to(device)

    res = inference_ram(raw_image , ram_model)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    tags_list = res[0].split("|")
    print(tags_list)
    tags = []
    for tag in tags_list:
        if not ("city" in tag or "scale" in tag or "model" in tag or "urban" in tag):
            tags.append(tag.strip())
    tags = ", ".join(tags)
    tags_chinese=res[1].replace(' |', ',')

    print("Image Tags: ", tags)
    print("图像标签: ", res[1])

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, tags, box_threshold, text_threshold, device=device
    )

    # # initialize SAM
    # if use_sam_hq:
    #     print("Initialize SAM-HQ Predictor")
    #     predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    # else:
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    # tags_chinese = check_tags_chinese(tags_chinese, pred_phrases)
    # print(f"Revise tags_chinese with number: {tags_chinese}")

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    # # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    plt.gca().set_facecolor("white")  
    plt.gcf().patch.set_facecolor("white")
    
    print("Masks Shape", masks.squeeze().shape)
    # masks = masks.detach().cpu().numpy()
    np.save(f"{output_dir}/masks_{img_name}.npy", masks.detach().cpu().numpy().squeeze())
    
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        break
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
        break

    os.makedirs(f"{output_dir}/{img_name}", exist_ok=True)

    plt.title('RAM-tags: ' + tags + '\n')
    plt.axis('off')
    plt.savefig(
        f"{output_dir}/{img_name}/output.jpg", 
        bbox_inches="tight", dpi=300, pad_inches=0.0, facecolor="white"
    )

    save_mask_data(f"{output_dir}/{img_name}", tags_chinese, masks, boxes_filt, pred_phrases)
