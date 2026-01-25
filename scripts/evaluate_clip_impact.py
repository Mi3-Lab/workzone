#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import numpy as np
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import open_clip
from PIL import Image
from tqdm import tqdm

# Constants (Match jetson_app.py)
CHANNELIZATION = {"Cone", "Drum", "Barricade", "Barrier", "Vertical Panel", "Tubular Marker", "Fence"}
WORKERS = {"Worker", "Police Officer"}
VEHICLES = {"Work Vehicle", "Police Vehicle"}
MESSAGE_BOARD = {"Temporary Traffic Control Message Board", "Arrow Board"}
TTC_SIGNS = {"Temporary Traffic Control Sign"}

CUE_PROMPTS = {
    "channelization": {
        "pos": ["traffic cone on road", "orange construction barrel on asphalt", "striped barricade on road", "road barrier", "vertical panel marker"],
        "neg": ["tree trunk", "street light pole", "mailbox", "pedestrian", "car wheel", "fire hydrant", "electricity pole", "bush"],
        "inactive": ["traffic cones stacked on a truck bed", "cones stored in a pile", "construction barrels on a trailer", "equipment in storage yard"]
    },
    "workers": {
        "pos": ["construction worker in high-visibility safety vest", "person wearing hard hat and safety gear", "road worker flagging traffic"],
        "neg": ["pedestrian in casual clothes", "business person in suit", "runner", "cyclist", "mannequin", "statue"]
    },
    "vehicles": {
        "pos": ["yellow construction excavator", "dump truck on road", "pickup truck with flashing amber lights", "road roller", "utility work truck"],
        "neg": ["sedan car", "family suv", "sports car", "motorcycle", "city bus", "taxi"]
    },
    "ttc_signs": {
        "pos": ["orange diamond construction sign facing camera", "road work ahead sign", "speed limit sign facing camera", "white rectangular regulatory sign"],
        "neg": ["commercial billboard advertisement", "shop sign", "street name sign", "parking sign", "restaurant sign"],
        "inactive": ["back of a road sign", "grey metal sign back", "sign facing away", "oblique sign edge"]
    },
    "message_board": {
        "pos": ["electronic arrow board trailer with lights on", "variable message sign displaying text", "digital traffic sign"],
        "neg": ["parked cargo trailer", "billboard", "back of a truck", "container"],
        "inactive": ["message board turned off", "black screen message board", "folded arrow board"]
    }
}

def get_cue_category(name):
    if name in CHANNELIZATION: return "channelization"
    if name in WORKERS: return "workers"
    if name in VEHICLES: return "vehicles"
    if name.startswith("Temporary Traffic Control Sign"): return "ttc_signs"
    if name in MESSAGE_BOARD: return "message_board"
    return None

class PerCueVerifier:
    def __init__(self, clip_bundle, device):
        self.clip = clip_bundle
        self.device = device
        self.embeddings = {}
        self.use_fp16 = True
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        tokenizer = self.clip["tokenizer"]
        model = self.clip["model"]
        for category, prompts in CUE_PROMPTS.items():
            pos_toks = tokenizer(prompts["pos"]).to(self.device)
            neg_toks = tokenizer(prompts["neg"]).to(self.device)
            with torch.no_grad():
                pos_emb = model.encode_text(pos_toks)
                pos_emb /= pos_emb.norm(dim=-1, keepdim=True)
                pos_mean = pos_emb.mean(dim=0)
                pos_mean /= pos_mean.norm()
                
                neg_emb = model.encode_text(neg_toks)
                neg_emb /= neg_emb.norm(dim=-1, keepdim=True)
                neg_mean = neg_emb.mean(dim=0)
                neg_mean /= neg_mean.norm()
                
                inactive_mean = None
                if "inactive" in prompts:
                    inact_toks = tokenizer(prompts["inactive"]).to(self.device)
                    inact_emb = model.encode_text(inact_toks)
                    inact_emb /= inact_emb.norm(dim=-1, keepdim=True)
                    inactive_mean = inact_emb.mean(dim=0)
                    inactive_mean /= inactive_mean.norm()
                
                self.embeddings[category] = (pos_mean, neg_mean, inactive_mean)

    def verify_single(self, crop_bgr, category):
        # Resize/Preprocess
        resized = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_input = self.clip["preprocess"](pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=self.use_fp16):
            img_emb = self.clip["model"].encode_image(img_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            img_emb = img_emb.squeeze()
            
            pos_emb, neg_emb, inactive_emb = self.embeddings[category]
            sim_pos = float(torch.dot(img_emb, pos_emb))
            sim_neg = float(torch.dot(img_emb, neg_emb))
            
            reject_score = sim_neg
            sim_inactive = -1.0
            if inactive_emb is not None:
                sim_inactive = float(torch.dot(img_emb, inactive_emb))
                if sim_inactive > sim_pos:
                    return -1.0, sim_pos, sim_neg, sim_inactive # Hard reject
                reject_score = max(sim_neg, sim_inactive)
            
            return sim_pos - reject_score, sim_pos, sim_neg, sim_inactive

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate_clip_impact.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    # Load Config
    with open("configs/jetson_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load YOLO
    print("Loading YOLO...")
    model_path = config['model']['path']
    if Path(model_path).with_suffix('.engine').exists():
        model_path = str(Path(model_path).with_suffix('.engine'))
    model = YOLO(model_path, task='detect')
    
    # Load CLIP
    print("Loading CLIP...")
    m_c, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", cache_dir="weights/clip")
    clip_bundle = {"model": m_c.to(device).eval(), "preprocess": prep, "tokenizer": open_clip.get_tokenizer("ViT-B-32")}
    verifier = PerCueVerifier(clip_bundle, device)
    
    # Setup Output Dirs
    out_dir = Path("results/debug_clip")
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "accepted").mkdir()
    (out_dir / "rejected").mkdir()
    
    # Process Video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} ({total_frames} frames)...")
    
    accepted_count = 0
    rejected_count = 0
    
    for i in tqdm(range(min(total_frames, 300))): # Limit to first 300 frames for speed
        ret, frame = cap.read()
        if not ret: break
        
        # YOLO Detect
        results = model.predict(frame, conf=0.25, verbose=False)[0]
        
        if not results.boxes: continue
        
        h_img, w_img = frame.shape[:2]
        
        for j, box in enumerate(results.boxes):
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = float(box.conf.cpu().numpy()[0])
            cid = int(box.cls.cpu().numpy()[0])
            name = model.names[cid]
            cat = get_cue_category(name)
            
            if not cat: continue
            
            # Extract Crop
            x1, y1, x2, y2 = map(int, xyxy)
            pad = 10
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w_img, x2+pad), min(h_img, y2+pad)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0: continue
            
            # CLIP Verify
            score, s_pos, s_neg, s_inact = verifier.verify_single(crop, cat)
            
            threshold = 0.05
            status = "accepted" if score > threshold else "rejected"
            
            # Save Debug Image
            fname = f"frame{i:04d}_obj{j}_{cat}_{status}_score{score:.2f}.jpg"
            dest = out_dir / status / fname
            
            # Annotate crop with scores
            debug_crop = crop.copy()
            # Add black border at top for text
            border_size = 40
            debug_crop = cv2.copyMakeBorder(debug_crop, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            cv2.putText(debug_crop, f"S:{score:.2f} P:{s_pos:.2f}", (5, 15), 0, 0.4, (255,255,255), 1)
            cv2.putText(debug_crop, f"N:{s_neg:.2f} I:{s_inact:.2f}", (5, 30), 0, 0.4, (255,255,255), 1)
            
            cv2.imwrite(str(dest), debug_crop)
            
            if status == "accepted": accepted_count += 1
            else: rejected_count += 1
            
    print(f"\nAnalysis Complete.")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Check the folder '{out_dir}' to see the crops.")

if __name__ == "__main__":
    main()
