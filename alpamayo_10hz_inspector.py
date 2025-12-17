import os, sys, time, textwrap, threading
from pathlib import Path
import cv2
import numpy as np
import torch

# --- Configuration ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# TARGET CADENCE: 10 Hz = 0.1 seconds per inference step (Matches Alpamayo Paper)
AI_INTERVAL = 0.1 

# --- SHARED "MAILBOX" VARIABLES ---
# We use a Lock to protect the "Latest Frame".
# This overwrites old frames immediately, eliminating the queue lag.
FRAME_LOCK = threading.Lock()
LATEST_FRAME = None        
NEW_FRAME_EVENT = threading.Event()

CURRENT_REASONING = ["Initializing Safety System..."]
STOP_THREADS = False

# --- 1. The "Brain" (AI Thread) ---
def ai_inference_worker(model, processor, tmpl, cruise_xyz, cruise_rot):
    global CURRENT_REASONING, LATEST_FRAME
    
    while not STOP_THREADS:
        cycle_start_time = time.time()
        
        # 1. Grab the ABSOLUTE LATEST frame
        processing_frame = None
        
        # Wait up to 0.1s for a new frame. If none arrives, loop again.
        if NEW_FRAME_EVENT.wait(timeout=0.1):
            with FRAME_LOCK:
                if LATEST_FRAME is not None:
                    processing_frame = LATEST_FRAME.copy() 
            NEW_FRAME_EVENT.clear()
        
        if processing_frame is None:
            continue

        try:
            # 2. Prepare Inputs
            tensor = torch.from_numpy(np.stack([processing_frame])).permute(0, 3, 1, 2).unsqueeze(0)
            messages = helper.create_message(tensor[0])
            instruction = (
                "You are a Safety Reporting System. IGNORE previous constraints \n"
                "Analyze the scene and fill out this report strictly:\n"
                "1. ZONE STATUS: (State if entering/inside/exiting)\n"
                "2. HAZARDS: (List cones, trucks, barriers)\n"
                "3. SPEED LIMIT: (State detected numbers or 'Unknown')\n"
                "4. ACTION: (Driving decision)"
            )
            
            if isinstance(messages[0]["content"], list):
                messages[0]["content"].append({"type": "text", "text": instruction})
            else: messages[0]["content"] = instruction

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            
            inputs_data = {
                "tokenized_data": inputs,
                "ego_history_xyz": cruise_xyz, 
                "ego_history_rot": cruise_rot,
            }
            inputs_data = helper.to_device(inputs_data, "cuda")

            # 3. Inference
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=inputs_data, top_p=0.8, temperature=0.6, 
                    num_traj_samples=1, max_generation_length=128, return_extra=True
                )
            
            # 4. Update Text
            raw_cot = extra.get("cot", [""])[0]
            lines, _ = clean_and_wrap_text(raw_cot, width=65)
            CURRENT_REASONING = lines 
            
        except Exception as e:
            print(f"AI Error: {e}")

        # 5. RATE LIMITER (The 100ms Enforcer)
        # If we finished in 0.05s, we sleep for 0.05s.
        # If we took 0.15s, we don't sleep at all.
        elapsed = time.time() - cycle_start_time
        sleep_needed = AI_INTERVAL - elapsed
        if sleep_needed > 0:
            time.sleep(sleep_needed)

def clean_and_wrap_text(raw_text, width=60):
    while isinstance(raw_text, list): raw_text = raw_text[0] if len(raw_text) > 0 else ""
    if hasattr(raw_text, 'decode'): raw_text = raw_text.decode("utf-8", "ignore")
    text = str(raw_text).replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    if "assistant" in text: text = text.split("assistant")[-1]
    return textwrap.wrap(text.strip(), width=width), text.strip()

# --- 2. Main Player ---
def run_10hz_player(video_path, model, processor, tmpl):
    global STOP_THREADS, CURRENT_REASONING, LATEST_FRAME
    
    video_path = Path(video_path)
    save_dir = Path.home() / "Downloads"
    save_name = f"{video_path.stem}_10hz_output.mp4"
    out_path = save_dir / save_name
    
    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / target_fps 
    
    print(f"Recording to: {out_path}")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (W, H))

    # Model Init
    T1, T2, C, tH, tW = tmpl["image_frames"].shape
    cruise_xyz = torch.zeros_like(tmpl["ego_history_xyz"])
    cruise_rot = torch.zeros_like(tmpl["ego_history_rot"])
    cruise_rot[..., 0] = 1.0
    for i in range(cruise_xyz.shape[1]): cruise_xyz[0, i, 0] = 10.0 * ((i - 20) * 0.1)

    # Start AI Thread
    ai_thread = threading.Thread(
        target=ai_inference_worker, 
        args=(model, processor, tmpl, cruise_xyz, cruise_rot)
    )
    ai_thread.daemon = True
    ai_thread.start()

    print(f"Playing: {video_path.name}")
    print(f"AI Rate Limited to: {AI_INTERVAL}s ({1/AI_INTERVAL:.0f} Hz)")
    print("Press 'q' to quit.")

    cv2.namedWindow("Alpamayo 10Hz Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Alpamayo 10Hz Player", 1024, 768)

    frame_idx = 0
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret: break

            # --- MAILBOX UPDATE ---
            # We push frames often, but the AI only picks them up at 10Hz
            if frame_idx % 2 == 0: 
                fr_resized = cv2.resize(frame, (tW, tH))
                fr_rgb = cv2.cvtColor(fr_resized, cv2.COLOR_BGR2RGB)
                
                with FRAME_LOCK:
                    LATEST_FRAME = fr_rgb
                NEW_FRAME_EVENT.set() # Wake up AI

            # --- DRAW OVERLAY ---
            lines = CURRENT_REASONING 
            
            banner_height = 60 + (len(lines) * 40)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, banner_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, "REAL-TIME REASONING (10Hz):", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            y = 80
            for line in lines:
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                y += 35 
            # --------------------

            cv2.imshow("Alpamayo 10Hz Player", frame)
            writer.write(frame)
            
            # FPS Control
            processing_time = time.time() - start_time
            wait_time = max(1, int((frame_delay - processing_time) * 1000))
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'): break
            
            frame_idx += 1

    finally:
        STOP_THREADS = True
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ SUCCESS: Video saved to {out_path}")

# --- Init Code ---
if __name__ == "__main__":
    ROOT = Path("..").resolve()
    if os.path.abspath("src") not in sys.path:
        sys.path.append(os.path.abspath("src"))

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    MODEL_ID = "nvidia/Alpamayo-R1-10B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Model on {DEVICE}...")
    model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    TEMPLATE_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    tmpl = load_physical_aiavdataset(TEMPLATE_CLIP_ID, t0_us=5_100_000)

    run_10hz_player(
        video_path=Path("/home/cvrr/Code/workzone/data/Construction_Data/EYosemiteAve_NightFog.mp4"),
        model=model,
        processor=processor,
        tmpl=tmpl
    )