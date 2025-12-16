import os, sys, time, textwrap, queue, threading
from pathlib import Path
import cv2
import numpy as np
import torch

# --- Configuration ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Uncomment to force GPU 1

# --- Threaded Video Reader ---
class ThreadedVideoReader(threading.Thread):
    def __init__(self, path, queue_obj, target_size, stride=1):
        super().__init__()
        self.cap = cv2.VideoCapture(str(path))
        self.queue = queue_obj
        self.target_h, self.target_w = target_size
        self.stride = stride
        self.stopped = False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def run(self):
        frame_idx = 0
        while not self.stopped:
            if self.queue.full():
                time.sleep(0.01)
                continue     
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            processed_frame = None
            if frame_idx % self.stride == 0:
                fr = cv2.resize(frame, (self.target_w, self.target_h))
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                processed_frame = fr

            self.queue.put((frame, processed_frame))
            frame_idx += 1
        self.cap.release()

    def stop(self): self.stopped = True

# --- Threaded Video Writer ---
class ThreadedVideoWriter(threading.Thread):
    def __init__(self, path, queue_obj, fps, width, height):
        super().__init__()
        self.out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        self.queue = queue_obj
        self.stopped = False

    def run(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.out.write(frame)
                self.queue.task_done()
            except queue.Empty: continue
        self.out.release()

    def stop(self): self.stopped = True

# --- Text Helper ---
def clean_and_wrap_text(raw_text, width=60):
    while isinstance(raw_text, list):
        if len(raw_text) > 0: raw_text = raw_text[0]
        else: raw_text = ""
    if hasattr(raw_text, 'decode'): raw_text = raw_text.decode("utf-8", "ignore")
    text = str(raw_text).replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    if "assistant" in text: text = text.split("assistant")[-1]
    return textwrap.wrap(text.strip(), width=width), text.strip()

# --- Main Logic ---
def run_final_visualization(video_path, model, processor, tmpl, out_path, stride=3):
    video_path = Path(video_path)
    
    read_queue = queue.Queue(maxsize=128) 
    write_queue = queue.Queue(maxsize=128)

    T1, T2, C, tH, tW = tmpl["image_frames"].shape
    reader = ThreadedVideoReader(video_path, read_queue, (tH, tW), stride=stride)
    reader.start()
    
    writer = ThreadedVideoWriter(out_path, write_queue, reader.fps, reader.width, reader.height)
    writer.start()

    print(f"Processing: {video_path.name}")
    print("="*60)

    cruise_xyz = torch.zeros_like(tmpl["ego_history_xyz"])
    cruise_rot = torch.zeros_like(tmpl["ego_history_rot"])
    cruise_rot[..., 0] = 1.0
    for i in range(cruise_xyz.shape[1]):
        cruise_xyz[0, i, 0] = 10.0 * ((i - 20) * 0.1)

    frames_buffer = []      
    model_inputs = []       
    BATCH_SIZE = int(T1*T2)
    prev_time = time.time()
    curr_fps = 0.0

    try:
        current_text_lines = ["Initializing..."]
        
        while True:
            if reader.stopped and read_queue.empty(): break
            try:
                raw_frame, model_frame = read_queue.get(timeout=1.0)
            except queue.Empty: continue

            frames_buffer.append(raw_frame)
            if model_frame is not None:
                model_inputs.append(model_frame)

            # --- INFERENCE ---
            if len(model_inputs) >= BATCH_SIZE:
                tensor = torch.from_numpy(np.stack(model_inputs)).permute(0, 3, 1, 2).unsqueeze(0)
                messages = helper.create_message(tensor[0])
                instruction_text = "Output the chain-of-thought reasoning."
                if isinstance(messages[0]["content"], list):
                    messages[0]["content"].append({"type": "text", "text": instruction_text})
                else: messages[0]["content"] = instruction_text

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

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=inputs_data, top_p=0.8, temperature=0.6, 
                        num_traj_samples=1, max_generation_length=256, return_extra=True
                    )
                
                # --- PRINTING TO TERMINAL ---
                raw_cot = extra.get("cot", [""])[0]
                current_text_lines, full_text = clean_and_wrap_text(raw_cot, width=65)
                
                # Force print to terminal (Clear line first to avoid messing up FPS bar)
                sys.stdout.write("\r" + " "*80 + "\r") 
                print(f"\n[AI THOUGHT]: {full_text}\n")
                print("-" * 60)
                
                model_inputs = []

            # --- WRITING & FPS ---
            if len(frames_buffer) >= (BATCH_SIZE * stride): 
                curr_time = time.time()
                if (curr_time - prev_time) > 0:
                    curr_fps = len(frames_buffer) / (curr_time - prev_time)
                prev_time = curr_time

                # Show FPS bar at bottom
                sys.stdout.write(f"\r>> Encoding Video... Current Speed: {curr_fps:.1f} FPS")
                sys.stdout.flush()

                for fr in frames_buffer:
                    banner_height = 50 + (len(current_text_lines) * 35)
                    cv2.rectangle(fr, (0, 0), (reader.width, banner_height), (0, 0, 0), -1)
                    cv2.putText(fr, f"FPS: {curr_fps:.1f}", (reader.width - 250, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    cv2.putText(fr, "ALPAYMAO REASONING:", (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    y = 80
                    for line in current_text_lines:
                        cv2.putText(fr, line, (20, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        y += 35
                    write_queue.put(fr)
                frames_buffer = []

    finally:
        print("\nStopping threads...")
        reader.stop()
        writer.stop()
        reader.join()
        writer.join()
        print(f"Done! Saved video to: {out_path}")

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
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float16

    print(f"Loading Model on {DEVICE}...")
    model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    TEMPLATE_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    tmpl = load_physical_aiavdataset(TEMPLATE_CLIP_ID, t0_us=5_100_000)

    run_final_visualization(
        video_path=Path("/home/wesleyferreiramaia/data/workzone/data/Construction_Data/EYosemiteAve_NightRain.mp4"),
        model=model,
        processor=processor,
        tmpl=tmpl,
        out_path=Path("/home/wesleyferreiramaia/data/workzone/data/Construction_Data/EYosemiteAve_NightRain_verbose.mp4"),
    )