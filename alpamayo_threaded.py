import os, sys, time, textwrap, queue, threading
from pathlib import Path
import cv2
import numpy as np
import torch

# --- Configuration ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force GPU 1 (physically second GPU) if you want to avoid the busy one
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# --- Threaded Video Reader ---
class ThreadedVideoReader(threading.Thread):
    def __init__(self, path, queue_obj, target_size, stride=1):
        super().__init__()
        self.cap = cv2.VideoCapture(str(path))
        self.queue = queue_obj
        self.target_h, self.target_w = target_size
        self.stride = stride
        self.stopped = False
        
        # Get video properties for the writer
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        frame_idx = 0
        while not self.stopped:
            # If queue is full, wait a bit to prevent RAM explosion
            if self.queue.full():
                time.sleep(0.01)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
                
            # Only keep every nth frame (stride) but we might need all for video writing?
            # Your original code read ALL frames but only SAMPLED every 3rd for the model.
            # To keep it simple and sync video output, we pass every frame.
            
            # Pre-processing on CPU (utilizing one of your 28 cores)
            processed_frame = None
            if frame_idx % self.stride == 0:
                fr = cv2.resize(frame, (self.target_w, self.target_h))
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                processed_frame = fr # This is for the Model

            # Put (original_frame, model_input_frame_or_None) in queue
            self.queue.put((frame, processed_frame))
            frame_idx += 1
            
        self.cap.release()

    def stop(self):
        self.stopped = True

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
            except queue.Empty:
                continue
        self.out.release()

    def stop(self):
        self.stopped = True

# --- Helper Functions ---
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
    
    # 1. Setup Queues
    # Limit size to prevent filling RAM if CPU is faster than GPU
    read_queue = queue.Queue(maxsize=128) 
    write_queue = queue.Queue(maxsize=128)

    # 2. Start Reader Thread
    T1, T2, C, tH, tW = tmpl["image_frames"].shape
    reader = ThreadedVideoReader(video_path, read_queue, (tH, tW), stride=stride)
    reader.start()
    
    # 3. Start Writer Thread
    writer = ThreadedVideoWriter(out_path, write_queue, reader.fps, reader.width, reader.height)
    writer.start()

    print(f"Processing: {video_path.name} | utilizing background threads...")

    # Cruise control dummy data
    cruise_xyz = torch.zeros_like(tmpl["ego_history_xyz"])
    cruise_rot = torch.zeros_like(tmpl["ego_history_rot"])
    cruise_rot[..., 0] = 1.0
    for i in range(cruise_xyz.shape[1]):
        cruise_xyz[0, i, 0] = 10.0 * ((i - 20) * 0.1)

    # Batch variables
    frames_buffer = []      # Raw BGR frames to be written
    model_inputs = []       # RGB frames for the model
    BATCH_SIZE = int(T1*T2) # How many frames the model needs at once

    try:
        current_text_lines = ["Initializing..."]
        
        while True:
            # Get data from Reader (this is fast now!)
            if reader.stopped and read_queue.empty():
                break
                
            try:
                raw_frame, model_frame = read_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frames_buffer.append(raw_frame)
            if model_frame is not None:
                model_inputs.append(model_frame)

            # Check if we have enough frames to run inference
            if len(model_inputs) >= BATCH_SIZE:
                # --- GPU INFERENCE START ---
                tensor = torch.from_numpy(np.stack(model_inputs))
                tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)

                messages = helper.create_message(tensor[0])
                instruction_text = "Output the chain-of-thought reasoning of the driving process."
                if isinstance(messages[0]["content"], list):
                    messages[0]["content"].append({"type": "text", "text": instruction_text})
                else:
                    messages[0]["content"] = instruction_text

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
                # --- GPU INFERENCE END ---

                # Update text for the *next* batch of video frames
                raw_cot = extra.get("cot", [""])[0]
                current_text_lines, _ = clean_and_wrap_text(raw_cot, width=65)
                print(f"Thought: {current_text_lines[0]}...")

                # Clear model buffer
                model_inputs = []

            # Draw text on frames and push to Writer
            # (Note: We draw the *latest* known reasoning on every frame)
            # This logic mimics your original loop where frames accumulated, then bulk processed.
            # To keep video smooth, we write frames as we go, using the last known text.
            
            # Since your original code accumulated frames_bgr and wrote them AFTER inference,
            # we need to mimic that behavior to sync text with the specific clip.
            # Here, we only push to writer when the batch is done.
            if len(frames_buffer) >= (BATCH_SIZE * stride): 
                for fr in frames_buffer:
                    banner_height = 50 + (len(current_text_lines) * 35)
                    cv2.rectangle(fr, (0, 0), (reader.width, banner_height), (0, 0, 0), -1)
                    cv2.putText(fr, "ALPAYMAO REASONING:", (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    y = 80
                    for line in current_text_lines:
                        cv2.putText(fr, line, (20, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        y += 35
                    write_queue.put(fr)
                
                frames_buffer = [] # Reset buffer

    finally:
        reader.stop()
        writer.stop()
        reader.join()
        writer.join()
        print(f"Saved: {out_path}")

# --- Init Code ---
if __name__ == "__main__":
    ROOT = Path("..").resolve()
    # Ensure src is in path
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

    # Load template
    TEMPLATE_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    tmpl = load_physical_aiavdataset(TEMPLATE_CLIP_ID, t0_us=5_100_000)

    run_final_visualization(
        video_path=Path("/home/wesleyferreiramaia/data/workzone/data/demo/jacksonville.mp4"),
        model=model,
        processor=processor,
        tmpl=tmpl,
        out_path=Path("jacksonville_optimized.mp4"),
    )