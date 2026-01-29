import cv2
import os

video_path = "data/demo/boston.mp4"
output_dir = "results/vlm_test"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

for sec in [10, 30, 50]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"{output_dir}/frame_{sec}s.jpg", frame)
        print(f"Extracted frame at {sec}s")
cap.release()
