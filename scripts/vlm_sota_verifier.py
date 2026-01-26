import cv2
import base64
import requests
import json
import re
import time

class VLMSotaVerifier:
    def __init__(self, model_name="qwen2.5vl:7b", device="cuda"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"
        self.available = self._check_ollama()
        
    def _check_ollama(self):
        try:
            requests.get("http://localhost:11434/api/tags", timeout=1)
            return True
        except:
            return False

    def encode_image(self, frame, target_width=960):
        # 960px is crucial for detection range
        h, w = frame.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
        _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_frame(self, frame):
        if not self.available: return None

        t0 = time.time()
        img_b64 = self.encode_image(frame)
        
        # PROMPT ROBUSTO (RESTAURADO): Foco em Geometria e Posição
        payload = {
            "model": self.model_name,
            "system": """You are a Driver Assistant. Analyze Work Zone State based on OBJECT POSITIONS.

RULES:
1. INSIDE: 
   - Cones/Barrels are ALONGSIDE the car (visible at bottom corners/peripheral).
   - We are driving parallel to the cone line.
   
2. APPROACHING:
   - Cones are visible DISTANT AHEAD (center/horizon).
   - We have NOT reached them yet.
   - You see a "Road Work" sign ahead.

3. OUT: 
   - No cones/barrels visible.

JSON format: { "state": "...", "reasoning": "..." }""",
            "prompt": "Are the cones alongside us (INSIDE) or far ahead (APPROACHING)? Look for ANY traffic cones (Orange/White).",
            "stream": False,
            "images": [img_b64],
            "options": {
                "temperature": 0.0, 
                "num_predict": 100,
                "num_ctx": 4096
            },
            "format": "json"
        }

        try:
            response = requests.post(self.url, json=payload, timeout=20)
            response.raise_for_status()
            res_json = response.json()
            raw_text = res_json.get('response', '')
            total_duration = res_json.get('total_duration', 0) / 1e9
            
            data = self._parse_json(raw_text)
            
            data['latency'] = round(time.time() - t0, 2)
            data['model_latency'] = round(total_duration, 2)
            return data

        except Exception as e:
            print(f"[VLM] Error: {e}")
            return None

    def _parse_json(self, text):
        try:
            text = re.sub(r'<.*?>', '', text)
            text = text.strip().strip('`').strip()
            if text.startswith('json'): text = text[4:]
            data = json.loads(text)
            
            if "state" not in data: 
                if data.get("inside") or data.get("INSIDE"): data["state"] = "INSIDE"
                elif data.get("approaching") or data.get("APPROACHING"): data["state"] = "APPROACHING"
                elif data.get("out") or data.get("OUT"): data["state"] = "OUT"
                else: data["state"] = "UNKNOWN"
            return data
        except:
            return {"state": "UNKNOWN", "reasoning": "Parse Error", "raw": text[:50]}