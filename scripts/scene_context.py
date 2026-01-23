import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

class SceneContextPredictor:
    def __init__(self, model_path, device="cuda"):
        # Handle int device (e.g. 0 -> "cuda:0")
        if isinstance(device, int):
            self.device = f"cuda:{device}"
        else:
            self.device = str(device)
            
        self.classes = ['highway', 'urban', 'suburban', 'mixed']
        self.model = self._load_model(model_path)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"[SceneContext] Loaded model from {model_path}")

    def _load_model(self, path):
        # ResNet18 architecture matching training (use legacy pretrained=False for max compat)
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.classes))
        
        # Load weights
        checkpoint = torch.load(path, map_location=self.device)
        # Handle state dict format if needed
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        # Optimize with half precision if on CUDA
        if self.device == "cuda":
            model.half()
            
        return model

    def predict(self, frame_bgr):
        # Convert to PIL
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Preprocess
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        if self.device == "cuda":
            input_tensor = input_tensor.half()
            
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        scene_class = self.classes[pred.item()]
        confidence = conf.item()
        
        return scene_class, confidence
