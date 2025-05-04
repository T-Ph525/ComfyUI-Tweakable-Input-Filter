import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROMPT_KEYWORDS = [
    "13 year old", "14 year old", "minor", "teen girl",
    "young boy", "schoolgirl", "loli", "child"
]

class IllegalContentScanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING",),
                "video_path": ("STRING", {"multiline": False}),
                "nsfw_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "age_threshold": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),
                "scan_video": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("flagged", "reason")
    FUNCTION = "run"
    CATEGORY = "Content Moderation"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nsfw_path = "models/nsfw"
        self.nsfw_processor = AutoImageProcessor.from_pretrained(nsfw_path)
        self.nsfw_model = AutoModelForImageClassification.from_pretrained(nsfw_path).to(self.device)
        self.nsfw_labels = self.nsfw_model.config.id2label

        age_path = "models/age"
        self.age_processor = AutoImageProcessor.from_pretrained(age_path)
        self.age_model = AutoModelForImageClassification.from_pretrained(age_path).to(self.device)
        self.age_labels = self.age_model.config.id2label

    def predict(self, image, processor, model, labels):
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        label_idx = scores.argmax().item()
        return labels[label_idx].lower(), scores[label_idx].item()

    def is_illegal(self, pil_image, nsfw_threshold, age_threshold):
        label, score = self.predict(pil_image, self.nsfw_processor, self.nsfw_model, self.nsfw_labels)
        if label == "nsfw" and score >= nsfw_threshold:
            age_label, age_score = self.predict(pil_image, self.age_processor, self.age_model, self.age_labels)
            if age_label in ["child", "teen"] and age_score > age_threshold:
                return True, f"Underage NSFW detected (age: {age_label}, confidence: {age_score:.2f})"
        return False, ""

    def run(self, image, prompt, video_path, nsfw_threshold, age_threshold, scan_video):
        if any(k in prompt.lower() for k in PROMPT_KEYWORDS):
            return (True, "🚫 Suspicious prompt: possible underage content")

        if isinstance(image, torch.Tensor):
            pil_img = Image.fromarray((image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
            flagged, reason = self.is_illegal(pil_img, nsfw_threshold, age_threshold)
            if flagged:
                return (True, reason)

        if scan_video and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate = max(total_frames // 10, 1)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_rate == 0:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(img_rgb)
                    flagged, reason = self.is_illegal(pil_frame, nsfw_threshold, age_threshold)
                    if flagged:
                        cap.release()
                        return (True, f"{reason} in frame {frame_idx}")
                frame_idx += 1
            cap.release()

        return (False, "✅ Passed all checks")
