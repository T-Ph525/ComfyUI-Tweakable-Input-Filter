import torch
from PIL import Image
from transformers import pipeline
from comfy.model_base import BaseNode
import cv2
import numpy as np
import os

class IllegalContentScanner(BaseNode):
    @classmethod
    def INPUT_TYPES(s):
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

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("flagged",)
    FUNCTION = "run"

    def __init__(self):
        self.nsfw_detector = pipeline("image-classification", model="models/nsfw")
        self.age_detector = pipeline("image-classification", model="models/age")

    def run(self, image, prompt, video_path, nsfw_threshold, age_threshold, scan_video):
        def to_pil(tensor_img):
            return Image.fromarray((tensor_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))

        def is_illegal(pil_image):
            if nsfw_threshold > 0:
                nsfw = self.nsfw_detector(pil_image)[0]
                if nsfw["label"].lower() != "nsfw" or nsfw["score"] < nsfw_threshold:
                    return False

            age = self.age_detector(pil_image)[0]
            underage = age["label"].lower() in ["child", "teen"] and age["score"] > age_threshold
            return underage

        # Keyword prompt scan
        flagged_prompt = any(k in prompt.lower() for k in [
            "13 year old", "14 year old", "minor", "teen girl", "young boy", "schoolgirl", "loli", "child"
        ])
        if flagged_prompt:
            raise PermissionError("🚫 403 Forbidden: Suspicious prompt detected")

        # Image scan
        if isinstance(image, torch.Tensor):
            pil_img = to_pil(image)
            if is_illegal(pil_img):
                raise PermissionError("🚫 403 Forbidden: Underage content detected in image")

        # Optional video scan
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
                    if is_illegal(pil_frame):
                        cap.release()
                        raise PermissionError(f"🚫 403 Forbidden: Underage NSFW frame detected at frame {frame_idx}")
                frame_idx += 1
            cap.release()

        return (False,)
