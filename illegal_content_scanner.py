
import torch
from PIL import Image
from transformers import pipeline
from comfy.model_base import BaseNode
import folder_paths

class IllegalContentScanner(BaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING",),
                "nsfw_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "age_threshold": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("flagged",)
    FUNCTION = "run"

    def __init__(self):
        self.nsfw_detector = pipeline("image-classification", model="path/to/nsfw/model")
        self.age_detector = pipeline("image-classification", model="nateraw/vit-age-classifier")

    def run(self, image, prompt, nsfw_threshold, age_threshold):
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))

        nsfw = self.nsfw_detector(image)[0]
        if nsfw["label"].lower() != "nsfw" or nsfw["score"] < nsfw_threshold:
            return (False,)

        age = self.age_detector(image)[0]
        underage = age["label"].lower() in ["child", "teen"] and age["score"] > age_threshold
        prompt_flagged = any(keyword in prompt.lower() for keyword in ["13 year old", "minor", "teen girl", "child", "schoolgirl", "young boy"])

        if underage or prompt_flagged:
            # Raise an error to stop workflow and return 403 response in API mode
            raise PermissionError("🚫 403 Forbidden: Illegal content detected (NSFW + underage or illegal prompt)")

        return (False,)
