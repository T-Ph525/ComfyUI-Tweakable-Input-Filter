import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from fastapi import HTTPException

PROMPT_KEYWORDS = [
    "13 year old", "14 year old", "15 year old", "16 year old", "17 year old",
    "minor", "teen", "teen girl", "teen boy", "underage", "schoolgirl", "school boy",
    "child", "toddler", "preteen", "pubescent", "young teen", "young girl", "young boy",
    "loli", "shota", "lolicon", "shotacon", "babyface", "middle school", "elementary school",
    "kiddy", "kindergarten", "junior high", "innocent girl", "innocent boy", "girl next door",
    "barely legal", "youthful", "petite teen", "young student"
]

class IllegalContentScanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING",),
                "age_threshold": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("flagged", "reason")
    FUNCTION = "run"
    CATEGORY = "Content Moderation"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        age_path = "models/age"
        self.age_processor = AutoImageProcessor.from_pretrained(age_path)
        self.age_model = AutoModelForImageClassification.from_pretrained(age_path).to(self.device)
        self.age_labels = self.age_model.config.id2label

    def predict_age(self, image):
        inputs = self.age_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.age_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        label_idx = scores.argmax().item()
        return self.age_labels[label_idx].lower(), scores[label_idx].item()

    def run(self, image, prompt, age_threshold):
        if any(k in prompt.lower() for k in PROMPT_KEYWORDS):
            raise HTTPException(status_code=403, detail="🚫 Suspicious prompt: possible underage content")

        if isinstance(image, torch.Tensor):
            pil_img = Image.fromarray((image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
            age_label, age_score = self.predict_age(pil_img)
            if age_label in ["child", "teen"] and age_score > age_threshold:
                raise HTTPException(status_code=403, detail=f"🚫 Underage detected (age: {age_label}, score: {age_score:.2f})")

        return (False, "✅ Passed all checks")
