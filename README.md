
# 🛡️ Illegal Content Scanner Node for ComfyUI

This is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that detects potentially illegal content in generated **images**, **videos**, and **prompts**. It is designed to:

- ✅ Allow all adult NSFW content
- 🚫 Flag only **illegal** content involving **minors**
- ❗ Cancel workflow execution and return **403 Forbidden** for flagged inputs

---

## 📦 Features

- 📸 NSFW image detection using [`nielsr/vit-b16-finetuned-nsfw`](https://huggingface.co/nielsr/vit-b16-finetuned-nsfw)
- 🧒 Age estimation using [`nateraw/vit-age-classifier`](https://huggingface.co/nateraw/vit-age-classifier)
- 🧠 Prompt keyword scan for underage references
- 📼 Optional video frame scanning (10 evenly sampled frames per clip)
- 🚫 Cancels ComfyUI workflow with `PermissionError(403)` if content is flagged

---

## 📂 Installation

1. **Extract this zip** into your `ComfyUI/custom_nodes/` directory:
   ```
   ComfyUI/
   └── custom_nodes/
       └── illegal_content_scanner/
           ├── __init__.py
           └── illegal_content_scanner.py
   ```

2. Restart ComfyUI.

---

## 🧠 Usage

Add `🛡️ Illegal Content Scanner` node **after image generation or before output**. You can connect it to halt or redirect workflows depending on its boolean output or raised exception.

### Inputs

| Name             | Type     | Description                                      |
|------------------|----------|--------------------------------------------------|
| `image`          | IMAGE    | A single frame or image to scan                 |
| `prompt`         | STRING   | The prompt used for generation                 |
| `video_path`     | STRING   | Path to a video file (optional)                |
| `nsfw_threshold` | FLOAT    | NSFW confidence threshold (default `0.95`)     |
| `age_threshold`  | FLOAT    | Underage confidence threshold (default `0.85`) |
| `scan_video`     | BOOLEAN  | Whether to scan video frames (default: True)   |

### Output

- `flagged` (BOOLEAN): Always returns `False` if no error, but **raises `403 Forbidden` exception** if content is illegal.

---

## ⚠️ Warning & Best Practices

This node uses machine learning models and keyword filters for heuristic flagging. It is not a substitute for human review.

- Do not log or store flagged images.
- Always inform users in your Terms of Service.
- For public services, consider layering this with AWS Rekognition, Google CSAI, or PhotoDNA.

---

## 🧪 Testing

To simulate detection:
- Use prompts like `"13 year old girl"` or images of child-like anime characters in NSFW scenarios.
- Include a video file with such content to test frame-level scanning.

---

## 📃 License

MIT License
