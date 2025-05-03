
# 🛡️ Illegal Content Scanner Node for ComfyUI

This is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that detects potentially illegal content in generated images based on:

- ✅ NSFW detection
- ✅ Age estimation
- ✅ Prompt keyword scanning

It allows all adult content but flags content that **may involve minors** or **suspicious prompts**.

---

## 📦 Features

- 🚨 Flags NSFW images likely involving underage subjects
- 🔍 Scans user prompt for risky keywords (e.g. "13 year old", "child")
- 🎛️ Adjustable thresholds for NSFW and age detection

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

2. **Update model path** in `illegal_content_scanner.py`:
   ```python
   self.nsfw_detector = pipeline("image-classification", model="path/to/nsfw/model")
   ```
   Example model: [`AdamCodd/vit-base-nsfw-detector`](https://huggingface.co/AdamCodd/vit-base-nsfw-detector)

3. Restart ComfyUI.

---

## 🧠 Usage

- Add `🛡️ Illegal Content Scanner` node after image generation.
- It outputs a boolean (`flagged`) that you can use to halt workflows or trigger alternate logic.

### Inputs

| Name            | Type    | Description |
|-----------------|---------|-------------|
| `image`         | IMAGE   | Image to scan |
| `prompt`        | STRING  | Generation prompt |
| `nsfw_threshold`| FLOAT   | Score above which an image is considered NSFW |
| `age_threshold` | FLOAT   | Score above which subject is considered underage |

### Output

- `flagged` (BOOLEAN): Returns `True` if illegal content is detected.

---

## ⚠️ Disclaimer

This node **does not guarantee perfect detection**. It is a heuristic filter and should be used as a **safety layer**.

If you are operating a public-facing generation service, you should:

- Use multiple moderation tools
- Log/report abuse to proper authorities
- Never store flagged images

---

## 📃 License

MIT
