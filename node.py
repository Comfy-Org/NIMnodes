import base64
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import torch

# TODO: where does this port come from? Should it be an input or can we use some API to get it?
invoke_url = "http://localhost:8003/v1/infer"


class NIMSDXLNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (["768", "832", "896", "960", "1024", "1088", "1152", "1216", "1280", "1344"], {
                    "default": "1024",
                    "tooltip": "Width of the image to generate, in pixels."
                }),
                "height": (["768", "832", "896", "960", "1024", "1088", "1152", "1216", "1280", "1344"], {
                    "default": "1024",
                    "tooltip": "Height of the image to generate, in pixels."
                }),
                "positive": ("STRING", {
                    "multiline": True,
                    "default": "beautiful scenery nature glass bottle landscape, purple galaxy bottle",
                    "tooltip": "The attributes you want to include in the image."
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "text, watermark",
                    "tooltip": "The attributes you want to exclude from the image."
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)"
                }),
                "sampler": (["DDIM", "K_EULER_ANCESTRAL", "K_DPM_2_ANCESTRAL"], {
                    "default": "K_DPM_2_ANCESTRAL",
                    "tooltip": "The sampler to use for generation. Varying diffusion samplers will vary outputs significantly."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,  # Max 64-bit unsigned int
                    "display": "number",
                    "tooltip": "The seed which governs generation. Use 0 for a random seed"
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Number of diffusion steps to run"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image generation"

    def generate(self, width, height, positive, negative, cfg_scale, sampler, seed, steps):
        payload = {
            "width": width,
            "height": height,
            "text_prompts": [
                {
                    "text": positive,
                    "weight": 1
                },
                {
                    "text": negative,
                    "weight": 1
                }
            ],
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "seed": seed,
            "steps": steps
        }

        response = requests.post(invoke_url, json=payload)
        response.raise_for_status()
        data = response.json()
        img_base64 = data['artifacts'][0]["base64"]
        img_bytes = base64.b64decode(img_base64)

        image = Image.open(BytesIO(img_bytes))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


# Update the mappings
NODE_CLASS_MAPPINGS = {
    "NIMSDXLNode": NIMSDXLNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NIMSDXLNode": "NIM SDXL Node"
}
