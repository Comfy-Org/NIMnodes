import base64
from io import BytesIO
import os
import sys
import tempfile
import time
import numpy as np
import requests
import torch
from PIL import Image
from typing import Dict, Tuple

from .install import download_installer, run_installer
from .nim import ModelType, NIMManager, OffloadingPolicy

manager = NIMManager()

class NIMFLUXNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "is_nim_started": ("STRING", {"forceInput": True}),
                "width": ("INT", {
                    "default": 1024,
                    "min": 672,
                    "max": 1568,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Width of the image to generate, in pixels."
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 672,
                    "max": 1568,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Height of the image to generate, in pixels."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "beautiful scenery nature glass bottle landscape, purple galaxy bottle",
                    "tooltip": "The attributes you want to include in the image."
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 9.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4294967295,
                    "display": "number",
                    "tooltip": "The seed which governs generation. Use 0 for a random seed"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Number of diffusion steps to run"
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The image used for depth, canny & kontext mode."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NVIDIA/NIM"

    def generate(self, width, height, prompt, cfg_scale, seed, steps, is_nim_started, image=None):
        # Validate that width and height are divisible by 32
        error_messages = []
        """ if width % 32 != 0:
            error_messages.append(f"Width ({width}) must be divisible by 32.")
        if height % 32 != 0:
            error_messages.append(f"Height ({height}) must be divisible by 32.")
        if error_messages:
            raise ValueError(" ".join(error_messages)) """

        if is_nim_started[0] == "":
            raise Exception("Please make sure use 'Load NIM' before this node to start NIM.")
        model_name = ModelType[is_nim_started[0]]
        port = manager.get_port(model_name)
        invoke_url = f"http://localhost:{port}/v1/infer"

        mode = NIMManager._get_variant(self, model_name)

        if model_name.value.split('_')[-1].lower() == 'schnell':
            cfg_scale = 0
            if steps > 4:
                raise Exception("Flux Schnell step value must be between 1-4 steps")
    
        payload = {
            "width": int(width),
            "height": int(height),
            "text_prompts": [
                {
                    "text": prompt,
                },
            ],
            "mode": mode,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "steps": steps
        }
        
        print(f'Payload is: payload {payload}')
        
        if model_name.value.split('_')[-1].lower() not in ['schnell', 'dev', 'base']:
            if image is None:
                raise Exception("Please use load image node to select image input for FLUX depth, canny and kontext modes.")
        
            def _comfy_image_to_bytes(img: torch.tensor, depth: int = 8):
                max_val = 2**depth - 1
                img = torch.clip(img * max_val, 0, max_val).to(dtype=torch.uint8)
                pil_img = Image.fromarray(img.squeeze(0).cpu().numpy())

                img_byte_arr = BytesIO()
                pil_img.save(img_byte_arr, format="PNG")
                return img_byte_arr.getvalue(), ".png"
            
            image_bytes, _ = _comfy_image_to_bytes(img=image)
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            image = f"data:image/png;base64,{base64_string}"
            payload.update({"image": image})

        try:
            response = requests.post(invoke_url, json=payload)
            print(response)
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Unable to connect to NIM API.")
            
        data = response.json()
        response.raise_for_status()
        img_base64 = data["artifacts"][0]["base64"]
        img_bytes = base64.b64decode(img_base64)

        print("Result: " + data["artifacts"][0]["finishReason"])

        image = Image.open(BytesIO(img_bytes))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]  

        return (image,)


class LoadNIMNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": ([e.value for e in ModelType], {
                    "default": ModelType.FLUX_DEV.value,
                    "tooltip": "The type of NIM model to use"
                }),
                "operation": (["Start", "Stop"],),
                "offloading_policy": ([e.value for e in OffloadingPolicy], {
                    "default": OffloadingPolicy.DEFAULT.value,
                    "tooltip": "Policy to offload models"
                }),
                "hf_token": ("STRING", {
                    "multiline": False,
                    "default": "Input HF Token",
                    "tooltip": "Input your Huggingface API Token"
                }),
                "is_nim_installed": ("BOOLEAN", {"forceInput": True}),
            }
        }
    
    # RETURN_TYPES = ()
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("is_nim_started",)
    OUTPUT_NODE = True
    FUNCTION = "prcoess_nim"
    CATEGORY = "NVIDIA/NIM"

    def prcoess_nim(self, model_type: str, operation: str, offloading_policy: str, hf_token: str, is_nim_installed: bool):
        if is_nim_installed:
            if operation == "Start":
                return (self.start_nim(model_type, offloading_policy, hf_token),)
            elif operation == "Stop":
                return (self.stop_nim(model_type),)
        else:
            raise Exception("Please make sure install NIMs before running this node")
    
    def start_nim(self, model_type: str, offloading_policy: str, hf_token: str):
        manager.deploy_nim(model_name=ModelType[model_type], offloading_policy=offloading_policy, hf_token=hf_token)
        return (model_type,)
    
    def stop_nim(self, model_type: str):
        manager.stop_nim(model_name=ModelType[model_type])
        return ("",)


class InstallNIMNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }
    
    # RETURN_TYPES = ()
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_nim_installed",)
    OUTPUT_NODE = True
    FUNCTION = "install_nim"
    CATEGORY = "NVIDIA/NIM"
    
    def install_nim(self):
        if os.name == 'nt':
            if manager.is_wsl_distribution_installed(distro_name="NVIDIA-Workbench"):
                print("NIM node setup is ready.")
                return (True, )
            else:
                import ctypes

                res = ctypes.windll.user32.MessageBoxW(None, "Detected you haven't set up NVIDIA NIM.\n\n" +
                                "Please ensure you download NIMSetup.exe and install it before attempting to use the node. Click OK to open download website.",
                                "NIM Installer", 1 | 48)
                if res == 1:
                    import webbrowser
                    webbrowser.open("https://assets.ngc.nvidia.com/products/api-catalog/rtx/NIMSetup.exe")

                raise Exception("Please install NVIDIA NIM first and try again.")
                
        else:
            raise Exception("NIM node setup is only supported for Windows")

        return (False,)

class Get_HFToken:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }
    
    # RETURN_TYPES = ()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hf_token",)
    FUNCTION = "get_token"
    CATEGORY = "NVIDIA/NIM"

    def get_token(self) -> Tuple[str]:
        """
        Retrieves the HF_TOKEN environment variable.

        Returns:
            Tuple[str]: A tuple containing the token string or an error message.
        """
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError("HF_TOKEN environment variable not set. Workflow execution halted.") #Raise an error.
        else:
            return (token,)

# Update the mappings
NODE_CLASS_MAPPINGS = {
    "LoadNIMNode": LoadNIMNode,
    "InstallNIMNode": InstallNIMNode,
    "NIMFLUXNode": NIMFLUXNode,
    "Get_HFToken": Get_HFToken
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadNIMNode": "Load NIM",
    "InstallNIMNode": "Install NIM",
    "NIMFLUXNode": "NIM Generate",
    "Get_HFToken": "Use HF_TOKEN EnVar"
}