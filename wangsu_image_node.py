"""Wangsu Image Generation Node for ComfyUI.

This module provides a ComfyUI node for generating images using Wangsu API.
Supports system/user prompts, multiple reference images, and configurable resolution/aspect ratio.
"""

import base64
import json
import os
import random
from pathlib import Path
from urllib.request import urlopen
from typing import Optional, Tuple

import torch
from dotenv import load_dotenv
from PIL import Image

from .jizhu_reporting import report_image
from .utils import tensor_to_pils, pils_to_tensor, pil_to_base64_data_url, base64_to_pil


DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

ENV_WANGSU_API_KEY = "WANGSU_API_KEY"
ENV_WANGSU_BASE_URL = "WANGSU_BASE_URL"


def _get_env_value(primary_key: str) -> Optional[str]:
    return os.getenv(primary_key)


class WangsuBananaImageNode:
    """ComfyUI node for generating images via Wangsu API.

    Supports:
    - System and user prompt inputs
    - Multiple reference images (up to 10)
    - Configurable resolution (0.5K, 1K, 2K, 4K)
    - Configurable aspect ratio (1:1, 2:3, 3:2, 16:9, 9:16, 4:3, 3:4)
    - Wangsu chat completions API with image generation modalities
    """

    # Model options for dropdown
    MODELS = [
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
    ]

    # Resolution options for dropdown
    RESOLUTIONS = [
        "0.5K",
        "1K",
        "2K",
        "4K",
    ]

    # Aspect ratio options for dropdown
    ASPECT_RATIOS = [
        "1:1",
        "2:3",
        "3:2",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
    ]

    # ComfyUI node attributes
    CATEGORY = "image_generation"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """Define input parameters for the node.

        Returns:
            Dictionary with 'required' and 'optional' input specifications.
        """
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an expert image generation assistant. Create high-quality, detailed images based on the user's description.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A beautiful landscape with mountains and a sunset",
                    },
                ),
                "model": (
                    cls.MODELS,
                    {
                        "default": cls.MODELS[0],
                    },
                ),
                "resolution": (
                    cls.RESOLUTIONS,
                    {
                        "default": "1K",
                    },
                ),
                "aspect_ratio": (
                    cls.ASPECT_RATIOS,
                    {
                        "default": "1:1",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Controls randomness: 0 = deterministic, higher = more creative",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Seed for reproducible generation. -1 = random each run",
                    },
                ),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            },
        }

    def _collect_reference_images(
        self,
        image1: Optional[torch.Tensor],
        image2: Optional[torch.Tensor],
        image3: Optional[torch.Tensor],
        image4: Optional[torch.Tensor],
        image5: Optional[torch.Tensor],
        image6: Optional[torch.Tensor],
        image7: Optional[torch.Tensor],
        image8: Optional[torch.Tensor],
        image9: Optional[torch.Tensor],
        image10: Optional[torch.Tensor],
    ) -> list[Image.Image]:
        """Collect and convert reference images from input tensors.

        Args:
            image1-10: Optional ComfyUI image tensors (B, H, W, C) float 0-1

        Returns:
            List of PIL Images in RGB mode
        """
        images = []
        for img_tensor in [
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        ]:
            if img_tensor is not None:
                try:
                    pils = tensor_to_pils(img_tensor)
                    images.extend(pils)
                except Exception as e:
                    print(
                        f"[ComfyUI-WangsuImage] Warning: Failed to convert image tensor: {e}"
                    )
        return images

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        reference_images: list[Image.Image],
    ) -> list[dict]:
        """Build the messages array for Wangsu API.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            reference_images: List of PIL Images to include as reference

        Returns:
            List of message dictionaries for the API
        """
        user_content = []

        if system_prompt and system_prompt.strip():
            user_content.append(
                {
                    "type": "text",
                    "text": system_prompt,
                }
            )

        for pil_img in reference_images:
            try:
                data_url = pil_to_base64_data_url(pil_img, format="jpeg")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high",
                        },
                    }
                )
            except Exception as e:
                print(
                    f"[ComfyUI-WangsuImage] Warning: Failed to encode reference image: {e}"
                )

        user_content.append(
            {
                "type": "text",
                "text": user_prompt,
            }
        )

        messages = [
            {"role": "user", "content": user_content},
        ]

        return messages

    def _extract_image_url(self, message) -> Optional[str]:
        images = getattr(message, "images", None)
        if images:
            image_urls = [
                image["image_url"]["url"]
                for image in images
                if isinstance(image, dict)
                and isinstance(image.get("image_url"), dict)
                and image["image_url"].get("url")
            ]
            if image_urls:
                return image_urls[-1]

        content = getattr(message, "content", None)
        if isinstance(content, list):
            typed_image_urls = [
                item["image_url"]["url"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "image_url"
                and isinstance(item.get("image_url"), dict)
                and item["image_url"].get("url")
            ]
            if typed_image_urls:
                return typed_image_urls[-1]

            fallback_image_urls = [
                item["image_url"]["url"]
                for item in content
                if isinstance(item, dict)
                and isinstance(item.get("image_url"), dict)
                and item["image_url"].get("url")
            ]
            if fallback_image_urls:
                return fallback_image_urls[-1]

        return None

    def _image_url_to_pil(self, image_url: str) -> Image.Image:
        if image_url.startswith("data:"):
            return base64_to_pil(image_url)

        if image_url.startswith("http"):
            with urlopen(image_url, timeout=60) as response:
                return Image.open(response).convert("RGB")

        return base64_to_pil(image_url)

    def _call_wangsu_api(
        self,
        model: str,
        messages: list[dict],
        resolution: str,
        aspect_ratio: str,
        temperature: float = 0.6,
        seed: int = -1,
    ) -> Tuple[Optional[Image.Image], str]:
        """Call Wangsu API to generate image.

        Args:
            model: Model identifier
            messages: Messages array for chat completion
            resolution: Image resolution (0.5K, 1K, 2K, 4K)
            aspect_ratio: Aspect ratio string (e.g., "1:1")

        Returns:
            Tuple of (PIL Image or None, status string)
        """
        api_key = _get_env_value(ENV_WANGSU_API_KEY)
        if not api_key:
            return (
                None,
                "Error: Wangsu API key is required. Set WANGSU_API_KEY in .env.",
            )

        base_url = _get_env_value(ENV_WANGSU_BASE_URL)
        if not base_url:
            return (
                None,
                "Error: Wangsu base URL is required. Set WANGSU_BASE_URL in .env.",
            )

        # Use OpenAI SDK
        try:
            from openai import OpenAI
        except ImportError:
            return (
                None,
                "Error: openai package not installed. Run: pip install openai",
            )

        # Validate resolution compatibility
        # 0.5K is only supported by gemini-3.1-flash-image-preview
        if resolution == "0.5K" and "gemini-3.1-flash" not in model:
            return (
                None,
                f"Error: 0.5K resolution is only supported by gemini-3.1-flash-image-preview, "
                f"but current model is {model}. Please use 1K, 2K, or 4K instead.",
            )

        extra_body = {
            "modalities": ["image", "text"],
            "temperature": temperature,
            "candidateCount": 1,
            "image_config": {
                "image_size": resolution,
                "aspect_ratio": aspect_ratio,
            },
            "eca_image_config": {
                "image_size": resolution,
                "aspect_ratio": aspect_ratio,
            },
        }

        # Use random seed if seed is -1, otherwise use specified seed
        if seed != -1:
            extra_body["seed"] = seed

        print(
            f"[ComfyUI-WangsuImage] Using OpenAI SDK with extra_body: {json.dumps(extra_body, indent=2)}"
        )

        # Initialize OpenAI client
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Call Wangsu API using OpenAI SDK
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
            )
        except Exception as e:
            return None, f"API Error: {str(e)}"

        try:
            message = response.choices[0].message

            image_url = self._extract_image_url(message)
            if image_url:
                pil_img = self._image_url_to_pil(image_url)
                return pil_img, "Image generated successfully"

            # No image found - return text content as status
            text_content = ""
            if hasattr(message, "content"):
                if isinstance(message.content, str):
                    text_content = message.content
                elif isinstance(message.content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in message.content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    text_content = " ".join(text_parts)

            if text_content:
                return (
                    None,
                    f"No image in response. Model returned text: {text_content[:200]}...",
                )

            return None, "Error: No image or text content in API response"

        except Exception as e:
            return None, f"Error parsing response: {str(e)}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        temperature: float = 0.6,
        seed: int = -1,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
        image6: Optional[torch.Tensor] = None,
        image7: Optional[torch.Tensor] = None,
        image8: Optional[torch.Tensor] = None,
        image9: Optional[torch.Tensor] = None,
        image10: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        """Generate image using Wangsu API.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User's image generation prompt
            model: Selected model identifier
            resolution: Image resolution (0.5K, 1K, 2K, 4K)
            aspect_ratio: Aspect ratio (1:1, 2:3, etc.)
            image1-10: Optional reference image tensors

        Returns:
            Tuple of (generated image tensor, status string)
        """
        # Create empty placeholder image for error cases
        placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Validate inputs
        if not user_prompt or not user_prompt.strip():
            return placeholder, "Error: User prompt is required"

        # Collect reference images
        reference_images = self._collect_reference_images(
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        )

        # Build messages for API
        messages = self._build_messages(system_prompt, user_prompt, reference_images)

        # Resolve seed: if -1, generate a random seed
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        print(f"[ComfyUI-WangsuImage] Using seed: {seed}, temperature: {temperature}")

        generated_img, status = report_image(
            model,
            "wangsu",
            lambda: self._call_wangsu_api(
                model=model,
                messages=messages,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                temperature=temperature,
                seed=seed,
            ),
        )

        if generated_img is None:
            return placeholder, status

        # Convert PIL image to ComfyUI tensor
        try:
            image_tensor = pils_to_tensor([generated_img])
            return image_tensor, status
        except Exception as e:
            return placeholder, f"Error converting image to tensor: {str(e)}"


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WangsuBananaImageNode": WangsuBananaImageNode,
}

# Node display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "WangsuBananaImageNode": "Wangsu Banana Image Generator",
}
