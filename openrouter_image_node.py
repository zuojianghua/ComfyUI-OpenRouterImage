"""OpenRouter Image Generation Node for ComfyUI.

This module provides a ComfyUI node for generating images using OpenRouter API.
Supports system/user prompts, multiple reference images, and configurable resolution/aspect ratio.
"""

from typing import Optional, Tuple

import torch
from PIL import Image

from .utils import tensor_to_pils, pils_to_tensor, pil_to_base64_data_url, base64_to_pil


class OpenRouterImageNode:
    """ComfyUI node for generating images via OpenRouter API.

    Supports:
    - System and user prompt inputs
    - Multiple reference images (up to 10)
    - Configurable resolution (0.5K, 1K, 2K, 4K)
    - Configurable aspect ratio (1:1, 2:3, 3:2, 16:9, 9:16, 4:3, 3:4)
    - OpenRouter chat completions API with image generation modalities
    """

    # Model options for dropdown
    MODELS = [
        "google/gemini-3-pro-image-preview",
        "google/gemini-3.1-flash-image-preview",
        "google/gemini-2.5-flash-image",
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
                "api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
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
                        f"[ComfyUI-OpenRouterImage] Warning: Failed to convert image tensor: {e}"
                    )
        return images

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        reference_images: list[Image.Image],
    ) -> list[dict]:
        """Build the messages array for OpenRouter API.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            reference_images: List of PIL Images to include as reference

        Returns:
            List of message dictionaries for the API
        """
        # Build user content with text and optional images
        user_content = []

        # Add reference images first (before text for better context)
        for pil_img in reference_images:
            try:
                data_url = pil_to_base64_data_url(pil_img, format="jpeg")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    }
                )
            except Exception as e:
                print(
                    f"[ComfyUI-OpenRouterImage] Warning: Failed to encode reference image: {e}"
                )

        # Add user prompt text
        user_content.append(
            {
                "type": "text",
                "text": user_prompt,
            }
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return messages

    def _call_openrouter_api(
        self,
        api_key: str,
        model: str,
        messages: list[dict],
        resolution: str,
        aspect_ratio: str,
    ) -> Tuple[Optional[Image.Image], str]:
        """Call OpenRouter API to generate image.

        Args:
            api_key: OpenRouter API key
            model: Model identifier
            messages: Messages array for chat completion
            resolution: Image resolution (0.5K, 1K, 2K, 4K)
            aspect_ratio: Aspect ratio string (e.g., "1:1")

        Returns:
            Tuple of (PIL Image or None, status string)
        """
        if not api_key:
            return None, "Error: API key is required"

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
                f"Error: 0.5K resolution is only supported by google/gemini-3.1-flash-image-preview, "
                f"but current model is {model}. Please use 1K, 2K, or 4K instead.",
            )

        # Build extra_body configuration for OpenAI SDK
        extra_body = {
            "modalities": ["image", "text"],
            "temperature": 0.6,
            "candidateCount": 1,
        }

        # Add image_config
        image_config = {}
        if aspect_ratio != "1:1":
            image_config["aspect_ratio"] = aspect_ratio
        if resolution != "1K":
            image_config["image_size"] = resolution

        if image_config:
            extra_body["image_config"] = image_config

        # Debug: print configuration being sent
        import json
        print(
            f"[ComfyUI-OpenRouterImage] Using OpenAI SDK with extra_body: {json.dumps(extra_body, indent=2)}"
        )

        # Initialize OpenAI client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Call OpenRouter API using OpenAI SDK
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
            )
        except Exception as e:
            return None, f"API Error: {str(e)}"

        # Parse response to extract generated image
        try:
            message = response.choices[0].message

            # Check for images in the response
            if hasattr(message, "images") and message.images:
                # Get the last image (usually the highest resolution)
                last_image_idx = len(message.images) - 1
                image_url = message.images[last_image_idx]["image_url"]["url"]

                if image_url.startswith("data:image"):
                    # Extract base64 data
                    base64_str = image_url.split(",", 1)[1]
                    pil_img = base64_to_pil(base64_str)
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
        api_key: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
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
        """Generate image using OpenRouter API.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User's image generation prompt
            api_key: OpenRouter API key
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

        # Call API
        generated_img, status = self._call_openrouter_api(
            api_key=api_key,
            model=model,
            messages=messages,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
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
    "OpenRouterImageNode": OpenRouterImageNode,
}

# Node display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterImageNode": "OpenRouter Image Generator",
}
