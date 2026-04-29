"""Wangsu Image Generation Node for ComfyUI (OpenAI images.* API compatible).

Dispatches to the three OpenAI image endpoints via the OpenAI Python SDK:

- generate  -> client.images.generate()           POST /v1/images/generations
- edit      -> client.images.edit()               POST /v1/images/edits
- variation -> client.images.create_variation()   POST /v1/images/variations

Each mode reads its own (base_url, api_key) pair from environment variables so
three independent Wangsu endpoints can be configured:

- WANGSU_IMAGE_GENERATE_URL  / WANGSU_IMAGE_GENERATE_API_KEY
- WANGSU_IMAGE_EDIT_URL      / WANGSU_IMAGE_EDIT_API_KEY
- WANGSU_IMAGE_VARIATION_URL / WANGSU_IMAGE_VARIATION_API_KEY

Inputs follow OpenAI's official field names (model, prompt, n, size, quality,
background). Reference images use image1..image10 slots.

Note: response_format is intentionally NOT exposed. gpt-image-1 family models
(both on OpenAI and on OpenAI-compatible gateways like Wangsu) reject this
parameter. The response decoder handles both b64_json and url transparently.
"""

import io
import os
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import Request, urlopen

import torch
from dotenv import load_dotenv
from PIL import Image

from .utils import base64_to_pil, pils_to_tensor, tensor_to_pils


DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)


MODE_ENV_VARS = {
    "generate": ("WANGSU_IMAGE_GENERATE_URL", "WANGSU_IMAGE_GENERATE_API_KEY"),
    "edit": ("WANGSU_IMAGE_EDIT_URL", "WANGSU_IMAGE_EDIT_API_KEY"),
    "variation": ("WANGSU_IMAGE_VARIATION_URL", "WANGSU_IMAGE_VARIATION_API_KEY"),
}


def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    mode = pil_img.mode
    if mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGBA" if mode in ("LA", "P") else "RGB")
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def _fetch_image_from_url(url: str) -> Image.Image:
    # Wangsu returns image URLs on CDNs (oss.filenest.top and others) with
    # inverted anti-scraping: Python-urllib's default UA gets 403, and a Chrome
    # UA gets the TCP connection RST/dropped. A curl-style UA is the only one
    # that consistently passes both kinds of gateways we've seen.
    req = Request(
        url,
        headers={
            "User-Agent": "curl/8.5.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=60) as resp:
        pil = Image.open(resp)
        pil.load()
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return pil


def _strip_size_label(size: str) -> str:
    # Dropdown entries look like "(1:1) 1024x1024"; the API only accepts the raw "1024x1024" part.
    if size and size.startswith("("):
        _, _, rest = size.partition(") ")
        return rest or size
    return size


def _extract_pil_images_from_response(response) -> List[Image.Image]:
    data = getattr(response, "data", None) or []
    images: List[Image.Image] = []
    for item in data:
        b64 = getattr(item, "b64_json", None)
        url = getattr(item, "url", None)
        if b64:
            images.append(base64_to_pil(b64))
        elif url:
            try:
                images.append(_fetch_image_from_url(url))
            except Exception as e:
                raise RuntimeError(
                    f"failed to fetch image from {url}: {e}"
                ) from e
    return images


class WangsuImageNode:
    """ComfyUI node that calls OpenAI-compatible image endpoints on Wangsu.

    Three modes via a single node: generate (text-to-image), edit (image-to-image
    with up to 10 reference images), variation (single-image variation). Each
    mode uses its own base_url/api_key pair from .env.
    """

    MODES = ["generate", "edit", "variation"]

    SIZES = [
        "auto",
        "(1:1) 1024x1024",
        "(3:2) 1536x1024",
        "(2:3) 1024x1536",
        "(4:3) 1216x912",
        "(3:4) 912x1216",
        "(16:9) 1824x1024",
        "(9:16) 1024x1824",
        "(21:9) 1008x432",
        "(1:1) 2048x2048",
        "(3:2) 2048x1360",
        "(2:3) 1360x2048",
        "(4:3) 2304x1728",
        "(3:4) 1728x2304",
        "(16:9) 2048x1152",
        "(9:16) 1152x2048",
        "(21:9) 2352x1008",
        "(1:1) 2880x2880",
        "(3:2) 3504x2336",
        "(2:3) 2336x3504",
        "(4:3) 2816x2112",
        "(3:4) 2112x2816",
        "(16:9) 3840x2160",
        "(9:16) 2160x3840",
        "(21:9) 3696x1584",
    ]

    QUALITIES = ["auto", "low", "medium", "high"]
    BACKGROUNDS = ["auto", "transparent", "opaque"]

    MODELS = ["gpt-image-2"]

    CATEGORY = "image_generation"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "mode": (cls.MODES, {"default": "generate"}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A beautiful landscape with mountains and a sunset",
                    },
                ),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "size": (cls.SIZES, {"default": "(1:1) 1024x1024"}),
                "quality": (cls.QUALITIES, {"default": "auto"}),
                "background": (cls.BACKGROUNDS, {"default": "auto"}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFF},
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

    def _collect_reference_pils(
        self,
        image_tensors: List[Optional[torch.Tensor]],
    ) -> List[Image.Image]:
        pils: List[Image.Image] = []
        for tensor in image_tensors:
            if tensor is None:
                continue
            try:
                pils.extend(tensor_to_pils(tensor))
            except Exception as e:
                print(
                    f"[ComfyUI-WangsuImage] Warning: failed to convert image tensor: {e}"
                )
        return pils

    def _get_credentials(
        self, mode: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        url_var, key_var = MODE_ENV_VARS[mode]
        base_url = os.getenv(url_var)
        api_key = os.getenv(key_var)
        if not base_url:
            return None, None, f"Error: {url_var} is not set in .env"
        if not api_key:
            return None, None, f"Error: {key_var} is not set in .env"
        return base_url, api_key, None

    def _build_common_kwargs(
        self,
        model: str,
        n: int,
        size: str,
        quality: str,
        background: str,
        include_background: bool,
        include_quality: bool,
    ) -> dict:
        kwargs: dict = {"model": model, "n": n}
        if size and size != "auto":
            kwargs["size"] = _strip_size_label(size)
        if include_quality and quality and quality != "auto":
            kwargs["quality"] = quality
        if include_background and background and background != "auto":
            kwargs["background"] = background
        return kwargs

    def _call_generate(self, client, prompt: str, common_kwargs: dict):
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required for mode='generate'")
        return client.images.generate(prompt=prompt, **common_kwargs)

    def _call_edit(
        self,
        client,
        prompt: str,
        reference_pils: List[Image.Image],
        common_kwargs: dict,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required for mode='edit'")
        if not reference_pils:
            raise ValueError(
                "mode='edit' requires at least one reference image "
                "(connect image1..image10)"
            )
        # SDK accepts (filename, bytes, mime) tuples; a list of them enables
        # multi-image edits without touching the filesystem.
        files = [
            (f"image_{i}.png", _pil_to_png_bytes(pil), "image/png")
            for i, pil in enumerate(reference_pils)
        ]
        image_arg = files if len(files) > 1 else files[0]
        return client.images.edit(image=image_arg, prompt=prompt, **common_kwargs)

    def _call_variation(
        self,
        client,
        reference_pils: List[Image.Image],
        model: str,
        n: int,
        size: str,
    ):
        if not reference_pils:
            raise ValueError(
                "mode='variation' requires a reference image (connect image1)"
            )
        if len(reference_pils) > 1:
            print(
                "[ComfyUI-WangsuImage] Warning: variation only uses image1; "
                f"ignoring {len(reference_pils) - 1} extra reference image(s)"
            )
        pil = reference_pils[0]
        image_arg = ("image.png", _pil_to_png_bytes(pil), "image/png")
        # /images/variations does not accept prompt, quality, or background.
        kwargs: dict = {"model": model, "n": n, "image": image_arg}
        if size and size != "auto":
            kwargs["size"] = _strip_size_label(size)
        return client.images.create_variation(**kwargs)

    def generate(
        self,
        mode: str,
        model: str,
        prompt: str,
        n: int,
        size: str,
        quality: str,
        background: str,
        seed: int,
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
        placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        if mode not in MODE_ENV_VARS:
            return placeholder, f"Error: unknown mode '{mode}'"
        if not model or not model.strip():
            return placeholder, "Error: model is required"

        base_url, api_key, err = self._get_credentials(mode)
        if err:
            return placeholder, err

        try:
            from openai import OpenAI
        except ImportError:
            return (
                placeholder,
                "Error: openai package not installed. Run: pip install openai",
            )

        client = OpenAI(base_url=base_url, api_key=api_key)

        reference_pils = self._collect_reference_pils(
            [
                image1, image2, image3, image4, image5,
                image6, image7, image8, image9, image10,
            ]
        )

        common_kwargs = self._build_common_kwargs(
            model=model,
            n=n,
            size=size,
            quality=quality,
            background=background,
            include_background=(mode in ("generate", "edit")),
            include_quality=(mode in ("generate", "edit")),
        )

        # seed is accepted by the node purely to force ComfyUI cache invalidation
        # across runs with identical prompts; OpenAI images endpoints do not
        # accept a seed parameter, so we log it but don't forward it.
        print(
            f"[ComfyUI-WangsuImage] mode={mode} model={model} "
            f"n={n} size={size} quality={quality} background={background} "
            f"seed={seed} ref_images={len(reference_pils)} base_url={base_url}"
        )

        try:
            if mode == "generate":
                response = self._call_generate(client, prompt, common_kwargs)
            elif mode == "edit":
                response = self._call_edit(
                    client, prompt, reference_pils, common_kwargs
                )
            else:
                response = self._call_variation(
                    client,
                    reference_pils,
                    model=model,
                    n=n,
                    size=size,
                )
        except ValueError as e:
            return placeholder, f"Error: {e}"
        except Exception as e:
            return placeholder, f"API Error ({mode}): {e}"

        try:
            pil_images = _extract_pil_images_from_response(response)
        except Exception as e:
            return placeholder, f"Error decoding response: {e}"

        if not pil_images:
            return placeholder, f"Error: no image data returned by {mode} endpoint"

        try:
            image_tensor = pils_to_tensor(pil_images)
        except Exception as e:
            return placeholder, f"Error converting image to tensor: {e}"

        status = f"OK: mode={mode} returned {len(pil_images)} image(s)"
        return image_tensor, status


NODE_CLASS_MAPPINGS = {
    "WangsuImageNode": WangsuImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WangsuImageNode": "Wangsu Image Generator",
}
