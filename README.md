# ComfyUI-OpenRouterImage

A ComfyUI custom node for generating images using the OpenRouter API. This node enables you to create high-quality images from text prompts and reference images using powerful models like Google's Gemini 3 Pro Image Preview, Gemini 3.1 Flash Image Preview, and Gemini 2.5 Flash Image.

## Features

- **Text-to-Image Generation**: Create images from detailed text descriptions
- **Image-to-Image Support**: Use up to 10 reference images to guide generation
- **Multiple AI Models**: Choose between Google Gemini 3 Pro Image Preview, Gemini 3.1 Flash Image Preview, and Gemini 2.5 Flash Image
- **Flexible Resolutions**: Generate images from 0.5K to 4K
- **Configurable Aspect Ratios**: Support for 1:1, 2:3, 3:2, 16:9, 9:16, 4:3, 3:4
- **System & User Prompts**: Separate system instructions and user prompts for better control
- **ComfyUI Native**: Seamlessly integrates with ComfyUI's node-based workflow

## Features

- **Text-to-Image Generation**: Create images from detailed text descriptions
- **Image-to-Image Support**: Use up to 4 reference images to guide generation
- **Multiple AI Models**: Choose between Google Gemini 3 Pro and Gemini 3.1 Flash
- **Flexible Resolutions**: Generate images from 0.5K to 4K
- **Configurable Aspect Ratios**: Support for 1:1, 2:3, 3:2, 16:9, 9:16, 4:3, 3:4
- **System & User Prompts**: Separate system instructions and user prompts for better control
- **ComfyUI Native**: Seamlessly integrates with ComfyUI's node-based workflow

## Requirements

- ComfyUI installed and running
- Python 3.8+
- OpenRouter API key (get one free at [openrouter.ai](https://openrouter.ai))

## Installation

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-OpenRouterImage.git
cd ComfyUI-OpenRouterImage
pip install -r requirements.txt
```

### Method 2: ComfyUI Manager

1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "OpenRouterImage"
4. Click Install
5. Restart ComfyUI

### Method 3: Manual Installation

1. Download the latest release
2. Extract to `ComfyUI/custom_nodes/ComfyUI-OpenRouterImage/`
3. Install dependencies:
   ```bash
   pip install openai>=1.0.0 pillow>=9.0.0 requests>=2.28.0
4. Restart ComfyUI

## Quick Start

1. **Get an API Key**: Sign up at [openrouter.ai](https://openrouter.ai) and copy your API key
2. **Add the Node**: In ComfyUI, right-click > Add Node > image_generation > OpenRouter Image Generator
3. **Configure**: Enter your API key, write a prompt, and click Queue Prompt
4. **View Results**: Generated image appears in the node's output

## Usage Guide

### Basic Text-to-Image

1. Add the OpenRouter Image Generator node to your workflow
2. Enter your OpenRouter API key in the `api_key` field
3. Write your image description in the `user_prompt` field
4. Select a model from the dropdown
5. Choose resolution and aspect ratio
6. Run the workflow

### Using Reference Images

The node accepts up to 10 reference images to guide generation:

1. Connect image outputs from Load Image nodes to `image1`, `image2`, ..., or `image10` inputs
2. Describe what you want in the `user_prompt`
3. The model will use reference images as visual context

Example workflow:
```
[Load Image] ---> [OpenRouter Image Generator]
                     ^
                     | (image1 input)
[Prompt Node] --- (user_prompt)
```

### Prompt Engineering Tips

**System Prompt**: Sets the behavior and role of the AI
- Default: "You are an expert image generation assistant. Create high-quality, detailed images..."
- Customize to control style: "You are a professional portrait photographer..."

**User Prompt**: Describes what you want to generate
- Be specific: "A serene mountain landscape at golden hour with snow-capped peaks"
- Include style: "Digital art style, vibrant colors, cinematic lighting"
- Add details: "8K quality, photorealistic, sharp focus"

## Parameter Reference

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `system_prompt` | String | System instructions for the AI | "You are an expert image generation assistant..." |
| `user_prompt` | String | Your image generation prompt | "A beautiful landscape..." |
| `api_key` | String | Your OpenRouter API key | (empty) |
| `model` | Dropdown | AI model to use | google/gemini-3-pro-image-preview |
| `resolution` | Dropdown | Output image resolution | 1K |
| `aspect_ratio` | Dropdown | Output aspect ratio | 1:1 |
| `image1-10` | Image | Optional reference images | (optional) |

### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `google/gemini-3-pro-image-preview` | Google's flagship image model | High-quality, detailed images |
| `google/gemini-3.1-flash-image-preview` | Faster, more efficient model | Quick iterations, prototyping (supports 0.5K resolution) |
| `google/gemini-2.5-flash-image` | Stable, proven model | Balanced performance and quality |

### Resolution Options

| Option | Dimensions | Best For |
|--------|------------|----------|
| 0.5K | 512x512 | Quick previews, testing (gemini-3.1-flash only) |
| 1K | 1024x1024 | Standard quality |
| 2K | 2048x2048 | High quality |
| 4K | 4096x4096 | Maximum quality |

**Note**: 0.5K resolution is only supported by `google/gemini-3.1-flash-image-preview`.

### Aspect Ratio Options

| Ratio | Orientation | Best For |
|-------|-------------|----------|
| 1:1 | Square | Social media posts |
| 2:3 | Portrait | Phone wallpapers, portraits |
| 3:2 | Landscape | Standard photography |
| 16:9 | Widescreen | Desktop wallpapers, video |
| 9:16 | Mobile Portrait | Stories, vertical content |
| 4:3 | Standard | Classic photography |
| 3:4 | Portrait | Portrait photography |

## Example Workflows

### Example 1: Simple Text-to-Image

```
┌─────────────────────────────┐
│ OpenRouter Image Generator  │
├─────────────────────────────┤
│ system_prompt: default      │
│ user_prompt: "A mystical    │
│   forest with glowing        │
│   mushrooms at twilight"     │
│ api_key: sk-or-...           │
│ model: gemini-3-pro-image-preview          │
│ resolution: 1K               │
│ aspect_ratio: 16:9           │
└─────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  Save Image   │
    └───────────────┘
```

### Example 2: Style Transfer with Reference

```
┌──────────────┐
│  Load Image  │──┐ (artwork reference)
└──────────────┘  │
                  ▼
┌─────────────────────────────┐
│ OpenRouter Image Generator  │
├─────────────────────────────┤
│ system_prompt: "Recreate    │
│   in the style of the       │
│   reference image"          │
│ user_prompt: "A modern city  │
│   skyline at night"          │
│ image1: [connected]         │
└─────────────────────────────┘
```

### Example 3: Iterative Refinement

Connect the output of one OpenRouter node to the input of another to iteratively refine your image through multiple passes with different prompts.

## API Reference

This plugin uses the OpenRouter Chat Completions API with image generation capabilities. For detailed API documentation, visit:

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [OpenRouter API Reference](https://openrouter.ai/docs/api)

## Troubleshooting

### Common Issues

**Error: "API key is required"**
- Solution: Enter your OpenRouter API key in the `api_key` field
- Get a free API key at [openrouter.ai](https://openrouter.ai/keys)

**Error: "requests package not installed"**
- Solution: Run `pip install requests>=2.28.0` in your ComfyUI environment

**Error: "No image in response"**
- The model may have returned text instead of an image
- Check the status output for details
- Try refining your prompt to be more specific about image generation

**Generated image is black/empty**
- Check that your API key is valid and has credits
- Verify the model supports image generation
- Check the status output for API error messages

**Node not appearing in ComfyUI**
- Restart ComfyUI after installation
- Check the console for import errors
- Verify Python dependencies are installed

**Error: 0.5K resolution is only supported by google/gemini-3.1-flash-image-preview**
- Solution: Use 1K, 2K, or 4K resolution with other models, or switch to gemini-3.1-flash-image-preview for 0.5K

### Getting Help

- Check the [OpenRouter Status Page](https://status.openrouter.ai/) for API issues
- Visit [ComfyUI Issues](https://github.com/comfyanonymous/ComfyUI/issues) for ComfyUI-specific problems
- Review the [OpenRouter Discord](https://discord.gg/openrouter) for community support

## Security Notes

- Never share your API key or commit it to version control
- The API key field accepts an empty string for security; enter your key each session or use environment variables
- Reference images are encoded and sent to OpenRouter's API
- Review OpenRouter's [Privacy Policy](https://openrouter.ai/privacy) for data handling details

## Limitations

- Image generation requires an active internet connection
- API usage is subject to OpenRouter's rate limits and pricing
- Generated images may vary in quality based on the prompt
- Reference images are limited to 10 per request

## Credits

- Built for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Powered by [OpenRouter](https://openrouter.ai)
- Uses Google's Gemini models via OpenRouter

## License

MIT License - See LICENSE file for details

## Changelog

### v1.1.0
- Updated model names to use -image-preview suffix
- Added support for up to 10 reference images
- Added gemini-2.5-flash-image model
- Added 0.5K resolution support (gemini-3.1-flash only)
- Updated to use direct requests API instead of OpenAI SDK

### v1.0.0
- Initial release
- Support for Gemini image models
- Text-to-image generation
- Image-to-image with reference images
- Configurable resolution and aspect ratio
- ComfyUI native node integration
