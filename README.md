# Remove BG – SAM3 Serverless Endpoint

Pure Python replacement for the ComfyUI "Remove BG V3.1" workflow, using the **exact same SAM3 model** from Meta AI.

## Features

- **1:1 replica** of ComfyUI SAM3Segment node behavior
- **Text-prompt segmentation** (default: "one single postcard")
- **Auto-rotate** portrait images to landscape (90° clockwise)
- **Transparent background** (Alpha) or custom color
- **Auto-crop** to bounding box
- All parameters configurable per request

## Deploy to RunPod Serverless

### 1. Push to GitHub → Docker image builds automatically via GitHub Actions

The image is pushed to `ghcr.io/YOUR_USERNAME/remove-bg:latest`.

### 2. Create RunPod Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. **New Endpoint** → Container Image: `ghcr.io/YOUR_USERNAME/remove-bg:latest`
3. GPU: 24GB VRAM recommended (RTX 4090 / A5000)

### 3. Use the API

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/image.jpg",
      "prompt": "one single postcard"
    }
  }'
```

## API Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_url` | *required* | URL of the image |
| `prompt` | `"one single postcard"` | Text prompt for segmentation |
| `threshold` | `0.05` | Confidence threshold |
| `max_segments` | `1` | Max detections |
| `mask_blur` | `0` | Gaussian blur on mask |
| `mask_offset` | `0` | Expand/shrink mask |
| `bbox_padding` | `1` | Crop padding (px) |
| `background` | `"Alpha"` | `"Alpha"` or `"Color"` |
| `bg_color` | `"#222222"` | Background color |
| `auto_rotate` | `true` | Rotate portrait → landscape |
| `invert` | `false` | Invert mask |

## Model

**SAM3** (Segment Anything Model 3) by Meta AI, loaded from public [1038lab/sam3](https://huggingface.co/1038lab/sam3) mirror.
