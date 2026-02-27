"""
Pure Python Background Removal Handler for RunPod Serverless.
Replaces the ComfyUI "Remove BG V3.1" workflow — uses the EXACT SAME SAM3 model
with the native SAM3 Python library (from comfyui-rmbg).

Pipeline (1:1 match to ComfyUI SAM3Segment node):
  1.  Load image from URL
  1b. Auto-rotate portrait → landscape (90° clockwise) + EXIF fix
  2.  SAM3: text-prompt → instance segmentation masks
  3.  Merge masks (if multiple) → single mask
  4.  Process mask (blur, offset, invert)
  5.  Apply background (Alpha / Color)
  6.  Compute bounding box from mask + crop
  7.  Return result as base64 PNG

ComfyUI SAM3Segment Defaults (from workflow):
  - prompt:          "one single postcard"
  - output_mode:     "Merged"
  - threshold:       0.05
  - max_segments:    1
  - segment_pick:    0
  - mask_blur:       0
  - mask_offset:     0
  - device:          "Auto" → CUDA
  - invert:          false
  - background:      "Alpha"
  - bg_color:        "#222222"
"""

import base64
import io
import logging
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter, ExifTags

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remove_bg")

# ---------------------------------------------------------------------------
# SAM3 library path — must be set BEFORE import
# ---------------------------------------------------------------------------
SAM3_LIB_DIR = os.environ.get("SAM3_LIB_DIR", "/app/sam3_lib")
if SAM3_LIB_DIR not in sys.path:
    sys.path.insert(0, SAM3_LIB_DIR)

# Now we can import SAM3
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

import runpod  # noqa: E402
import requests  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
SAM3_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT",
    "/app/sam3_models/sam3.pt",
)
SAM3_BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    os.path.join(SAM3_LIB_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AUTOCAST = DEVICE == "cuda"

# Global model cache
sam3_processor = None


def load_models():
    """Load SAM3 model once at worker startup (matches ComfyUI _load_processor)."""
    global sam3_processor

    logger.info("Loading SAM3 model from: %s", SAM3_CHECKPOINT)
    logger.info("BPE tokenizer from: %s", SAM3_BPE_PATH)
    logger.info("Device: %s (autocast=%s)", DEVICE, USE_AUTOCAST)
    t0 = time.time()

    model = build_sam3_image_model(
        bpe_path=SAM3_BPE_PATH,
        device=DEVICE,
        eval_mode=True,
        checkpoint_path=SAM3_CHECKPOINT,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    sam3_processor = Sam3Processor(model, device=DEVICE)

    logger.info("SAM3 loaded in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# Helper Functions (1:1 port from comfyui-rmbg AILab_SAM3Segment.py)
# ---------------------------------------------------------------------------

def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """Download an image from a URL and return as PIL Image (RGB)."""
    logger.info("Downloading image from: %s", url)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    logger.info("Image loaded: %dx%d", image.width, image.height)
    return image


def fix_exif_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag so pixel data matches visual orientation."""
    try:
        exif = image.getexif()
        orientation_key = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orientation_key = tag
                break
        if orientation_key and orientation_key in exif:
            orientation = exif[orientation_key]
            transforms = {
                2: Image.FLIP_LEFT_RIGHT,
                3: Image.ROTATE_180,
                4: Image.FLIP_TOP_BOTTOM,
                5: Image.TRANSPOSE,
                6: Image.ROTATE_270,
                7: Image.TRANSVERSE,
                8: Image.ROTATE_90,
            }
            if orientation in transforms:
                image = image.transpose(transforms[orientation])
                logger.info("Applied EXIF orientation=%d", orientation)
    except Exception:
        pass
    return image


def ensure_landscape(image: Image.Image) -> Image.Image:
    """If portrait (height > width), rotate 90° clockwise."""
    if image.height > image.width:
        image = image.rotate(-90, expand=True)
        logger.info("Rotated portrait → landscape: %dx%d", image.width, image.height)
    return image


def process_mask(mask_image: Image.Image, invert=False, blur=0, offset=0) -> Image.Image:
    """
    Post-process mask — exact port from comfyui-rmbg process_mask().
    """
    if invert:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)
    if blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur))
    if offset != 0:
        filt = ImageFilter.MaxFilter if offset > 0 else ImageFilter.MinFilter
        size = abs(offset) * 2 + 1
        for _ in range(abs(offset)):
            mask_image = mask_image.filter(filt(size))
    return mask_image


def apply_background_color(image: Image.Image, mask_image: Image.Image,
                           background="Alpha", background_color="#222222") -> Image.Image:
    """
    Apply background — exact port from comfyui-rmbg apply_background_color().
    """
    rgba_image = image.copy().convert("RGBA")
    rgba_image.putalpha(mask_image.convert("L"))
    if background == "Color":
        hex_color = background_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        bg_image = Image.new("RGBA", image.size, (r, g, b, 255))
        composite = Image.alpha_composite(bg_image, rgba_image)
        return composite.convert("RGB")
    return rgba_image


def compute_bounding_box(mask_image: Image.Image, padding: int = 1) -> tuple:
    """Compute tight bounding box from mask (matches MaskBoundingBox+ node)."""
    mask_np = np.array(mask_image)
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)

    if not rows.any():
        h, w = mask_np.shape[:2]
        return (0, 0, w, h)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    h, w = mask_np.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding + 1)
    y_max = min(h, y_max + padding + 1)

    logger.info("BBox: x=%d y=%d w=%d h=%d", x_min, y_min, x_max - x_min, y_max - y_min)
    return (x_min, y_min, x_max, y_max)


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as base64."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_sam3_segmentation(
    image: Image.Image,
    prompt: str,
    confidence: float,
    max_segments: int,
    segment_pick: int,
    mask_blur: int,
    mask_offset: int,
    invert: bool,
    background: str,
    background_color: str,
    output_mode: str,
) -> tuple:
    """
    Run SAM3 segmentation — exact port from comfyui-rmbg _run_single_merged / _run_single_per_instance.
    
    Returns:
        (result_image, mask_image) — PIL Images
    """
    text = prompt.strip() or "object"

    # Use autocast for CUDA (bfloat16) — matches ComfyUI behavior
    if USE_AUTOCAST:
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        # Set image
        state = sam3_processor.set_image(image)
        sam3_processor.reset_all_prompts(state)
        sam3_processor.set_confidence_threshold(confidence, state)

        # Run text-prompted segmentation
        state = sam3_processor.set_text_prompt(text, state)

    masks = state.get("masks")
    logits = state.get("masks_logits")

    if masks is None or masks.numel() == 0:
        logger.warning("SAM3 found no instances for prompt='%s'", prompt)
        return None, None, 0

    masks = masks.float()
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Compute scores from logits
    scores = None
    if logits is not None:
        logits = logits.float()
        if logits.ndim == 4:
            logits = logits.squeeze(1)
        scores = logits.mean(dim=(-2, -1))
    if scores is None:
        scores = torch.ones((masks.shape[0],), device=masks.device)

    num_found = masks.shape[0]
    logger.info("SAM3 found %d instance(s)", num_found)

    # Limit to max_segments
    if max_segments > 0 and masks.shape[0] > max_segments:
        topk = torch.topk(scores, k=max_segments)
        masks = masks[topk.indices]
        scores = scores[topk.indices]

    # Sort by score
    sorted_idx = torch.argsort(scores, descending=True)
    masks = masks[sorted_idx]

    # Pick specific segment
    if segment_pick > 0:
        idx = segment_pick - 1
        if idx >= masks.shape[0]:
            logger.warning("segment_pick=%d > available masks=%d", segment_pick, masks.shape[0])
            return None, None, num_found
        masks = masks[idx:idx + 1]

    if output_mode == "Merged":
        # Merge all masks (union) — matches comfyui-rmbg _run_single_merged
        merged = masks.amax(dim=0)
        mask_np = (merged.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np, mode="L")
        mask_image = process_mask(mask_image, invert, mask_blur, mask_offset)
        result_image = apply_background_color(image, mask_image, background, background_color)
        if background == "Alpha":
            result_image = result_image.convert("RGBA")
        else:
            result_image = result_image.convert("RGB")
        return result_image, mask_image, num_found
    else:
        # "Separate" mode — return first/best mask
        single_mask = masks[0]
        mask_np = (single_mask.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np, mode="L")
        mask_image = process_mask(mask_image, invert, mask_blur, mask_offset)
        result_image = apply_background_color(image, mask_image, background, background_color)
        if background == "Alpha":
            result_image = result_image.convert("RGBA")
        else:
            result_image = result_image.convert("RGB")
        return result_image, mask_image, num_found


# ---------------------------------------------------------------------------
# RunPod Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """
    RunPod Serverless handler — exact SAM3 replica of ComfyUI Remove BG V3.1.

    Input:
    {
        "image_url": "https://...",
        "prompt": "one single postcard",       // default from ComfyUI
        "threshold": 0.05,                     // confidence_threshold
        "max_segments": 1,                     // limit detections
        "segment_pick": 0,                     // 0=all, n=pick nth
        "mask_blur": 0,                        // Gaussian blur radius
        "mask_offset": 0,                      // expand/shrink mask
        "bbox_padding": 1,                     // crop padding
        "output_mode": "Merged",               // "Merged" or "Separate"
        "background": "Alpha",                 // "Alpha" or "Color"
        "bg_color": "#222222",                 // background color
        "invert": false,                       // invert mask
        "auto_rotate": true                    // portrait → landscape
    }
    """
    try:
        inp = job.get("input", {})

        image_url = inp.get("image_url")
        if not image_url:
            return {"status": "error", "error": "Missing required: image_url"}

        # Parse with exact ComfyUI defaults
        prompt = inp.get("prompt", "one single postcard")
        threshold = float(inp.get("threshold", 0.05))
        max_segments = int(inp.get("max_segments", 1))
        segment_pick = int(inp.get("segment_pick", 0))
        mask_blur = int(inp.get("mask_blur", 0))
        mask_offset = int(inp.get("mask_offset", 0))
        bbox_padding = int(inp.get("bbox_padding", 1))
        output_mode = inp.get("output_mode", "Merged")
        background = inp.get("background", "Alpha")
        bg_color = inp.get("bg_color", "#222222")
        invert = bool(inp.get("invert", False))
        auto_rotate = bool(inp.get("auto_rotate", True))

        logger.info(
            "Job: prompt='%s' thresh=%.2f max_seg=%d blur=%d offset=%d mode=%s bg=%s",
            prompt, threshold, max_segments, mask_blur, mask_offset, output_mode, background,
        )
        t_start = time.time()

        # --- Step 1: Load image ---
        image = load_image_from_url(image_url)

        # --- Step 1b: EXIF + auto-rotate ---
        image = fix_exif_orientation(image)
        rotated = False
        if auto_rotate and image.height > image.width:
            image = ensure_landscape(image)
            rotated = True

        # --- Step 2-5: SAM3 segmentation + mask processing + background ---
        result_image, mask_image, num_found = run_sam3_segmentation(
            image=image,
            prompt=prompt,
            confidence=threshold,
            max_segments=max_segments,
            segment_pick=segment_pick,
            mask_blur=mask_blur,
            mask_offset=mask_offset,
            invert=invert,
            background=background,
            background_color=bg_color,
            output_mode=output_mode,
        )

        if result_image is None:
            return {
                "status": "error",
                "error": f"SAM3 found no objects for prompt: '{prompt}'",
            }

        # --- Step 6: Crop to bounding box ---
        crop_bbox = compute_bounding_box(mask_image, padding=bbox_padding)
        result_image = result_image.crop(crop_bbox)
        logger.info("Cropped: %dx%d", result_image.width, result_image.height)

        # --- Step 7: Encode ---
        result_base64 = image_to_base64(result_image)

        elapsed = time.time() - t_start
        logger.info("Done in %.2fs", elapsed)

        return {
            "status": "success",
            "image_base64": result_base64,
            "width": result_image.width,
            "height": result_image.height,
            "num_instances": num_found,
            "rotated": rotated,
            "processing_time_s": round(elapsed, 2),
        }

    except requests.exceptions.RequestException as e:
        logger.error("Download failed: %s", e)
        return {"status": "error", "error": f"Image download failed: {e}"}
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_models()
    runpod.serverless.start({"handler": handler})
