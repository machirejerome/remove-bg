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


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right: smallest x-y
    rect[3] = pts[np.argmax(d)]   # bottom-left: largest x-y
    return rect


def perspective_correct(image: Image.Image, mask_image: Image.Image,
                        padding: int = 5) -> tuple:
    """
    Straighten a rectangular postcard using its mask.
    
    1. Find the largest contour in the mask
    2. Approximate to a 4-corner polygon
    3. Apply perspective transform to get a perfect rectangle
    
    Args:
        image: Full-size result image (RGBA or RGB)
        mask_image: Grayscale mask (L mode)
        padding: Extra pixels around the postcard to avoid cutting edges
    
    Returns:
        (corrected_image, corrected_mask) or (original_image, original_mask)
        if correction fails
    """
    mask_np = np.array(mask_image)
    
    # Find contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("Perspective: no contours found")
        return image, mask_image
    
    # Take largest contour by area
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate polygon — try to get 4 corners
    peri = cv2.arcLength(contour, True)
    approx = None
    
    # Try different epsilon values to find exactly 4 corners
    for eps_mult in [0.02, 0.03, 0.04, 0.05, 0.01]:
        candidate = cv2.approxPolyDP(contour, eps_mult * peri, True)
        if len(candidate) == 4:
            approx = candidate
            break
    
    if approx is None or len(approx) != 4:
        # Fallback: use the minimum area rotated rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = box.reshape(4, 1, 2).astype(np.float32)
        logger.info("Perspective: using minAreaRect fallback")
    
    # Get the 4 source corners, ordered consistently
    src_pts = approx.reshape(4, 2).astype(np.float32)
    src_pts = order_corners(src_pts)
    
    # Calculate target dimensions from the source corners
    # Width: max of top edge and bottom edge
    width_top = np.linalg.norm(src_pts[1] - src_pts[0])
    width_bot = np.linalg.norm(src_pts[2] - src_pts[3])
    target_w = int(max(width_top, width_bot)) + 2 * padding
    
    # Height: max of left edge and right edge
    height_left = np.linalg.norm(src_pts[3] - src_pts[0])
    height_right = np.linalg.norm(src_pts[2] - src_pts[1])
    target_h = int(max(height_left, height_right)) + 2 * padding
    
    # Target rectangle with padding
    dst_pts = np.array([
        [padding, padding],
        [target_w - padding - 1, padding],
        [target_w - padding - 1, target_h - padding - 1],
        [padding, target_h - padding - 1],
    ], dtype=np.float32)
    
    # Compute perspective transform
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp the image
    img_np = np.array(image)
    if image.mode == "RGBA":
        # Warp with transparent background
        warped = cv2.warpPerspective(
            img_np, matrix, (target_w, target_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        corrected_image = Image.fromarray(warped, "RGBA")
    else:
        warped = cv2.warpPerspective(
            img_np, matrix, (target_w, target_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        corrected_image = Image.fromarray(warped, "RGB")
    
    # Warp the mask too
    warped_mask = cv2.warpPerspective(
        mask_np, matrix, (target_w, target_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    corrected_mask = Image.fromarray(warped_mask, "L")
    
    logger.info(
        "Perspective corrected: %dx%d → %dx%d",
        image.width, image.height, target_w, target_h,
    )
    return corrected_image, corrected_mask


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
        "auto_rotate": true,                   // portrait → landscape
        "perspective_correct": false            // straighten skewed postcards
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
        do_perspective = bool(inp.get("perspective_correct", False))

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

        # --- Step 5b: Perspective correction (optional) ---
        perspective_applied = False
        if do_perspective:
            result_image, mask_image = perspective_correct(
                result_image, mask_image, padding=bbox_padding,
            )
            perspective_applied = True

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
            "perspective_corrected": perspective_applied,
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
