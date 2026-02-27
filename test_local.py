"""
Local test script – runs the SAM3 handler without RunPod.

Usage:
    python test_local.py [IMAGE_URL] [PROMPT]

Prerequisites:
    - SAM3 library at SAM3_LIB_DIR (default: ./sam3_lib)
    - SAM3 checkpoint at SAM3_CHECKPOINT (default: ./sam3_models/sam3.pt)
    - BPE tokenizer at SAM3_BPE_PATH (default: ./sam3_lib/assets/bpe_simple_vocab_16e6.txt.gz)
    
    Or set environment variables:
        export SAM3_LIB_DIR=/path/to/sam3_lib
        export SAM3_CHECKPOINT=/path/to/sam3.pt
        export SAM3_BPE_PATH=/path/to/bpe_simple_vocab_16e6.txt.gz
"""

import sys
import os
import base64

# Set default paths for local testing
if "SAM3_LIB_DIR" not in os.environ:
    os.environ["SAM3_LIB_DIR"] = os.path.join(os.path.dirname(__file__), "sam3_lib")
if "SAM3_CHECKPOINT" not in os.environ:
    os.environ["SAM3_CHECKPOINT"] = os.path.join(os.path.dirname(__file__), "sam3_models", "sam3.pt")

# Mock runpod for local testing
import types
mock_runpod = types.ModuleType("runpod")
mock_runpod.serverless = types.SimpleNamespace(start=lambda config: None)
sys.modules["runpod"] = mock_runpod

from handler import load_models, handler

DEFAULT_URL = "https://res.cloudinary.com/dghf8rbpk/image/upload/v1768812708/0084005157Vorderseite_ywfjan.jpg"
DEFAULT_PROMPT = "one single postcard"


def main():
    image_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT

    print(f"🖼  URL: {image_url}")
    print(f"📝 Prompt: {prompt}")
    print(f"⚙️  Exact ComfyUI defaults: threshold=0.05, max_segments=1, Merged, Alpha")
    print()

    print("⏳ Loading SAM3...")
    load_models()

    print("🚀 Running...")
    result = handler({
        "input": {
            "image_url": image_url,
            "prompt": prompt,
            # All other params use ComfyUI defaults
        }
    })

    print()
    print(f"Status: {result.get('status')}")

    if result.get("status") == "success":
        print(f"Size: {result['width']}x{result['height']}")
        print(f"Instances: {result['num_instances']}")
        print(f"Rotated: {result['rotated']}")
        print(f"Time: {result['processing_time_s']}s")

        img_data = base64.b64decode(result["image_base64"])
        with open("test_output.png", "wb") as f:
            f.write(img_data)
        print("✅ Saved: test_output.png")
    else:
        print(f"❌ {result.get('error')}")


if __name__ == "__main__":
    main()
