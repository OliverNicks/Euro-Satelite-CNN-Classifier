"""Command-line inference tool for EuroSAT satellite image classification.

Usage:
    python -m src.predict --image samples/Forest.jpg
    python -m src.predict --image samples/ --top-k 3
    python -m src.predict --image samples/SeaLake.jpg --json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union
from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import get_inference_transforms
from src.model import CLASS_DESCRIPTIONS, CLASS_NAMES, IDX_TO_CLASS, SateliteCNN, load_model


def predict_image(
    image_input: Union[str, Image.Image],
    model: SateliteCNN,
    device: torch.device,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Run inference on a single image and return top-k predictions with probabilities."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
        image_name = os.path.basename(image_input)
    else:
        image = image_input.convert("RGB")
        image_name = "in-memory-image"

    transform = get_inference_transforms()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

    predictions: List[Dict[str, Any]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        class_name = IDX_TO_CLASS[idx]
        predictions.append({
            "class": class_name,
            "confidence": round(prob * 100, 2),
            "probability": round(prob, 4),
            "description": CLASS_DESCRIPTIONS.get(class_name, ""),
        })

    return {
        "image": image_name,
        "predicted_class": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict land use/land cover class from satellite imagery."
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        default="app/satellite_model_9715.pth",
        help="Path to trained PyTorch weights (.pth).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of top class predictions to display (default: 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output predictions as JSON.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Computing device (default: auto-detect).",
    )

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    try:
        model = load_model(weights_path=args.weights, device=device)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Collect images
    if os.path.isdir(args.image):
        image_paths = [
            os.path.join(args.image, f)
            for f in sorted(os.listdir(args.image))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]
    elif os.path.isfile(args.image):
        image_paths = [args.image]
    else:
        print(f"Error: Target path '{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not image_paths:
        print(f"No valid image files found in '{args.image}'.", file=sys.stderr)
        sys.exit(1)

    results = [predict_image(p, model, device, top_k=args.top_k) for p in image_paths]

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        print("\n" + "=" * 65)
        print(" EuroSAT Satellite Land Cover Classification Results")
        print("=" * 65)
        for res in results:
            print(f"\nImage: {res['image']}")
            print(f"--> Predicted: {res['predicted_class']} ({res['confidence']}%)")
            print("    Top Candidates:")
            for rank, item in enumerate(res["top_predictions"], 1):
                bar = "#" * int(item["confidence"] / 5)
                print(f"    [{rank}] {item['class']:<22} {item['confidence']:>6.2f}%  {bar}")
        print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
