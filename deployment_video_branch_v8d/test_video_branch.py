from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from video_branch_v8d_inference import predict_video_emotion


DEPLOYMENT_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = DEPLOYMENT_DIR / "video_branch_v8d_bundle.pth"
EXPECTED_CLASSES = {"happy", "sad", "stressed", "neutral"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True, help="Path to a test video")
    args = parser.parse_args()

    status = "FAIL"
    result = None
    error = ""
    missing_assets: list[str] = []

    try:
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"Missing bundle: {BUNDLE_PATH}")
        bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
        for key, value in bundle.get("asset_paths", {}).items():
            values = value if isinstance(value, list) else [value]
            for rel_path in values:
                asset_path = DEPLOYMENT_DIR / rel_path
                if not asset_path.exists():
                    missing_assets.append(f"{key}: {rel_path}")
        if missing_assets:
            raise FileNotFoundError("Missing assets: " + "; ".join(missing_assets))

        result = predict_video_emotion(str(args.video))
        probabilities = result["probabilities"]
        if set(probabilities) != EXPECTED_CLASSES:
            raise AssertionError(f"Unexpected probability classes: {sorted(probabilities)}")
        total = sum(float(v) for v in probabilities.values())
        if abs(total - 1.0) > 1e-4:
            raise AssertionError(f"Probabilities do not sum to 1: {total}")
        if result["pred_label"] not in EXPECTED_CLASSES:
            raise AssertionError(f"Invalid pred_label: {result['pred_label']}")
        if abs(float(result["confidence"]) - max(float(v) for v in probabilities.values())) > 1e-8:
            raise AssertionError("confidence is not the highest probability")
        status = "PASS"
    except Exception as exc:
        error = str(exc)

    print(f"DEPLOYMENT_TEST={status}")
    if result is not None:
        print(f"pred_label={result['pred_label']}")
        print("probabilities=" + json.dumps(result["probabilities"], sort_keys=True))
        print(f"confidence={result['confidence']}")
        print(f"model_name={result['model_name']}")
        print(f"alpha={result['alpha']}")
    else:
        print("pred_label=")
        print("probabilities={}")
        print("confidence=")
        print("model_name=V8D")
        print("alpha=0.68")
    if error:
        print(f"error={error}")
    print(f"ANY_MISSING_ASSETS={missing_assets if missing_assets else 'none'}")
    print(f"READY_FOR_MULTIMODAL_INTEGRATION={'YES' if status == 'PASS' and not missing_assets else 'NO'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
