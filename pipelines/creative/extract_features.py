#!/usr/bin/env python3
"""
Creative Feature Extraction
- Sharpness (variance of Laplacian) via OpenCV or numpy fallback
- Objects (YOLOv8) if ultralytics + weights available
- Motion (optical flow magnitude) for videos if OpenCV available
- Text (OCR) via pytesseract if available
- CLIP embeddings via open_clip if available

CLI:
  python pipelines/creative/extract_features.py --in assets/ --out artifacts/creative_features.parquet
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image

# Optional deps
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
try:
    import open_clip  # type: ignore
    import torch  # type: ignore
except Exception:
    open_clip = None  # type: ignore
    torch = None  # type: ignore


def variance_of_laplacian(img: np.ndarray) -> float:
    if cv2 is None:
        # simple high-frequency proxy if cv2 not available
        arr = img.astype(np.float32)
        fx = np.abs(np.diff(arr, axis=1)).mean()
        fy = np.abs(np.diff(arr, axis=0)).mean()
        return float((fx + fy) / 2.0)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def extract_image_basic_features(path: Path) -> Dict[str, Any]:
    img = Image.open(path).convert('L')  # grayscale for sharpness
    arr = np.array(img)
    features: Dict[str, Any] = {
        'asset_path': str(path),
        'sharpness': variance_of_laplacian(arr),
        'width': img.width,
        'height': img.height,
        'aspect_ratio': img.width / img.height if img.height else 0.0,
    }
    return features

def extract_video_basic_features(path: Path) -> Dict[str, Any]:
    if cv2 is None:
        return {}
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {}
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        idxs = [int(n*0.1), int(n*0.5), int(n*0.9)] if n > 0 else [0]
        sharp_vals = []
        motion = 0.0
        prev = None
        for i in range(max(idxs)+1):
            ret, frame = cap.read()
            if not ret:
                break
            if i in idxs:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharp_vals.append(variance_of_laplacian(gray))
            gsmall = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diff = cv2.absdiff(gsmall, prev)
                motion += float(diff.mean())
            prev = gsmall
        cap.release()
        return {
            'asset_path': str(path),
            'sharpness': float(np.mean(sharp_vals) if sharp_vals else 0.0),
            'width': w,
            'height': h,
            'aspect_ratio': (w / h) if h else 0.0,
            'motion_mag': motion / max(1, len(idxs))
        }
    except Exception:
        return {}


def detect_objects_yolo(path: Path) -> Dict[str, Any]:
    if YOLO is None:
        return {}
    try:
        # ultralytics will auto-download 'yolov8n.pt' if missing and use GPU when available
        model = YOLO('yolov8n.pt')
        res = model(str(path), verbose=False)
        names = model.model.names  # type: ignore
        counts: Dict[str, int] = {}
        for r in res:
            for c in r.boxes.cls.cpu().numpy().astype(int):
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                counts[label] = counts.get(label, 0) + 1
        # Flatten top-5
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return {f'obj_{i+1}': k for i, (k, _) in enumerate(top)} | {f'obj_{k}_count': v for k, v in counts.items()}
    except Exception:
        return {}


def ocr_text(path: Path) -> Dict[str, Any]:
    if pytesseract is None:
        return {}
    try:
        txt = pytesseract.image_to_string(Image.open(path))
        txt = ' '.join(txt.split())[:512]
        return {'ocr_text': txt}
    except Exception:
        return {}


def clip_embed(path: Path) -> Dict[str, Any]:
    if open_clip is None or torch is None:
        return {}
    if os.getenv('ENABLE_CLIP', '1') != '1':
        return {}
    try:
        device = 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
        model.eval()
        # If video, sample a middle frame via OpenCV and convert to PIL
        if path.suffix.lower() in {'.mp4','.mov','.webm','.mkv','.avi'} and cv2 is not None:
            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            mid = int(n*0.5) if n>0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
            else:
                img_pil = Image.new('RGB', (224,224), color=(0,0,0))
        else:
            img_pil = Image.open(path).convert('RGB')
        img = preprocess(img_pil).unsqueeze(0)
        with torch.no_grad():
            img = img.to(device)
            feats = model.encode_image(img)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = feats[0].cpu().numpy()[:16]  # truncate to 16 for storage
        return {f'clip_{i}': float(v) for i, v in enumerate(vec)}
    except Exception:
        return {}


def extract_from_path(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {'.mp4','.mov','.webm','.mkv','.avi'}:
        base = extract_video_basic_features(path)
    else:
        base = extract_image_basic_features(path)
    base |= detect_objects_yolo(path)
    base |= ocr_text(path)
    base |= clip_embed(path)
    return base


def run(input_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for p in input_dir.rglob('*'):
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            records.append(extract_from_path(p))
    return pd.DataFrame.from_records(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='input_dir', required=True)
    ap.add_argument('--out', dest='out', required=True)
    args = ap.parse_args()

    df = run(Path(args.input_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"Wrote creative features for {len(df)} assets to {out}")


if __name__ == '__main__':
    main()
