#!/usr/bin/env python3
from __future__ import annotations
"""
Detect objects on mid-frames using YOLOv8n (CPU). Reports counts of person and cell phone.
Outputs: AELP2/reports/object_counts.jsonl
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FIN=ROOT/'AELP2'/'outputs'/'finals'
KFD=ROOT/'AELP2'/'reports'/'keyframes'
OUT=ROOT/'AELP2'/'reports'/'object_counts.jsonl'

def main():
    try:
        from ultralytics import YOLO
    except Exception as e:
        OUT.write_text(json.dumps({'error': 'ultralytics not available', 'detail': str(e)}, indent=2))
        print(json.dumps({'error': 'ultralytics not available'})); return
    model=YOLO('yolov8n.pt')
    with OUT.open('w') as f:
        for stemdir in sorted(KFD.iterdir()):
            if not stemdir.is_dir():
                continue
            imgs=sorted(stemdir.glob('*.jpg'))
            if not imgs:
                continue
            # use up to first 3 frames
            persons=phones=0
            for fp in imgs[:3]:
                im=Image.open(fp).convert('RGB')
                res=model.predict(source=np.array(im), verbose=False)[0]
                for cls in res.boxes.cls.tolist():
                    cls=int(cls)
                    if cls==0: persons+=1     # person
                    if cls==67: phones+=1     # cell phone in COCO
            f.write(json.dumps({'file': stemdir.name+'.mp4', 'persons': persons, 'phones': phones})+'\n')
    print(json.dumps({'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

