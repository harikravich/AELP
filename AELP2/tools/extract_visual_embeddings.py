#!/usr/bin/env python3
from __future__ import annotations
"""
Compute OpenCLIP embeddings for keyframes under AELP2/reports/keyframes/* and
save a video-level mean embedding per stem.

Outputs: AELP2/reports/creative_visual_embeddings/<stem>.npy
         AELP2/reports/creative_visual_embeddings/index.json
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
USE_OPENCLIP=True
try:
    import torch  # type: ignore
    import open_clip  # type: ignore
except Exception:
    USE_OPENCLIP=False

ROOT=Path(__file__).resolve().parents[2]
KFD=ROOT/'AELP2'/'reports'/'keyframes'
OUTD=ROOT/'AELP2'/'reports'/'creative_visual_embeddings'

def to_tensor(im: Image.Image, preprocess):
    return preprocess(im.convert('RGB'))

def main():
    if USE_OPENCLIP:
        device='cpu'
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
        )
        model.eval()
    OUTD.mkdir(parents=True, exist_ok=True)
    index={}
    for stemdir in sorted(KFD.iterdir()):
        if not stemdir.is_dir():
            continue
        imgs=sorted(stemdir.glob('*.jpg'))
        if not imgs:
            continue
        xs=[]
        if USE_OPENCLIP:
            with torch.no_grad():
                for fp in imgs[:8]:  # cap to 8 frames for speed
                    try:
                        im=Image.open(fp)
                    except Exception:
                        continue
                    x=to_tensor(im, preprocess).unsqueeze(0)
                    feat=model.encode_image(x)
                    feat/=feat.norm(dim=-1, keepdim=True)
                    xs.append(feat.cpu().numpy()[0])
        else:
            # Fallback: 64-dim color histogram per frame, L2-normalized
            for fp in imgs[:8]:
                try:
                    im=Image.open(fp).convert('RGB').resize((224,224))
                except Exception:
                    continue
                arr=np.asarray(im)
                hist=[]
                for c in range(3):
                    h,_=np.histogram(arr[:,:,c], bins=21, range=(0,255), density=True)
                    hist.extend(h)
                xs.append(np.array(hist, dtype=np.float32))
        if not xs:
            continue
        emb=np.mean(np.stack(xs,0),0)
        out=OUTD/f"{stemdir.name}.npy"
        np.save(out, emb)
        index[stemdir.name]=str(out)
        print(f"embedded {stemdir.name}")
    (OUTD/'index.json').write_text(json.dumps(index, indent=2))
    print(json.dumps({'count': len(index), 'out_dir': str(OUTD)}, indent=2))

if __name__=='__main__':
    main()
