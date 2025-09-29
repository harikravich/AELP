#!/usr/bin/env python3
from __future__ import annotations
"""
Build simple proof overlay PNGs and an end-card image from brand pack.
Outputs under AELP2/outputs/proof_images and AELP2/outputs/endcards.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BRAND = json.loads((ROOT / 'AELP2' / 'branding' / 'brand_pack.json').read_text())
UICFG = json.loads((ROOT / 'AELP2' / 'branding' / 'ui_overlays.json').read_text())
PROOF_DIR = ROOT / 'AELP2' / 'outputs' / 'proof_images'
END_DIR = ROOT / 'AELP2' / 'outputs' / 'endcards'

def font(size: int) -> ImageFont.FreeTypeFont:
    # Rely on default fonts; Inter may not be installed in runtime
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except Exception:
        return ImageFont.load_default()

def rounded_rect(im: Image.Image, xy, radius, fill, outline=None, width=2):
    draw = ImageDraw.Draw(im)
    x0,y0,x1,y1 = xy
    draw.rounded_rectangle([x0,y0,x1,y1], radius=radius, fill=fill, outline=outline, width=width)

def make_proof_images():
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    for ov in UICFG['overlays']:
        im = Image.new('RGBA', (720, 1280), (0,0,0,0))
        draw = ImageDraw.Draw(im)
        text = ov['text']
        f = font(36)
        bbox = draw.textbbox((0,0), text, font=f)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        pad = 16
        x = int(ov['placement']['x'] * im.width)
        y = int(ov['placement']['y'] * im.height)
        w = int(ov['placement']['w'] * im.width)
        bx0 = x
        bx1 = min(im.width - 24, x + w)
        by0 = y
        by1 = y + th + 2*pad
        rounded_rect(im, (bx0,by0,bx1,by1), radius=18, fill=(20,20,20,210), outline=(255,255,255,160), width=2)
        draw.text((bx0+pad, by0+pad), text, font=f, fill=(255,255,255,255))
        dest = PROOF_DIR / f"proof_{ov['id']}.png"
        im.save(dest)

def make_end_card():
    END_DIR.mkdir(parents=True, exist_ok=True)
    colors = BRAND['colors']
    cta = BRAND['end_card']['cta_variants'][0]
    disclaimer = BRAND['policy']['disclaimer_default']
    im = Image.new('RGB', (720, 1280), tuple(int(colors['bg_dark'].lstrip('#')[i:i+2],16) for i in (0,2,4)))
    draw = ImageDraw.Draw(im)
    f_title = font(56)
    f_sub = font(32)
    f_disc = font(20)
    title = "Aura protects what matters"  # non-claim line
    sub = cta
    # Title
    draw.text((60, 420), title, font=f_title, fill=(255,255,255))
    # CTA button
    btn_w, btn_h = 420, 80
    bx0, by0 = 60, 520
    rounded_rect(im, (bx0, by0, bx0+btn_w, by0+btn_h), radius=20, fill=tuple(int(colors['primary'].lstrip('#')[i:i+2],16) for i in (0,2,4)))
    draw.text((bx0+24, by0+20), sub, font=f_sub, fill=(255,255,255))
    # Disclaimer
    draw.text((60, 1160), disclaimer, font=f_disc, fill=(200,200,200))
    (END_DIR / 'endcard_default.png').write_bytes(im.tobytes())
    im.save(END_DIR / 'endcard_default.png')

def main():
    make_proof_images()
    make_end_card()
    print({
        'proof_images': len(list(PROOF_DIR.glob('*.png'))),
        'endcards': len(list(END_DIR.glob('*.png')))
    })

if __name__=='__main__':
    main()
