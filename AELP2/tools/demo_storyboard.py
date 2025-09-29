#!/usr/bin/env python3
from __future__ import annotations
"""
Generate a 4-frame 9:16 storyboard from the Identity KB:
  1) Hook (non-claim benefit)
  2) Proof (generic)
  3) Claim + mandatory disclaimer
  4) End card + CTA

Outputs PNGs to AELP2/outputs/demo_ads/01_hook.png .. 04_cta.png.

No external assets; uses Pillow only.
"""
import json
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / 'AELP2' / 'knowledge' / 'products' / 'identity.json'
OUTDIR = ROOT / 'AELP2' / 'outputs' / 'demo_ads'

W, H = 1080, 1920  # 9:16 vertical


def load_kb():
    d = json.loads(KB.read_text())
    claim = d['approved_claims'][0]
    cta = (d.get('approved_ctas') or ['Get Protected'])[0]
    non_claim = (d.get('non_claim_benefit_lines') or [
        'Get alerts fast so you can act before damage spreads.'
    ])[0]
    return non_claim, claim['text'], claim.get('mandatory_disclaimer', ''), cta


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # Try common system fonts; Pillow falls back to default if not found
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, fnt: ImageFont.FreeTypeFont, fill=(255, 255, 255)):
    # Simple centered multiline text with word-wrapping at ~24 chars/line
    max_w = box[2] - box[0]
    words = text.split()
    lines = []
    cur = []
    for w in words:
        test = ' '.join(cur + [w])
        tw, th = draw.textbbox((0, 0), test, font=fnt)[2:]
        if tw <= max_w:
            cur.append(w)
        else:
            if cur:
                lines.append(' '.join(cur))
            cur = [w]
    if cur:
        lines.append(' '.join(cur))
    total_h = sum(draw.textbbox((0, 0), ln, font=fnt)[3] for ln in lines) + (len(lines) - 1) * 10
    y = box[1] + (box[3] - box[1] - total_h) // 2
    for ln in lines:
        tw, th = draw.textbbox((0, 0), ln, font=fnt)[2:]
        x = box[0] + (max_w - tw) // 2
        draw.text((x, y), ln, font=fnt, fill=fill)
        y += th + 10


def panel(bg: Tuple[int, int, int], title: str, body: str, disclaimer: str | None = None, cta: str | None = None, endcard=False) -> Image.Image:
    im = Image.new('RGB', (W, H), bg)
    dr = ImageDraw.Draw(im)
    # Safe area
    margin = 64
    # Title
    if title:
        f_title = font(80, bold=True)
        draw_centered_text(dr, (margin, 180, W - margin, 600), title, f_title)
    # Body
    if body:
        f_body = font(56, bold=False)
        draw_centered_text(dr, (margin, 640, W - margin, 1180), body, f_body)
    # CTA or brand end card
    if endcard:
        # Brand
        f_brand = font(88, bold=True)
        brand = 'Aura'
        tw, th = dr.textbbox((0, 0), brand, font=f_brand)[2:]
        dr.text(((W - tw)//2, 1220), brand, font=f_brand, fill=(255, 255, 255))
        if cta:
            btn_w, btn_h = 720, 140
            x0 = (W - btn_w)//2; y0 = 1400
            dr.rounded_rectangle([x0, y0, x0 + btn_w, y0 + btn_h], radius=24, fill=(255, 255, 255))
            f_cta = font(60, bold=True)
            tw, th = dr.textbbox((0, 0), cta, font=f_cta)[2:]
            dr.text((x0 + (btn_w - tw)//2, y0 + (btn_h - th)//2), cta, font=f_cta, fill=(20, 20, 20))
        # URL
        f_url = font(42)
        url = 'aura.com/identity-theft-protection'
        tw, th = dr.textbbox((0, 0), url, font=f_url)[2:]
        dr.text(((W - tw)//2, 1560), url, font=f_url, fill=(230, 230, 230))
    # Disclaimer ribbon
    if disclaimer:
        ribbon_h = 180
        dr.rectangle([0, H - ribbon_h, W, H], fill=(12, 12, 12))
        f_disc = font(28)
        # simple left-aligned wrap
        x, y = 40, H - ribbon_h + 30
        max_w = W - 80
        words = disclaimer.split()
        line = ''
        for w in words:
            test = (line + ' ' + w).strip()
            tw, th = dr.textbbox((0, 0), test, font=f_disc)[2:]
            if tw <= max_w:
                line = test
            else:
                dr.text((x, y), line, font=f_disc, fill=(220, 220, 220))
                y += th + 6
                line = w
        if line:
            dr.text((x, y), line, font=f_disc, fill=(220, 220, 220))
    return im


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    nonclaim, claim, disclaimer, cta = load_kb()
    frames = [
        ( (32, 32, 36), "Did you get breached?", nonclaim, None, None, False ),
        ( (24, 28, 60), "See it. Fix it.", "Check exposures and get fast alerts.", None, None, False ),
        ( (20, 60, 40), "On eligible plans:", claim, disclaimer, None, False ),
        ( (15, 15, 18), "", "", None, cta, True ),
    ]
    names = ['01_hook.png', '02_proof.png', '03_claim.png', '04_cta.png']
    for (bg, title, body, disc, cta_text, endcard), name in zip(frames, names):
        im = panel(bg, title, body, disclaimer=disc, cta=cta_text, endcard=endcard)
        im.save(OUTDIR / name, format='PNG')
    print(json.dumps({
        'out_dir': str(OUTDIR),
        'frames': names,
        'note': 'Use demo_make_ad_vm.sh on the VM to compile into MP4 with audio.'
    }, indent=2))


if __name__ == '__main__':
    main()

