#!/usr/bin/env python3
from __future__ import annotations
"""
Parse brand guide PDFs to extract a usable brand_config.json for the creative engine.

Heuristics:
- Extract text via `pdftotext` if available; otherwise fall back to a naive binary scan.
- Find hex colors (#RRGGBB) and frequent RGB triplets; dedupe and keep top 8.
- Detect font family names by matching common tokens near words like Typography/Font/Typeface.
- Save AELP2/creative/brand_config.json with palette, primary/secondary, CTA defaults,
  and link to installed brand font files if present.
"""
import json, re, subprocess, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ROOT/'AELP2'/'assets'/'brand'/'guides'
FONTS  = ROOT/'AELP2'/'assets'/'brand'/'fonts'
OUT    = ROOT/'AELP2'/'creative'/'brand_config.json'

HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}")
RGB_RE = re.compile(r"\b(?:RGB|R\s*G\s*B)\s*[:\-]?\s*(\d{1,3})\s*[,/ ]\s*(\d{1,3})\s*[,/ ]\s*(\d{1,3})\b", re.I)
FONT_HINT = re.compile(r"(Typography|Typeface|Font|Primary\s+typeface)\s*[:\-]?\s*([A-Za-z0-9 \-]+)")
FONT_NAME = re.compile(r"\b([A-Z][A-Za-z0-9\-]+(?: [A-Z][A-Za-z0-9\-]+){0,2})\b")

def pdftotext(path: Path) -> str:
    try:
        out = subprocess.check_output(['pdftotext','-layout',str(path),'-'], stderr=subprocess.DEVNULL)
        return out.decode('utf-8','ignore')
    except Exception:
        return ''

def rgb_to_hex(r:int,g:int,b:int)->str:
    r=max(0,min(255,r)); g=max(0,min(255,g)); b=max(0,min(255,b))
    return f"#{r:02X}{g:02X}{b:02X}"

def scan_pdf_text(txt: str) -> dict:
    colors = []
    colors += HEX_RE.findall(txt)
    for r,g,b in RGB_RE.findall(txt):
        colors.append(rgb_to_hex(int(r),int(g),int(b)))
    # frequency order keeping original case
    counts = {}
    for c in colors:
        cu=c.upper()
        counts[cu]=counts.get(cu,0)+1
    palette = [c for c,_ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    # fonts: try explicit hints, else collect candidates near Typography sections
    fonts=set()
    for m in FONT_HINT.finditer(txt):
        name=m.group(2).strip()
        if name:
            fonts.add(name)
    # fallback: collect Title‑case tokens following the word Font/Typeface
    for m in re.finditer(r"(?:Font|Typeface)\s*[:\-]?\s*([\w \-]{1,40})", txt, re.I):
        tail=m.group(1)
        nm=FONT_NAME.search(tail)
        if nm:
            fonts.add(nm.group(1))

    return {'palette': palette, 'fonts': sorted(fonts)}

def pick_defaults(palette: list[str]) -> dict:
    # choose the first as primary, next as secondary. Provide sensible text colors.
    primary = palette[0] if palette else '#0EA5E9'
    secondary = palette[1] if len(palette)>1 else '#F59E0B'
    return {
        'palette': palette[:8],
        'primary': primary,
        'secondary': secondary,
        'overlay_dark': '#000000CC',
        'overlay_light': '#FFFFFFCC',
        'text_on_dark': '#FFFFFF',
        'text_on_light': '#111111',
        'cta_text': 'Start Free Trial',
        'cta_bg': primary,
        'cta_text_color': '#FFFFFF'
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(GUIDES))
    args = ap.parse_args()

    gdir = Path(args.dir)
    pdfs = sorted(gdir.glob('**/*.pdf'))
    if not pdfs:
        print(json.dumps({'error':'no brand PDFs found', 'dir': str(gdir)}, indent=2)); return

    combined={'palette':[], 'fonts':[]}
    for pdf in pdfs:
        txt = pdftotext(pdf)
        if not txt:
            continue
        res=scan_pdf_text(txt)
        combined['palette'] += res['palette']
        combined['fonts'] += res['fonts']

    # dedupe while preserving order
    seen=set(); palette=[]
    for c in combined['palette']:
        cu=c.upper()
        if cu not in seen:
            seen.add(cu); palette.append(cu)
    fonts=sorted(set(combined['fonts']))

    cfg = pick_defaults(palette)
    # link installed brand font files if present
    font_files=[str(p) for p in FONTS.glob('**/*') if p.suffix.lower() in ('.ttf','.otf')]
    # filter out accidental numeric-only captures
    fonts=[f for f in fonts if any(ch.isalpha() for ch in f)]
    cfg.update({
        'font_primary': fonts[0] if fonts else None,
        'font_files': font_files,
        'source_guides': [str(p) for p in pdfs]
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=2))
    print(json.dumps({'brand_config': str(OUT), 'palette': cfg['palette'], 'font_primary': cfg['font_primary'], 'font_files': len(font_files)}, indent=2))

if __name__=='__main__':
    main()
