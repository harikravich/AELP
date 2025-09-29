#!/usr/bin/env python3
from __future__ import annotations
"""
Render timed quiz-style overlay cards on a 9:16 video using ffmpeg drawbox/drawtext.

Usage:
  python3 AELP2/tools/render_quiz_overlay.py \
      --video AELP2/outputs/renders/runway/clip.mp4 \
      --out AELP2/outputs/finals/clip_quiz.mp4 \
      [--pack AELP2/creative/overlays/quiz_pack.json]

Notes:
- Picks a brand font automatically from AELP2/assets/brand/fonts if available,
  otherwise falls back to DejaVuSans.
- Adds top title band and two-line options band near bottom.
"""
import json, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def pick_font() -> str:
    # Try brand fonts
    fonts_dir = ROOT/'AELP2'/'assets'/'brand'/'fonts'
    cand = None
    if fonts_dir.exists():
        for ext in ('.ttf','.otf'):
            for p in sorted(fonts_dir.glob(f'*{ext}')):
                if p.name.startswith('._'):
                    continue
                cand = p; break
            if cand: break
    if cand:
        return str(cand)
    # Fallbacks
    for p in (
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ):
        if p.exists():
            return str(p)
    return 'sans-serif'

def esc(t: str) -> str:
    return t.replace(':','\\:').replace("'","\\'")

def _split_color_alpha(hex_or_rgba: str, fallback_rgb: str, fallback_a: float) -> tuple[str, float]:
    s=(hex_or_rgba or '').strip()
    if s.startswith('#') and len(s) in (7,9):
        rgb=s[:7]
        if len(s)==9:
            try:
                a=int(s[7:9],16)/255.0
            except Exception:
                a=fallback_a
        else:
            a=fallback_a
        return rgb, max(0.0,min(1.0,a))
    return (fallback_rgb, fallback_a)

def build_filters(cards: list[dict], style: dict, fontfile: str) -> list[str]:
    filters = [
        # optional subtle vignette to improve readability
        "[0:v]format=yuv420p[v0]"
    ]
    label_in = '[v0]'
    label_out = 'vcur'
    chain = label_in
    # Use named colors for robustness across ffmpeg builds; vary only alpha
    _, top_a = _split_color_alpha(style.get('top_color','#000000CC'), '#000000', 0.55)
    _, bot_a = _split_color_alpha(style.get('bot_color','#00000099'), '#000000', 0.45)
    top_rgb = 'black'
    bot_rgb = 'black'
    font_rgb = 'white'
    for idx, c in enumerate(cards):
        s, e = float(c['start']), float(c['end'])
        title = c.get('title','')
        opts = c.get('options', [])
        # Top band
        draw_top = (
            f"drawbox=x=0:y=0:w=iw:h=ih*0.14:color={top_rgb}@{top_a}:t=fill:enable='between(t,{s},{e})',"
            f"drawtext=fontfile={fontfile}:text='{esc(title)}':fontcolor={font_rgb}:fontsize={style.get('title_size',68)}:x=(w-tw)/2:y=h*0.03:enable='between(t,{s},{e})'"
        )
        # Bottom options (up to 2 lines)
        opt1 = esc(opts[0]) if len(opts)>0 else ''
        opt2 = esc(opts[1]) if len(opts)>1 else ''
        draw_bot = (
            f",drawbox=x=iw*0.07:y=ih*0.78:w=iw*0.86:h=ih*0.18:color={bot_rgb}@{bot_a}:t=fill:enable='between(t,{s},{e})'"
            + (f",drawtext=fontfile={fontfile}:text='{opt1}':fontcolor={font_rgb}:fontsize={style.get('opt_size',50)}:x=(w-tw)/2:y=h*0.79:enable='between(t,{s},{e})'" if opt1 else '')
            + (f",drawtext=fontfile={fontfile}:text='{opt2}':fontcolor={font_rgb}:fontsize={style.get('opt_size',50)}:x=(w-tw)/2:y=h*0.86:enable='between(t,{s},{e})'" if opt2 else '')
        )
        filters.append(f"{chain}{draw_top}{draw_bot}[{label_out}{idx}]")
        chain = f"[{label_out}{idx}]"
    filters.append(f"{chain}format=yuv420p[vout]")
    return filters

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pack', default=str(ROOT/'AELP2'/'creative'/'overlays'/'quiz_pack.json'))
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out)
    pack = json.loads(Path(args.pack).read_text())
    fontfile = pick_font()
    filters = build_filters(pack['cards'], pack.get('style',{}), fontfile)

    cmd = ['ffmpeg','-y','-i',str(video), '-filter_complex', ';'.join(filters), '-map','[vout]','-map','0:a?','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k', str(out)]
    subprocess.run(cmd, check=True)
    print(json.dumps({'in': str(video), 'out': str(out), 'pack': args.pack, 'font': fontfile}, indent=2))

if __name__=='__main__':
    main()
