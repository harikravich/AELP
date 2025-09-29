#!/usr/bin/env python3
from __future__ import annotations
"""
Generate 9:16 hook clips on Runway (Gen‑4 Turbo) using image_to_video.

Reads prompts from AELP2/prompts/runway_prompts.json.
Creates a small 720x1280 seed PNG (neutral) as promptImage (data URI).
Submits tasks, polls until SUCCEEDED/FAILED, downloads MP4s to:
  AELP2/outputs/renders/runway/<id>.mp4

Env: RUNWAY_API_KEY
"""
import os, json, time, base64, io
from pathlib import Path
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / 'AELP2' / 'prompts' / 'runway_prompts.json'
OUTDIR = ROOT / 'AELP2' / 'outputs' / 'renders' / 'runway'
LOG = ROOT / 'AELP2' / 'reports' / 'runway_tasks.json'

API_BASE = 'https://api.runwayml.com'
API_VERSION = '2024-11-06'

def seed_data_uri() -> str:
    """Create a clean, text-free neutral plate to avoid seeding on-frame text.
    We keep a subtle vignette and neutral grey to bias exposure but write no text.
    """
    im = Image.new('RGB', (720, 1280), (22, 22, 22))
    dr = ImageDraw.Draw(im)
    # soft vignette corners
    for r, alpha in [(40, 12), (80, 8), (120, 5)]:
        dr.rectangle([r, r, 720-r, 1280-r], outline=(0,0,0), width=2)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'

def create_task(session: requests.Session, prompt_text: str, negatives: str) -> str:
    url = f'{API_BASE}/v1/image_to_video'
    # Forbid on-frame text/UI; keep motion subtle and handheld.
    safe_neg = (
        (negatives or '') + 
        ", on-frame text, captions, UI elements, watermark, logo, extra fingers, warped hands"
    ).strip(', ')
    payload = {
        'model': 'gen4_turbo',
        'promptImage': seed_data_uri(),
        'promptText': f"{prompt_text}. Do not show any on-screen text or UI. Avoid: {safe_neg}",
        'ratio': '720:1280',
        'duration': 5,
        'watermark': False
    }
    r = session.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()['id']

def poll_task(session: requests.Session, task_id: str, timeout_s=900):
    url = f'{API_BASE}/v1/tasks/{task_id}'
    start=time.time()
    while True:
        r = session.get(url, timeout=30)
        if r.status_code==429:
            time.sleep(5); continue
        r.raise_for_status()
        js=r.json()
        st=js.get('status') or js.get('state')
        if st in ('SUCCEEDED','COMPLETED'):
            return js
        if st in ('FAILED','CANCELED','ERROR'):
            return js
        if time.time()-start>timeout_s:
            return {'id': task_id, 'status': 'TIMEOUT'}
        time.sleep(5)

def download_asset(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(1<<14):
                if chunk:
                    f.write(chunk)

def main():
    key=os.getenv('RUNWAY_API_KEY')
    if not key:
        print(json.dumps({'error':'RUNWAY_API_KEY missing'}, indent=2)); return
    prompts=json.loads(PROMPTS.read_text())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    headers={'Authorization': f'Bearer {key}', 'X-Runway-Version': API_VERSION}
    sess=requests.Session(); sess.headers.update(headers)
    tasks=[]
    for p in prompts['prompts']:
        tid=create_task(sess, p['text'], p.get('negatives',''))
        tasks.append({'id': tid, 'prompt_id': p['id']})
    LOG.write_text(json.dumps({'submitted': tasks}, indent=2))
    # poll and download
    results=[]
    for t in tasks:
        js=poll_task(sess, t['id'])
        t['result']=js
        st=js.get('status')
        # find asset URL
        url=None
        for k in ('output','assets','result','data'):
            v=js.get(k)
            if isinstance(v, dict):
                url=v.get('video') or v.get('url')
            if isinstance(v, list) and v and isinstance(v[0], dict):
                url=v[0].get('url') or v[0].get('video')
            if url:
                break
        if st in ('SUCCEEDED','COMPLETED') and url:
            dest=OUTDIR/(f"{t['id']}.mp4")
            try:
                download_asset(url, dest)
                t['local']=str(dest)
            except Exception as e:
                t['error']=str(e)
        results.append(t)
    LOG.write_text(json.dumps({'results': results}, indent=2))
    ok=sum(1 for r in results if r.get('local'))
    print(json.dumps({'submitted': len(tasks), 'completed': ok, 'out_dir': str(OUTDIR)}, indent=2))

if __name__=='__main__':
    main()
