#!/usr/bin/env python3
from __future__ import annotations
"""
Generate 9:16 hook clips on Vertex AI (Veo 3 Fast) and write to GCS.

Reads prompts from AELP2/prompts/veo_prompts.json.
For each prompt, calls predictLongRunning and polls the operation until done.

Env: GOOGLE_CLOUD_PROJECT, CREATIVE_GCS_BUCKET
"""
import os, json, time, subprocess
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / 'AELP2' / 'prompts' / 'veo_prompts.json'
OUTLOG = ROOT / 'AELP2' / 'reports' / 'veo_tasks.json'

LOCATION = 'us-central1'
MODEL = 'veo-3.0-fast-generate-001'

def access_token() -> str:
    # Use gcloud to obtain an access token (ADC)
    return subprocess.check_output(['gcloud','auth','print-access-token'], text=True).strip()

def start_job(project: str, bucket: str, prompt: str, negative: str, run_id: str):
    url = f'https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{project}/locations/{LOCATION}/publishers/google/models/{MODEL}:predictLongRunning'
    storage = f'gs://{bucket}/creatives/veo/{run_id}/'
    payload = {
        'instances': [{ 'prompt': prompt }],
        'parameters': {
            'aspectRatio': '9:16',
            'durationSeconds': 8,
            'resolution': '1080p',
            'personGen': 'allow_adult',
            'negativePrompt': negative,
            'storageUri': storage,
            'generateAudio': False
        }
    }
    headers={'Authorization': f'Bearer {access_token()}', 'Content-Type':'application/json'}
    r=requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()['name'], storage

def poll_op(name: str, timeout_s=1200):
    # Use fetchPredictOperation per Vertex docs
    model_path = '/'.join(name.split('/')[:8])  # projects/.../publishers/google/models/MODEL_ID
    url=f'https://{LOCATION}-aiplatform.googleapis.com/v1/{model_path}:fetchPredictOperation'
    headers={'Authorization': f'Bearer {access_token()}', 'Content-Type':'application/json'}
    start=time.time()
    while True:
        r=requests.post(url, headers=headers, json={'operationName': name}, timeout=30)
        r.raise_for_status()
        js=r.json()
        if js.get('done'):
            return js
        if time.time()-start>timeout_s:
            return {'name': name, 'done': False, 'timeout': True}
        time.sleep(10)

def main():
    project=os.getenv('GOOGLE_CLOUD_PROJECT'); bucket=os.getenv('CREATIVE_GCS_BUCKET')
    if not (project and bucket):
        print(json.dumps({'error':'missing GOOGLE_CLOUD_PROJECT or CREATIVE_GCS_BUCKET'}, indent=2)); return
    prompts=json.loads(PROMPTS.read_text())
    tasks=[]
    for i,p in enumerate(prompts['prompts']):
        run_id=f"run_{int(time.time())}_{i}"
        op, storage=start_job(project, bucket, p['text'], p.get('negativePrompt',''), run_id)
        tasks.append({'prompt_id': p['id'], 'op': op, 'storage': storage})
    OUTLOG.write_text(json.dumps({'submitted': tasks}, indent=2))
    # poll
    results=[]
    for t in tasks:
        js=poll_op(t['op'])
        t['result']=js
        results.append(t)
    OUTLOG.write_text(json.dumps({'results': results}, indent=2))
    print(json.dumps({'submitted': len(tasks), 'completed': sum(1 for r in results if (r.get('result') or {}).get('done')), 'gcs_prefix': tasks[0]['storage'] if tasks else ''}, indent=2))

if __name__=='__main__':
    main()
