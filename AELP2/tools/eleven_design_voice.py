#!/usr/bin/env python3
from __future__ import annotations
"""
Design and create an ElevenLabs voice from a natural-language prompt, then store it in our library.

Inputs via CLI args or env:
  --prompt "An earnest voice ..."
  --name   "Earnest Parent (UK)"

Env: ELEVENLABS_API_KEY
Writes AELP2/branding/voice_library.json (appends entry with name, prompt, voice_id)
"""
import os, json, sys
from pathlib import Path
import requests
import argparse

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / 'AELP2' / 'branding' / 'voice_library.json'

def design_voice(key: str, prompt: str):
    url='https://api.elevenlabs.io/v1/text-to-voice/design'
    payload={
        'voice_description': prompt,
        # Optional knobs: gender, accent; keep prompt-driven for flexibility
    }
    r=requests.post(url, headers={'xi-api-key': key, 'Content-Type':'application/json'}, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()['generated_voice_id']

def create_voice(key: str, generated_voice_id: str, name: str):
    url='https://api.elevenlabs.io/v1/text-to-voice/create'
    payload={
        'voice_name': name,
        'generated_voice_id': generated_voice_id
    }
    r=requests.post(url, headers={'xi-api-key': key, 'Content-Type':'application/json'}, json=payload, timeout=60)
    r.raise_for_status()
    js=r.json()
    return js['voice_id']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--name', required=True)
    args=ap.parse_args()
    key=os.getenv('ELEVENLABS_API_KEY')
    if not key:
        print(json.dumps({'error':'ELEVENLABS_API_KEY missing'}, indent=2)); sys.exit(1)
    gen_id=design_voice(key, args.prompt)
    voice_id=create_voice(key, gen_id, args.name)
    # persist
    LIB.parent.mkdir(parents=True, exist_ok=True)
    lib=json.loads(LIB.read_text()) if LIB.exists() else {'voices': []}
    lib['voices'].append({'name': args.name, 'prompt': args.prompt, 'voice_id': voice_id})
    LIB.write_text(json.dumps(lib, indent=2))
    print(json.dumps({'voice_id': voice_id, 'name': args.name}, indent=2))

if __name__=='__main__':
    main()

