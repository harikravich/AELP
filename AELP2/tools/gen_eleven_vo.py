#!/usr/bin/env python3
from __future__ import annotations
"""
Generate VO lines with ElevenLabs v3 and save MP3s.

Env: ELEVENLABS_API_KEY
Usage: python3 AELP2/tools/gen_eleven_vo.py
Writes to AELP2/outputs/audio/
"""
import os, json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'outputs' / 'audio'

LINES = [
  { 'id': 'hook_identity', 'text': "Did you get another breach alert? Take a breath—Aura can help." },
  { 'id': 'proof_identity', 'text': "See exposures fast and lock things down before damage spreads." },
  { 'id': 'cta_identity', 'text': "Start your free trial and get protected today." }
]

def pick_voice(key: str) -> str:
    r=requests.get('https://api.elevenlabs.io/v1/voices', headers={'xi-api-key': key}, timeout=30)
    r.raise_for_status()
    voices=r.json().get('voices',[])
    # Prefer a non-default; fallback to Rachel
    for v in voices:
        name=(v.get('name') or '').lower()
        if 'earnest' in name or 'parent' in name:
            return v['voice_id']
    for v in voices:
        if v.get('name')=='Rachel':
            return v['voice_id']
    return voices[0]['voice_id'] if voices else '21m00Tcm4TlvDq8ikWAM'

def tts(key: str, voice_id: str, text: str) -> bytes:
    url=f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
    payload={
        'model_id': 'eleven_multilingual_v2',
        'text': text,
        'voice_settings': { 'stability': 0.55, 'similarity_boost': 0.7, 'style': 0.35, 'use_speaker_boost': True }
    }
    headers={'xi-api-key': key, 'Accept': 'audio/mpeg', 'Content-Type':'application/json'}
    r=requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.content

def main():
    key=os.getenv('ELEVENLABS_API_KEY'); assert key, 'ELEVENLABS_API_KEY missing'
    OUT.mkdir(parents=True, exist_ok=True)
    vid=pick_voice(key)
    out_files=[]
    for ln in LINES:
        data=tts(key, vid, ln['text'])
        fp=OUT/f"{ln['id']}.mp3"
        fp.write_bytes(data)
        out_files.append(str(fp))
    print(json.dumps({'voice_id': vid, 'files': out_files}, indent=2))

if __name__=='__main__':
    main()
