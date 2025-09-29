#!/usr/bin/env python3
from __future__ import annotations
"""
Generate a pack of VO lines from AELP2/prompts/vo_scripts.json using ElevenLabs.
Writes MP3s to AELP2/outputs/audio/<script_id>/hook.mp3|proof.mp3|cta.mp3

Env: ELEVENLABS_API_KEY
"""
import os, json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'AELP2' / 'prompts' / 'vo_scripts.json'
OUT = ROOT / 'AELP2' / 'outputs' / 'audio'

def pick_voice(key: str) -> str:
    r=requests.get('https://api.elevenlabs.io/v1/voices', headers={'xi-api-key': key}, timeout=30)
    r.raise_for_status()
    voices=r.json().get('voices',[])
    for v in voices:
        name=(v.get('name') or '').lower()
        if any(k in name for k in ('earnest','conversational','warm')):
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
        'voice_settings': { 'stability': 0.5, 'similarity_boost': 0.7, 'style': 0.4, 'use_speaker_boost': True }
    }
    headers={'xi-api-key': key, 'Accept': 'audio/mpeg', 'Content-Type':'application/json'}
    r=requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.content

def main():
    key=os.getenv('ELEVENLABS_API_KEY')
    assert key, 'ELEVENLABS_API_KEY missing'
    data=json.loads(SCRIPTS.read_text())
    voice=pick_voice(key)
    OUT.mkdir(parents=True, exist_ok=True)
    results=[]
    for s in data['scripts']:
        d=OUT / s['id']
        d.mkdir(parents=True, exist_ok=True)
        for role in ('hook','proof','cta'):
            mp3 = d / f'{role}.mp3'
            if not mp3.exists():
                mp3.write_bytes(tts(key, voice, s[role]))
        results.append({'id': s['id'], 'dir': str(d)})
    print(json.dumps({'voice_id': voice, 'packs': results}, indent=2))

if __name__=='__main__':
    main()

