#!/usr/bin/env python3
from __future__ import annotations
import json, re, math
from pathlib import Path
from datetime import datetime
import re

ROOT = Path(__file__).resolve().parents[2]
BASIC = ROOT / 'AELP2' / 'reports' / 'creative_with_dna'
if not BASIC.exists():
    BASIC = ROOT / 'AELP2' / 'reports' / 'creative'
OBJS = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUT = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUT.mkdir(parents=True, exist_ok=True)

def text_stats(s: str) -> dict:
    if not s:
        return {'wc':0,'chars':0,'avg_len':0.0}
    words = re.findall(r"[A-Za-z0-9_']+", s)
    wc = len(words)
    chars = sum(len(w) for w in words)
    avg_len = (chars/wc) if wc else 0.0
    return {'wc': wc, 'chars': chars, 'avg_len': round(avg_len,2)}

def bow16(s: str) -> list:
    # Tiny hashing bag-of-words with 16 bins, normalized
    if not s:
        return [0.0]*16
    toks = re.findall(r"[a-z0-9']+", s.lower())
    if not toks:
        return [0.0]*16
    bins = [0]*16
    for t in toks:
        h = (hash(t) & 0x7fffffff) % 16
        bins[h] += 1
    total = float(sum(bins)) or 1.0
    return [round(b/total,4) for b in bins]

def bow64(s: str) -> list:
    if not s:
        return [0.0]*64
    toks = re.findall(r"[a-z0-9']+", s.lower())
    if not toks:
        return [0.0]*64
    bins = [0]*64
    for t in toks:
        h = (hash(t) & 0x7fffffff) % 64
        bins[h] += 1
    total = float(sum(bins)) or 1.0
    return [round(b/total,4) for b in bins]

def infer_ratio_tag(placement_mix: dict) -> str:
    keys = ' '.join(placement_mix.keys()).lower() if placement_mix else ''
    if 'story' in keys or 'stories' in keys or 'reels' in keys:
        return '9x16'
    if 'video_feeds' in keys or 'instream' in keys:
        return 'video'
    if 'feed' in keys or 'facebook' in keys or 'instagram' in keys:
        return '1x1-4x5'
    return 'unknown'

def parse_obj(obj: dict) -> dict:
    creative = obj.get('creative') or {}
    oss = creative.get('object_story_spec') or {}
    link = oss.get('link_data') or {}
    video = oss.get('video_data') or {}
    body = creative.get('body') or link.get('message') or ''
    title = creative.get('title') or link.get('name') or ''
    desc = link.get('description') or ''
    cta = ((link.get('call_to_action') or {}).get('type')) or ''
    t = ' '.join([str(x) for x in [title, body, desc] if x])
    stats = text_stats(t)
    # Simple sentiment via small lexicon
    POS = set(['fast','easy','secure','save','protect','win','best','safe','quick','smart'])
    NEG = set(['slow','hard','risk','scam','hack','problem','issue','bad','worst','expensive'])
    toks = re.findall(r"[A-Za-z']+", t.lower())
    pos_hits = sum(1 for w in toks if w in POS)
    neg_hits = sum(1 for w in toks if w in NEG)
    sent = 0.0
    total = pos_hits + neg_hits
    if total:
        sent = (pos_hits - neg_hits) / total
    # Rough Flesch-Kincaid grade using syllable heuristic
    def syllables(word: str) -> int:
        word = word.lower()
        vowels = 'aeiouy'
        count = 0; prev=False
        for ch in word:
            isv = ch in vowels
            if isv and not prev: count += 1
            prev = isv
        if word.endswith('e') and count>1: count -= 1
        return max(1, count)
    words = toks
    sentences = max(1, t.count('.') + t.count('!') + t.count('?'))
    syl = sum(syllables(w) for w in words)
    wc = max(1, len(words))
    fk_grade = 0.39*(wc/max(1, sentences)) + 11.8*(syl/max(1, wc)) - 15.59
    fields = {
        'title_wc': stats['wc'], 'avg_word_len': stats['avg_len'],
        'has_cta_button': 1 if cta else 0,
        'sentiment': round(sent,3),
        'fk_grade': round(fk_grade,2),
        'bow16': bow16(t),
        'bow64': bow64(t),
    }
    # created_time
    ad = obj.get('ad') or {}
    ct = ad.get('created_time')
    if ct:
        try:
            fields['created_time'] = ct[:19]
            fields['created_epoch'] = int(datetime.strptime(ct[:19], '%Y-%m-%dT%H:%M:%S').timestamp())
        except Exception:
            pass
    return fields

def main():
    # Determine last test day to compute age
    simf = ROOT / 'AELP2' / 'reports' / 'sim_fidelity_campaigns_temporal_v2.json'
    last_day = None
    if simf.exists():
        try:
            js = json.loads(simf.read_text())
            d = [r['date'] for r in js.get('daily', [])]
            last_day = max(d) if d else None
        except Exception:
            pass
    last_ts = None
    if last_day:
        try:
            last_ts = int(datetime.strptime(last_day, '%Y-%m-%d').timestamp())
        except Exception:
            pass

    for jf in sorted(BASIC.glob('*.json')):
        data = json.loads(jf.read_text())
        items = data.get('items') or []
        out_items = []
        for it in items:
            adid = it.get('creative_id')
            objp = OBJS / f'{adid}.json'
            obj = json.loads(objp.read_text()) if objp.exists() else {}
            fields = parse_obj(obj)
            # age
            age_days = None
            if last_ts and fields.get('created_epoch'):
                age_days = max(0, int((last_ts - fields['created_epoch'])/86400))
            ratio_tag = infer_ratio_tag(it.get('placement_mix') or {})
            out_items.append({**it, 'dna_real': {
                'title_wc': fields.get('title_wc',0),
                'avg_word_len': fields.get('avg_word_len',0.0),
                'has_cta_button': fields.get('has_cta_button',0),
                'ratio_tag': ratio_tag,
                'age_days': age_days
            }})
        out = {'campaign_id': data.get('campaign_id'), 'items': out_items}
        (OUT / jf.name).write_text(json.dumps(out, indent=2))
    print(f'Wrote {len(list(OUT.glob("*.json")))} files to {OUT}')

if __name__ == '__main__':
    main()
