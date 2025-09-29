#!/usr/bin/env python3
from __future__ import annotations
"""
Join per-final features into one JSONL for analysis.
Inputs: motion_features.jsonl, aesthetic_features.jsonl, legibility_features.jsonl, object_counts.jsonl, audio_lufs.jsonl
Output: AELP2/reports/creative_enriched/finals_features.jsonl
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'AELP2'/'reports'
OUTD=RD/'creative_enriched'

def load_jl(path: Path, key='file'):
    idx={}
    if not path.exists():
        return idx
    with path.open() as f:
        for line in f:
            d=json.loads(line)
            idx[d[key]]=d
    return idx

def main():
    motion=load_jl(RD/'motion_features.jsonl')
    aest=load_jl(RD/'aesthetic_features.jsonl')
    leg=load_jl(RD/'legibility_features.jsonl')
    obj=load_jl(RD/'object_counts.jsonl')
    lufs=load_jl(RD/'audio_lufs.jsonl')
    OUTD.mkdir(parents=True, exist_ok=True)
    out=OUTD/'finals_features.jsonl'
    keys=set(motion.keys())|set(aest.keys())|set(leg.keys())|set(obj.keys())|set(lufs.keys())
    with out.open('w') as f:
        for k in sorted(keys):
            row={'file': k}
            row.update({x:y for x,y in motion.get(k,{}).items() if x!='file'})
            row.update({x:y for x,y in aest.get(k,{}).items() if x!='file'})
            row.update({x:y for x,y in leg.get(k,{}).items() if x!='file'})
            row.update({x:y for x,y in obj.get(k,{}).items() if x!='file'})
            row.update({x:y for x,y in lufs.get(k,{}).items() if x!='file'})
            f.write(json.dumps(row)+'\n')
    print(json.dumps({'out': str(out), 'rows': len(keys)}, indent=2))

if __name__=='__main__':
    main()

