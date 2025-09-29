#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / 'AELP2' / 'knowledge' / 'schema' / 'product_kb.schema.json'
DIR = ROOT / 'AELP2' / 'knowledge' / 'products'

def load_schema():
    return json.loads(SCHEMA.read_text())

def validate(obj, schema):
    # Minimal required-field validator to avoid adding external deps
    def require(path, node, fields):
        for f in fields:
            if f not in node:
                raise ValueError(f"Missing {'.'.join(path+[f])}")
    require(['root'], obj, ['product_id','name','meta','personas','value_props','approved_claims','approved_ctas','creative_guidelines','compliance'])
    require(['meta'], obj['meta'], ['owner','last_reviewed','expiry_days','sources','trust_tier'])
    for c in obj.get('approved_claims', []):
        require(['approved_claims'], c, ['id','text','qualifier','evidence_link','mandatory_disclaimer','last_reviewed','expiry_days'])

def main():
    schema=load_schema()
    ok=0; fail=[]
    for f in sorted(DIR.glob('*.json')):
        try:
            d=json.loads(f.read_text())
            validate(d, schema)
            ok+=1
        except Exception as e:
            fail.append({'file': f.name, 'error': str(e)})
    out={'validated': ok, 'errors': fail}
    (ROOT/'AELP2'/'reports'/'kb_validation.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main()

