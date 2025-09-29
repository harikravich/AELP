#!/usr/bin/env python3
from __future__ import annotations
"""
Build a minimal brand pack from existing ads + site:
- Reads Meta creative_objects/* for domain(s), CTAs, titles
- Reads KB products for CTAs and disclaimers
- Fetches brand site (aura.com) pages + CSS to extract colors and fonts

Outputs:
- AELP2/branding/brand_pack.json

Note: Best effort; can be refined with provided assets later.
"""
import json, re, os
from pathlib import Path
from urllib.parse import urlparse
import requests

ROOT = Path(__file__).resolve().parents[2]
COBJ_DIR = ROOT / 'AELP2' / 'reports' / 'creative_objects'
KB_DIR = ROOT / 'AELP2' / 'knowledge' / 'products'
OUT_DIR = ROOT / 'AELP2' / 'branding'
OUT = OUT_DIR / 'brand_pack.json'

HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
FONT_RE = re.compile(r"font-family\s*:\s*([^;]+);", re.IGNORECASE)

GENERIC_FONTS = {"sans-serif","serif","monospace","system-ui","ui-sans-serif","ui-serif","ui-monospace","ui-rounded"}

def read_creatives():
    domains = []
    titles = []
    ctas = []
    for fp in COBJ_DIR.glob('*.json'):
        try:
            d=json.loads(fp.read_text())
        except Exception:
            continue
        spec = (((d.get('creative') or {}).get('asset_feed_spec')) or {})
        for lu in spec.get('link_urls', []) or []:
            u = lu.get('website_url') or lu.get('display_url') or ''
            if not u: continue
            try:
                host = urlparse(u).netloc or urlparse('https://' + u).netloc
                if host and host not in domains:
                    domains.append(host)
            except Exception:
                pass
        for t in spec.get('titles', []) or []:
            if t.get('text'): titles.append(t['text'])
        for ct in spec.get('call_to_action_types', []) or []:
            ctas.append(ct)
    return domains, titles, ctas

def read_kb():
    ctalist=[]; disclaimers=set()
    for fp in KB_DIR.glob('*.json'):
        try:
            d=json.loads(fp.read_text())
        except Exception:
            continue
        for c in d.get('approved_ctas',[]) or []:
            ctalist.append(c)
        for c in d.get('approved_claims',[]) or []:
            if c.get('mandatory_disclaimer'):
                disclaimers.add(c['mandatory_disclaimer'])
        for n in (d.get('compliance',{}) or {}).get('notes',[]) or []:
            disclaimers.add(n)
    return sorted(set(ctalist)), sorted(disclaimers)

def fetch_site_colors_fonts(hosts: list[str]):
    # Prefer aura.com host if present
    base = None
    for h in hosts:
        if 'aura.com' in h:
            base = 'https://' + h
            break
    if not base and hosts:
        base = 'https://' + hosts[0]
    if not base:
        return [], []
    session = requests.Session()
    pages=[base, base.rstrip('/') + '/identity-theft-protection']
    css_texts=[]
    for url in pages:
        try:
            r=session.get(url, timeout=15, headers={'User-Agent':'Mozilla/5.0'})
            if r.status_code!=200: continue
            # inline colors
            css_texts.append(r.text)
            # linked stylesheets
            for m in re.finditer(r'<link[^>]+rel=["\']stylesheet["\'][^>]+>', r.text, re.IGNORECASE):
                tag=m.group(0)
                hrefm=re.search(r'href=["\']([^"\']+)["\']', tag)
                if not hrefm: continue
                href=hrefm.group(1)
                if href.startswith('//'): href='https:' + href
                elif href.startswith('/'): href=base.rstrip('/') + href
                if not href.startswith('http'): continue
                try:
                    cr=session.get(href, timeout=15, headers={'User-Agent':'Mozilla/5.0'})
                    if cr.status_code==200 and 'text/css' in cr.headers.get('Content-Type',''):
                        css_texts.append(cr.text)
                except Exception:
                    pass
        except Exception:
            continue
    # Extract colors and fonts
    colors=[]; fonts=[]
    for t in css_texts:
        colors.extend(HEX_RE.findall(t))
        for fm in FONT_RE.findall(t):
            fams=[f.strip().strip('"\'') for f in fm.split(',')]
            for f in fams:
                if not f or f.lower() in GENERIC_FONTS: continue
                fonts.append(f)
    # Count and sort colors by frequency, prefer 6-digit hex uppercased
    from collections import Counter
    def norm_hex(h):
        h=h.upper()
        if len(h)==4: # #ABC -> #AABBCC
            h='#' + ''.join(c*2 for c in h[1:])
        return h
    ccount=Counter(norm_hex(h) for h in colors)
    top_colors=[h for h,_ in ccount.most_common(12)]
    fcount=Counter(fonts)
    top_fonts=[f for f,_ in fcount.most_common(6)]
    return top_colors, top_fonts

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hosts, titles, meta_ctas = read_creatives()
    ctalist, disclaimers = read_kb()
    colors, fonts = fetch_site_colors_fonts(hosts)
    pack = {
        'brand_domain_hosts': hosts,
        'palette_hex': colors,
        'fonts': fonts,
        'cta_library': sorted(set(ctalist + meta_ctas)),
        'disclaimers': disclaimers,
        'examples': {
            'recent_titles': titles[:12]
        }
    }
    OUT.write_text(json.dumps(pack, indent=2))
    print(json.dumps({'out': str(OUT), 'hosts': hosts[:3], 'colors': colors[:8], 'fonts': fonts[:4], 'ctas': pack['cta_library'][:6]}, indent=2))

if __name__=='__main__':
    main()

