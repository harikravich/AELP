#!/usr/bin/env python3
from __future__ import annotations
"""
Generate Balance-focused creative blueprints from Aura parental-controls concepts
and produce a Top-20 list + forecasts + RL pack (separate from the core security set).
"""
import json, itertools, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
TOP = ROOT / 'AELP2' / 'reports' / 'ad_balance_blueprints_top20.json'
FORE = ROOT / 'AELP2' / 'reports' / 'us_balance_forecasts.json'
RLP = ROOT / 'AELP2' / 'reports' / 'rl_balance_pack.json'


def combos():
    motifs = ['balance_trends','parent_insights','sleep_patterns','social_persona','focus_time']
    subjects = ['parent_kid','teen','no_person']
    palettes = ['trust']
    formats = ['static','video6s']
    ctas = ['LEARN_MORE','SIGN_UP']
    placements = [
        {'publisher_platforms':['facebook','instagram'], 'facebook_positions':['feed'], 'instagram_positions':['stream'], 'name':'feed_only'},
        {'publisher_platforms':['facebook','instagram'], 'facebook_positions':['facebook_reels','story'], 'instagram_positions':['reels','story'], 'name':'reels_stories'},
    ]
    for m,s,p,f,c,pl in itertools.product(motifs, subjects, palettes, formats, ctas, placements):
        yield {'motif':m,'subject':s,'palette':p,'format':f,'cta':c,'placement_name':pl['name'],'placement_spec':pl}


def blueprint_to_cobj(bp: dict, idx: int) -> dict:
    title_map = {
        'balance_trends': 'See Changes in Your Kid’s Online Patterns',
        'parent_insights': 'Get Personalized Parent Insights',
        'sleep_patterns': 'Spot Sleep and Screen Time Shifts',
        'social_persona': 'Understand Their Social Persona',
        'focus_time': 'Help Your Kid Focus—Block Distractions',
    }
    body_map = {
        'balance_trends': 'Track trends in social interactions, activity, and sleep—spot meaningful changes early.',
        'parent_insights': 'Science-backed tips and conversation starters tailored to your child’s routines.',
        'sleep_patterns': 'View day/night activity and bedtime patterns to support healthier sleep.',
        'social_persona': 'See high-level patterns in how your kid engages online—no message reading.',
        'focus_time': 'Set study mode and app limits to reduce doomscrolling and improve focus.',
    }
    body = body_map[bp['motif']]
    videos = [{}] if bp['format'].startswith('video') else []
    rules = [{
        'customization_spec': {
            'publisher_platforms': bp['placement_spec']['publisher_platforms'],
            'facebook_positions': bp['placement_spec'].get('facebook_positions', []),
            'instagram_positions': bp['placement_spec'].get('instagram_positions', []),
        }
    }]
    cid = f"bpbal_{idx:04d}"
    return {
        'ad': {'id': cid, 'name': f"{bp['motif']}_{bp['subject']}_{bp['format']}_{bp['cta']}_{bp['placement_name']}"},
        'creative': {
            'name': f"{bp['motif']} {bp['format']} {bp['cta']}",
            'asset_feed_spec': {
                'titles': [{'text': title_map[bp['motif']]}],
                'bodies': [{'text': body}],
                'link_urls': [{'website_url': 'https://www.aura.com/parental-controls'}],
                'call_to_action_types': [bp['cta']],
                'videos': videos,
                'asset_customization_rules': rules,
            }
        },
        '_blueprint': bp,
        '_source': 'balance_blueprint'
    }


def run():
    COBJ.mkdir(parents=True, exist_ok=True)
    bps = list(itertools.islice(combos(), 0, 120))
    for i, bp in enumerate(bps, 1):
        d = blueprint_to_cobj(bp, i)
        cid = d['ad']['id']
        (COBJ / f'vendor_balance_{cid}.json').write_text(json.dumps(d, indent=2))
    # features+scores
    subprocess.check_call([sys.executable, str(ROOT / 'AELP2' / 'tools' / 'build_features_from_creative_objects.py')])
    subprocess.check_call([sys.executable, str(ROOT / 'AELP2' / 'tools' / 'score_vendor_creatives.py')])
    scores = json.loads((ROOT / 'AELP2' / 'reports' / 'vendor_scores.json').read_text())['items']
    top = [r for r in scores if str(r['creative_id']).startswith('bpbal_')][:20]
    TOP.write_text(json.dumps({'count': len(top), 'items': top}, indent=2))

    # Forecasts: reuse the main forecaster by swapping TOP/FORE paths
    # We'll write a small shim here using the same logic as forecast_us_cac_volume
    base = json.loads((ROOT / 'AELP2' / 'reports' / 'us_meta_baselines.json').read_text())
    def tri(p10,p50,p90):
        import random
        return random.triangular(p10,p90,p50)
    cpm_p10=float(base['cpm_p10']); cpm_p50=float(base['cpm_p50']); cpm_p90=float(base['cpm_p90'])
    ctr_p10=float(base['ctr_p10']); ctr_p50=float(base['ctr_p50']); ctr_p90=float(base['ctr_p90'])
    # paid CVR from baselines percentiles with clamps
    cvr_p10=float(base.get('cvr_p10',0.01)); cvr_p50=float(base.get('cvr_p50',0.02)); cvr_p90=float(base.get('cvr_p90',0.04))
    def clamp01(x): return max(0.0,min(1.0,float(x)))
    ctr_p10,ctr_p50,ctr_p90 = map(clamp01,(ctr_p10,ctr_p50,ctr_p90))
    def clamp_cvr(x): return max(0.002, min(0.05, float(x)))
    cvr_p10,cvr_p50,cvr_p90 = map(clamp_cvr,(cvr_p10,cvr_p50,cvr_p90))
    budgets=[30000.0,50000.0]
    # Load top and reconstruct multipliers from p_win
    items = []
    for r in top:
        p=float(r['p_win']);
        def mctr(pp): return 0.5+1.0*pp
        def mcvr(pp): return 0.8+0.4*pp
        entry={'creative_id':r['creative_id'],'p_win':p,'lcb':float(r.get('lcb',0.0)),'budgets':{}}
        for B in budgets:
            sims=[]
            for _ in range(2000):
                cpm=tri(cpm_p10,cpm_p50,cpm_p90)
                ctr=tri(ctr_p10,ctr_p50,ctr_p90)*mctr(p)
                cvr=tri(cvr_p10,cvr_p50,cvr_p90)*mcvr(p)
                imps=(B/cpm)*1000.0; clicks=imps*ctr; su=clicks*cvr; cac=B/max(su,1e-6)
                sims.append((imps,clicks,su,cac))
            sims.sort(key=lambda t:t[3])
            def pct(a,q):
                i=int(q*(len(a)-1)); return float(a[i])
            imps_list,clicks_list,su_list,cac_list=zip(*sims)
            entry['budgets'][str(int(B))]={
                'impressions':{'p10':pct(imps_list,0.1),'p50':pct(imps_list,0.5),'p90':pct(imps_list,0.9)},
                'clicks':{'p10':pct(clicks_list,0.1),'p50':pct(clicks_list,0.5),'p90':pct(clicks_list,0.9)},
                'signups':{'p10':pct(su_list,0.1),'p50':pct(su_list,0.5),'p90':pct(su_list,0.9)},
                'cac':{'p10':pct(cac_list,0.1),'p50':pct(cac_list,0.5),'p90':pct(cac_list,0.9)},
            }
        items.append(entry)
    FORE.write_text(json.dumps({'items':items,'budgets':budgets},indent=2))

    # Simple RL pack for balance
    RLP.write_text(json.dumps({'items':[{'creative_id':r['creative_id']} for r in top]}, indent=2))
    print(json.dumps({'top_out': str(TOP), 'fore_out': str(FORE), 'rl_out': str(RLP), 'count': len(items)}, indent=2))


if __name__ == '__main__':
    run()
