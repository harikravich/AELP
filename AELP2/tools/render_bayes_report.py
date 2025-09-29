#!/usr/bin/env python3
import os, json, math, csv, datetime
from pathlib import Path

from typing import Dict, List, Optional, Tuple

REPORT_DIR = Path('AELP2/reports')
JSON_PATH = Path('/tmp/meta_campaign_bayes_out.json')
CSV_PATH = REPORT_DIR / 'meta_bayes_summary.csv'
PDF_PATH = REPORT_DIR / 'meta_bayes_onepager.pdf'
CREATIVE_DIR = REPORT_DIR / 'creative'


def load_json(path: Path) -> Dict:
    s = path.read_text()
    try:
        return json.loads(s)
    except Exception:
        i = s.rfind(']}')
        if i != -1:
            s2 = s[: i + 2] + '}\n'
            return json.loads(s2)
        raise


def find_spend_for_threshold(grid: List[float], curve: List[float], thr: float) -> Optional[float]:
    for s, c in zip(grid, curve):
        if c <= thr:
            return float(s)
    return None


def derive_actions(cur: Optional[float], grid: List[float], curves: Dict[str, List[float]], marg_cac: Dict[str, float]) -> Dict[str, Optional[str]]:
    def sset(th):
        return (
            find_spend_for_threshold(grid, curves['p10'], th),
            find_spend_for_threshold(grid, curves['p50'], th),
            find_spend_for_threshold(grid, curves['p90'], th),
        )

    s150 = sset(150.0)
    s120 = sset(120.0)
    s100 = sset(100.0)

    rec = None
    target = None
    if cur is None or not curves:
        rec = 'Insufficient data; build volume to ≥50 purchases/week and re-fit'
    else:
        s150_p50 = s150[1]
        if s150_p50 is None:
            # no crossing within 0.5–2x range
            if marg_cac and marg_cac.get('p50', 9999) > 150:
                rec = 'Reduce and rebuild: optimize to Purchase, DCO ON, broaden; retest'
            else:
                rec = 'Hold and expand grid after fundamentals; re-fit in 7 days'
        else:
            if cur < s150_p50:
                rec = 'Scale stepwise (+10–20%/day) toward s@$150 while CAC ≤ target'
                target = f"${s150[0]:.0f}–${s150[2]:.0f} (p10–p90)"
            elif cur > s150_p50:
                rec = 'Reduce toward s@$150 to improve CAC; refresh creatives; retest'
                target = f"${s150[1]:.0f} (p50); guard ${s150[2]:.0f} (p90)"
            else:
                rec = 'At target; maintain and probe s@$120 with creative/LP gains'
                if s120[1]:
                    target = f"Probe s@$120 p50 ${s120[1]:.0f}"

    return {
        's150_p10': None if s150[0] is None else f"{s150[0]:.2f}",
        's150_p50': None if s150[1] is None else f"{s150[1]:.2f}",
        's150_p90': None if s150[2] is None else f"{s150[2]:.2f}",
        's120_p50': None if s120[1] is None else f"{s120[1]:.2f}",
        's100_p50': None if s100[1] is None else f"{s100[1]:.2f}",
        'rec': rec,
        'target': target,
    }


def write_csv(data: Dict, out_path: Path) -> List[Dict]:
    rows_out = []
    for c in data.get('campaigns', []):
        name = c.get('name'); cid = c.get('id')
        m = c.get('metrics_14d') or {}
        h = c.get('headroom') or {}
        grid = (h.get('grid') or [])
        curves = h.get('cac_curve') or {}
        marg = h.get('marginal_cac') or {}
        cur = h.get('current_spend_med7')

        actions = {'s150_p10': None, 's150_p50': None, 's150_p90': None, 's120_p50': None, 's100_p50': None, 'rec': None, 'target': None}
        if grid and curves:
            actions = derive_actions(cur, grid, curves, marg)

        purch = m.get('purch_14d') or 0
        spend14 = m.get('spend_14d') or 0.0
        cac14 = (spend14 / purch) if purch else None

        row = {
            'campaign_id': cid,
            'campaign_name': name,
            'current_spend_med7': f"{cur:.2f}" if cur else None,
            'purchases_14d': purch,
            'spend_14d': f"{spend14:.2f}",
            'cac_14d': f"{cac14:.2f}" if cac14 else None,
            'marginal_cac_p10': None if not marg else f"{marg.get('p10', None):.2f}",
            'marginal_cac_p50': None if not marg else f"{marg.get('p50', None):.2f}",
            'marginal_cac_p90': None if not marg else f"{marg.get('p90', None):.2f}",
            's150_p10': actions['s150_p10'],
            's150_p50': actions['s150_p50'],
            's150_p90': actions['s150_p90'],
            's120_p50': actions['s120_p50'],
            's100_p50': actions['s100_p50'],
            'recommendation': actions['rec'],
            'target_spend_range': actions['target'],
        }
        rows_out.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    return rows_out


def render_pdf(data: Dict, csv_rows: List[Dict], out_path: Path) -> None:
    # PDF with embedded images using reportlab
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit, ImageReader

    # Try to import matplotlib here to generate images
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
    })

    import requests

    # Build images dir
    IMG_DIR = REPORT_DIR / 'images'
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    def render_chart(row: Dict) -> Optional[Path]:
        name=row['campaign_name']; cid=row['campaign_id']
        # Lookup raw headroom for curves
        h=None
        for c in data.get('campaigns', []):
            if c.get('id')==cid:
                h=c.get('headroom')
                break
        if not h:
            return None
        grid=h.get('grid') or []
        curves=h.get('cac_curve') or {}
        cur = h.get('current_spend_med7')
        if not grid or not curves:
            return None

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
        g=grid
        p10=curves.get('p10') or []
        p50=curves.get('p50') or []
        p90=curves.get('p90') or []
        ax.plot(g, p50, color='#1f77b4', label='CAC median (p50)', linewidth=2)
        ax.plot(g, p10, color='#1f77b4', linestyle='--', alpha=0.6, label='p10/p90 band')
        ax.plot(g, p90, color='#1f77b4', linestyle='--', alpha=0.6)
        ax.fill_between(g, p10, p90, color='#1f77b4', alpha=0.08)

        # Targets
        for thr, col in [(150,'#2ca02c'), (120,'#ff7f0e'), (100,'#d62728')]:
            ax.axhline(thr, color=col, linewidth=1, linestyle=':')
        # Current spend
        if cur:
            ax.axvline(cur, color='#7f7f7f', linestyle='--', linewidth=1)
            ax.annotate('Current spend', xy=(cur, p50[min(range(len(g)), key=lambda i: abs(g[i]-cur))]), xytext=(cur, max(p50)*0.9),
                        arrowprops=dict(arrowstyle='-|>', color='#7f7f7f'), fontsize=10, color='#333')

        # s@$150 median annotation if available
        s150 = row.get('s150_p50')
        try:
            s150v = float(s150) if s150 else None
        except Exception:
            s150v = None
        if s150v:
            ax.axvline(s150v, color='#2ca02c', linestyle='--', linewidth=1)
            ax.annotate('s@$150 (median)', xy=(s150v, 150), xytext=(s150v, 150+max(p50)*0.05),
                        arrowprops=dict(arrowstyle='-|>', color='#2ca02c'), fontsize=10, color='#225522', ha='center')

        ax.set_title(f"CAC vs Daily Spend — {name}")
        ax.set_xlabel('Daily Spend ($)')
        ax.set_ylabel('CAC ($ per purchase)')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)

        out_file = IMG_DIR / f"{cid}.png"
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)
        return out_file

    # Download recent ad images per campaign (best-effort)
    def fetch_ad_images_for_campaign(cid: str, token: Optional[str], limit: int=6) -> List[Path]:
        out_paths: List[Path] = []
        if not token:
            return out_paths
        acct = os.getenv('META_ACCOUNT_ID')
        API = 'https://graph.facebook.com/v21.0'
        try:
            # Pull latest ads (limit)
            r = requests.get(f"{API}/{cid}/ads", params={
                'fields': 'id,name,creative{effective_object_story_id,thumbnail_url,object_story_spec,asset_feed_spec}',
                'limit': limit,
                'access_token': token,
            }, timeout=60).json()
        except Exception:
            return out_paths
        ads = r.get('data', []) if isinstance(r, dict) else []
        if not ads:
            return out_paths
        ad_dir = REPORT_DIR / 'ads' / cid
        ad_dir.mkdir(parents=True, exist_ok=True)

        def download(url: str, idx: int) -> Optional[Path]:
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code==200 and resp.content:
                    p = ad_dir / f"ad_{idx:02d}.jpg"
                    p.write_bytes(resp.content)
                    return p
            except Exception:
                return None
            return None

        for i, ad in enumerate(ads, start=1):
            cr = (ad.get('creative') or {})
            # 1) Try effective_object_story_id → /{id}?fields=full_picture
            sid = cr.get('effective_object_story_id')
            url = None
            if sid:
                try:
                    post = requests.get(f"{API}/{sid}", params={'fields':'full_picture,attachments','access_token':token}, timeout=60).json()
                    url = post.get('full_picture')
                    if not url:
                        atts = (post.get('attachments') or {}).get('data') or []
                        if atts and (atts[0].get('media') or {}).get('image'):
                            url = atts[0]['media']['image'].get('src')
                except Exception:
                    pass
            # 2) Fallbacks
            if not url:
                url = cr.get('thumbnail_url')
            if not url:
                oss = cr.get('object_story_spec') or {}
                link = (oss.get('link_data') or {}).get('picture')
                url = url or link
            # 3) Asset feed spec (first image)
            if not url:
                afs = cr.get('asset_feed_spec') or {}
                imgs = (afs.get('images') or [])
                if imgs:
                    url = imgs[0].get('url')

            if url:
                p = download(url, i)
                if p:
                    out_paths.append(p)
        return out_paths

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=LETTER)
    W, H = LETTER
    margin = 0.75 * inch

    def text_block(title: str, lines: List[str]):
        nonlocal c
        c.setFont('Helvetica-Bold', 12)
        c.drawString(margin, c._curr_y, title)
        c._curr_y -= 14
        c.setFont('Helvetica', 10)
        for ln in lines:
            wrapped = simpleSplit(ln, 'Helvetica', 10, W - 2 * margin)
            for wln in wrapped:
                if c._curr_y < margin:
                    c.showPage(); c._curr_y = H - margin
                    c.setFont('Helvetica', 10)
                c.drawString(margin, c._curr_y, wln)
                c._curr_y -= 12
        c._curr_y -= 6

    # Cover + explainer for marketers
    c._curr_y = H - margin
    c.setFont('Helvetica-Bold', 16)
    c.drawString(margin, c._curr_y, 'Meta Bayesian Headroom — One Pager')
    c._curr_y -= 16
    c.setFont('Helvetica', 10)
    c.drawString(margin, c._curr_y, f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    c._curr_y -= 18

    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, c._curr_y, 'How to Use This (Marketer-friendly)')
    c._curr_y -= 14
    c.setFont('Helvetica', 10)
    expl = [
        '- CAC = cost to get one purchase. We estimate how CAC changes as you raise/lower daily spend.',
        '- “Marginal CAC” is the cost of the next dollar at today’s spend. If it is below your target, you can add budget and stay efficient.',
        '- p10/p50/p90 are confidence bands (10th/median/90th percentile). Use p50 to plan, p90 as a safety guardrail.',
        '- s@$150 is the daily spend where CAC hits $150 on the median curve. If current spend is below this, you can stair-step toward it.',
        '- Stair-step rule: increase spend +10–20% per day while CAC p50 ≤ target and p90 ≤ target + $15. Otherwise, hold or reduce.',
        '- Improve CAC first (signal, creative, landing/checkout), then increase volume. This keeps cost trending down as you scale.',
    ]
    for ln in expl:
        for w in simpleSplit(ln, 'Helvetica', 10, W - 2 * margin):
            if c._curr_y < margin:
                c.showPage(); c._curr_y = H - margin; c.setFont('Helvetica', 10)
            c.drawString(margin, c._curr_y, w)
            c._curr_y -= 12
    c.showPage()

    # For each campaign
    for row in csv_rows:
        title = f"{row['campaign_name']} (ID {row['campaign_id']})"
        stats = [
            f"Current spend (med7): ${row['current_spend_med7'] or '—'} | 14d Purchases: {row['purchases_14d']} | 14d CAC: ${row['cac_14d'] or '—'}",
            f"Marginal CAC p10/p50/p90: ${row['marginal_cac_p10'] or '—'}/{row['marginal_cac_p50'] or '—'}/{row['marginal_cac_p90'] or '—'}",
            f"s@$150 p10/p50/p90: ${row['s150_p10'] or '—'}/{row['s150_p50'] or '—'}/{row['s150_p90'] or '—'} | s@$120 p50: ${row['s120_p50'] or '—'} | s@$100 p50: ${row['s100_p50'] or '—'}",
            f"Recommendation: {row['recommendation'] or '—'}",
            f"Target daily spend: {row['target_spend_range'] or '—'}",
        ]
        text_block(title, stats)

        # Embed chart if available
        img_path = render_chart(row)
        if img_path and img_path.exists():
            try:
                img = ImageReader(str(img_path))
                iw, ih = img.getSize()
                maxw = W - 2 * margin
                scale = maxw / iw
                w = maxw
                h = ih * scale
                if c._curr_y - h < margin:
                    c.showPage(); c._curr_y = H - margin
                c.drawImage(img, margin, c._curr_y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
                c._curr_y -= (h + 12)
            except Exception:
                pass

        # Embed real ad images (best-effort from API)
        token=os.getenv('META_ACCESS_TOKEN')
        ad_imgs = fetch_ad_images_for_campaign(row['campaign_id'], token, limit=6)
        if ad_imgs:
            c.setFont('Helvetica-Bold', 12)
            if c._curr_y < margin + 20:
                c.showPage(); c._curr_y = H - margin
            c.drawString(margin, c._curr_y, 'Recent Ads (visuals)')
            c._curr_y -= 14
            maxw = (W - 3*margin)/2
            col = 0
            for ap in ad_imgs:
                try:
                    img = ImageReader(str(ap))
                    iw, ih = img.getSize()
                    scale = maxw / iw
                    w = maxw
                    h = ih * scale
                    if c._curr_y - h < margin:
                        c.showPage(); c._curr_y = H - margin
                        c.setFont('Helvetica-Bold', 12)
                        c.drawString(margin, c._curr_y, 'Recent Ads (cont.)')
                        c._curr_y -= 14
                    x = margin if col==0 else margin*2 + maxw
                    c.drawImage(img, x, c._curr_y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
                    if col==1:
                        c._curr_y -= (h + 12)
                    col = 1 - col
                except Exception:
                    continue

    # Marketer summary page — volumes & headroom in plain terms
    c.showPage(); c._curr_y = H - margin
    c.setFont('Helvetica-Bold', 14)
    c.drawString(margin, c._curr_y, 'What This Means — Volume, Headroom, CAC (Plain English)')
    c._curr_y -= 16
    c.setFont('Helvetica', 10)

    # Compute volumes
    for row in csv_rows:
        name=row['campaign_name']; cid=row['campaign_id']
        cur_spend = float(row['current_spend_med7']) if row['current_spend_med7'] else None
        purch14 = float(row['purchases_14d']) if row['purchases_14d'] else 0.0
        daily_purch_now = purch14/14.0 if purch14 else None
        s150 = row.get('s150_p50'); s150_val = float(s150) if s150 else None
        daily_purch_at150 = (s150_val/150.0) if s150_val else None
        if c._curr_y < margin+60:
            c.showPage(); c._curr_y = H - margin; c.setFont('Helvetica', 10)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(margin, c._curr_y, name)
        c._curr_y -= 12
        c.setFont('Helvetica', 10)
        msg = []
        if cur_spend is not None:
            msg.append(f"Today you spend about ${cur_spend:,.0f}/day.")
        if daily_purch_now is not None and daily_purch_now>0:
            msg.append(f"That buys ~{daily_purch_now:,.1f} purchases/day at recent performance.")
        if s150_val and daily_purch_at150:
            if cur_spend and s150_val>cur_spend:
                add_p = daily_purch_at150 - (cur_spend/150.0)
                msg.append(f"Model says you can raise spend to ~${s150_val:,.0f}/day and still hit ~$150 CAC, yielding ~{daily_purch_at150:,.1f} purchases/day (≈+{add_p:,.1f}/day).")
            else:
                msg.append(f"Model suggests efficiency at ~$150 CAC occurs around ${s150_val:,.0f}/day. Spending above this likely pushes CAC up.")
        if not s150_val:
            msg.append("No safe $150 CAC level in range — fix creative/targeting/LP, then re-measure.")
        for ln in msg:
            for wln in simpleSplit(ln, 'Helvetica', 10, W-2*margin):
                c.drawString(margin, c._curr_y, wln); c._curr_y -= 12
        c._curr_y -= 6

    # Creative roadmap visuals
    c.showPage(); c._curr_y = H - margin
    c.setFont('Helvetica-Bold', 14)
    c.drawString(margin, c._curr_y, 'Creative Roadmap — Visual Examples (9:16 Reels)')
    c._curr_y -= 16

    # Generate mockups if missing
    CREATIVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default()
    except Exception:
        Image = None
        font = None

    concepts = [
        ('Proof-to-solution', 'Cold open: breach proof → instant setup demo → CTA'),
        ('Install-in-minutes', 'Stopwatch + 3-step install + live toggles'),
        ('Parent story UGC', 'Selfie testimonial + before/after alerts'),
        ('Comparison', 'Manual removal vs Aura; time/risk/price'),
        ('Data breach explainer', 'Leak visuals → shield → plan value'),
        ('Device safety demo', 'Kid phone safety; filters; location'),
        ('Social proof carousel', '3 reviews + stars + guarantee'),
        ('Offer clarity', 'Trial terms; what’s included; cancel anytime'),
        ('Myth-busting', '“One-time removal?” → continuous monitoring'),
        ('Urgency-lite', 'New breaches daily; set and forget'),
    ]

    created_imgs = []
    if Image:
        W9, H9 = 1080, 1920
        for idx, (title, subtitle) in enumerate(concepts, start=1):
            path = CREATIVE_DIR / f"concept_{idx:02d}.png"
            if not path.exists():
                img = Image.new('RGB', (W9, H9), (245, 246, 250))
                d = ImageDraw.Draw(img)
                # phone bezel
                d.rounded_rectangle([60, 60, W9-60, H9-60], radius=48, outline=(80,80,90), width=6)
                # hero text
                d.rectangle([120, 140, W9-120, 360], fill=(255,255,255), outline=(200,200,210))
                d.text((140, 170), f"{title}", fill=(20,20,20), font=font)
                d.text((140, 210), subtitle, fill=(60,60,60), font=font)
                # UI mock sections
                y = 420
                for _ in range(4):
                    d.rounded_rectangle([140, y, W9-140, y+140], radius=16, fill=(255,255,255), outline=(222,226,233))
                    # left thumbnail
                    d.rectangle([160, y+20, 320, y+120], fill=(230, 242, 255), outline=(160,190,220))
                    # copy lines
                    d.rectangle([350, y+30, W9-180, y+60], fill=(235,235,240))
                    d.rectangle([350, y+70, W9-240, y+95], fill=(235,235,240))
                    y += 160
                # CTA
                d.rounded_rectangle([320, H9-260, W9-320, H9-200], radius=24, fill=(34,139,230))
                d.text((W9//2-120, H9-248), 'Get Protected in Minutes', fill=(255,255,255), font=font)
                img.save(path)
            created_imgs.append(path)

    # Place creative images (2 per page)
    col = 0
    if created_imgs:
        for i, p in enumerate(created_imgs):
            img = ImageReader(str(p))
            iw, ih = img.getSize()
            maxw = (W - 3*margin)/2
            scale = maxw / iw
            w = maxw
            h = ih * scale
            if c._curr_y - h < margin:
                c.showPage(); c._curr_y = H - margin
                c.setFont('Helvetica-Bold', 14); c.drawString(margin, c._curr_y, 'Creative Roadmap — Visual Examples (cont.)')
                c._curr_y -= 16
            x = margin if col==0 else margin*2 + maxw
            c.drawImage(img, x, c._curr_y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
            if col==1:
                c._curr_y -= (h + 16)
            col = 1 - col

    # Landing & Checkout wireframes
    c.showPage(); c._curr_y = H - margin
    c.setFont('Helvetica-Bold', 14)
    c.drawString(margin, c._curr_y, 'Landing & Checkout Wireframes (with fixes)')
    c._curr_y -= 16

    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default()
        # LP wireframe
        lp = CREATIVE_DIR / 'lp_wireframe.png'
        if not lp.exists():
            Wp, Hp = 1440, 1024
            im = Image.new('RGB', (Wp, Hp), (250, 250, 252))
            d = ImageDraw.Draw(im)
            # header
            d.rectangle([0,0,Wp,80], fill=(35, 40, 48))
            d.text((20, 28), 'Header: Logo  |  Nav  |  Trust badges', fill=(255,255,255), font=font)
            # hero
            d.rectangle([60,120, 700, 680], fill=(230,240,255), outline=(170,190,220))
            d.rectangle([740,120, 1380, 320], fill=(255,255,255))
            d.text((760, 140), 'Benefit stack (3 bullets):', fill=(20,20,20), font=font)
            d.rectangle([760, 180, 1340, 210], fill=(235,235,240))
            d.rectangle([760, 220, 1280, 250], fill=(235,235,240))
            d.rectangle([760, 260, 1240, 290], fill=(235,235,240))
            # price/CTA
            d.rounded_rectangle([740, 360, 1380, 520], radius=12, fill=(255,255,255), outline=(220,220,225))
            d.text((760, 380), 'Clear price & term | Guarantee | Badges', fill=(20,20,20), font=font)
            d.rounded_rectangle([980, 440, 1360, 500], radius=18, fill=(34,139,230))
            d.text((1000, 456), 'Get Protected in Minutes', fill=(255,255,255), font=font)
            # trust row
            y=560
            for x in [740, 900, 1060, 1220]:
                d.rounded_rectangle([x, y, x+120, y+80], radius=12, fill=(255,255,255), outline=(225,225,230))
            # notes
            notes=[
                'Speed: aim LCP <2.5s (defer JS, preconnect, compress images).',
                'Above fold: price/term visible, social proof near CTA.',
                'One primary CTA; remove competing links above fold.'
            ]
            yy=660
            for n in notes:
                d.text((60, yy), f"• {n}", fill=(30,30,30), font=font); yy+=24
            im.save(lp)

        # Checkout wireframe
        co = CREATIVE_DIR / 'checkout_wireframe.png'
        if not co.exists():
            Wp, Hp = 1440, 1024
            im = Image.new('RGB', (Wp, Hp), (252, 250, 250))
            d = ImageDraw.Draw(im)
            d.rectangle([0,0,Wp,80], fill=(35, 40, 48))
            d.text((20, 28), 'Checkout: Simple, fast, trustworthy', fill=(255,255,255), font=font)
            # left form
            d.rounded_rectangle([60,120, 900, 900], radius=12, fill=(255,255,255), outline=(224,224,230))
            d.text((80,140), 'Contact & Payment (Apple/Google Pay on top, guest checkout)', fill=(20,20,20), font=font)
            # express pay buttons
            d.rounded_rectangle([80, 180, 340, 230], radius=10, fill=(0,0,0))
            d.text((100, 195), 'Apple Pay', fill=(255,255,255), font=font)
            d.rounded_rectangle([360, 180, 620, 230], radius=10, fill=(66,133,244))
            d.text((380, 195), 'Google Pay', fill=(255,255,255), font=font)
            # fields
            y=260
            for _ in range(6):
                d.rectangle([80, y, 860, y+52], fill=(245,245,248), outline=(230,230,235)); y+=66
            d.text((80, y+10), 'Trust badges | Money-back guarantee | Security lock icon', fill=(80,80,80), font=font)
            # right order summary
            d.rounded_rectangle([940, 120, 1380, 480], radius=12, fill=(255,255,255), outline=(224,224,230))
            d.text((960, 140), 'Order Summary', fill=(20,20,20), font=font)
            d.rectangle([960, 180, 1360, 210], fill=(240,240,244))
            d.rectangle([960, 220, 1360, 250], fill=(240,240,244))
            d.rectangle([960, 260, 1360, 290], fill=(240,240,244))
            d.rounded_rectangle([1000, 400, 1340, 460], radius=18, fill=(34,139,230))
            d.text((1040, 418), 'Complete Purchase', fill=(255,255,255), font=font)
            # notes
            notes=[
                'Minimize fields; inline validation; saved payment.',
                'Upfront taxes/fees; no surprises; trust badges near CTA.',
                'Recovery: cart/checkout reminders 30–60m as backup.'
            ]
            yy=520
            for n in notes:
                d.text((940, yy), f"• {n}", fill=(30,30,30), font=font); yy+=24
            im.save(co)

        # Embed both
        for pth, ttl in [(lp, 'Landing Page Wireframe (with fixes)'), (co, 'Checkout Wireframe (with fixes)')]:
            img = ImageReader(str(pth))
            iw, ih = img.getSize()
            maxw = W - 2*margin
            scale = maxw / iw
            w = maxw
            h = ih * scale
            if c._curr_y - h < margin:
                c.showPage(); c._curr_y = H - margin
            c.setFont('Helvetica-Bold', 12)
            c.drawString(margin, c._curr_y, ttl)
            c._curr_y -= 14
            c.drawImage(img, margin, c._curr_y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
            c._curr_y -= (h + 16)

    except Exception as e:
        # If Pillow missing, skip wireframes
        pass

    # Creative roadmap
    text_block('Creative Roadmap (next 10 concepts)', [
        'All concepts: 9:16 Reels primary, 1:1 secondary; 15–30s; hook <2s; captions; dynamic subs.',
        '1) Proof-to-solution: Cold open “Your SSN was found on X sites.” Immediately pivot to setup demo; CTA “Get Protected in Minutes.” Shots: over-shoulder phone, UI close-ups, parent face.',
        '2) Install-in-minutes: Stopwatch overlay; 3-step install; live protection toggles. Shots: screen record + hand B-roll; end with family shot.',
        '3) Parent story UGC: “We caught a fraud attempt in 48h.” Before/after notif screenshots; CTA. Shots: selfie, text overlays.',
        '4) Comparison: “Manual removal vs Aura.” Side-by-side time and risk; price tag; guarantee. Shots: split-screen UI + presenter.',
        '5) Data breach explainer: Visualize leaks; blur PII; then shield animation + plan value. Shots: motion graphics + app UI.',
        '6) Device safety demo: Kid’s phone protection; content filters; location safety. Shots: parent + teen interaction + UI.',
        '7) Social proof carousel: 3 quick reviews, star animations; money-back badge; CTA. Shots: review cards + presenter punch-in.',
        '8) Offer clarity: Trial terms, what’s included, cancel anytime; no surprises. Shots: pricing screen + confident VO.',
        '9) Myth-busting: “Removing my info once is enough?” Bust with facts; show continuous monitoring. Shots: presenter + graphics.',
        '10) Urgency-lite: “New breaches daily—set it and forget it.” Subtle urgency + reassurance. Shots: calendar tick + app alerts.'
    ])

    # Landing & Checkout checklist with impact bands
    text_block('Landing/Checkout Fixes (expected CAC impact)', [
        'Speed: LCP <2.5s (preconnect/prefetch, defer non-critical JS, compress images) → −10–15% CAC.',
        'Above-the-fold clarity: benefit stack + price/term + trust badges + primary CTA → −5–10% CAC.',
        'Form friction: guest checkout, auto-complete, Apple/Google Pay, upfront taxes/fees → −10–20% CAC.',
        'Trust & proof: live count of items removed/monitored, reviews near CTA → −3–7% CAC.',
        'Recovery: cart/checkout reminders 30–60m with single-use code test → +3–8% purchases.'
    ])

    c.showPage(); c.save()


def main():
    data = load_json(JSON_PATH)
    rows = write_csv(data, CSV_PATH)
    try:
        render_pdf(data, rows, PDF_PATH)
    except Exception as e:
        # Still leave CSV even if PDF lib missing
        print(f"PDF render failed: {e}")
    print(json.dumps({
        'csv': str(CSV_PATH),
        'pdf': str(PDF_PATH),
        'count': len(rows)
    }))


if __name__ == '__main__':
    main()
