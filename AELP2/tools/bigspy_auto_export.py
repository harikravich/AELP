#!/usr/bin/env python3
from __future__ import annotations
"""
Headless BigSpy exporter (best-effort, cookie-based login reuse)

What it does
- Reuses your existing BigSpy session via cookies (Netscape cookies.txt or JSON)
- Opens BigSpy in a headless Chromium, optionally applies basic filters
- Scrolls results and scrapes core creative metadata
- Writes a CSV to AELP2/vendor_imports/bigspy_export_YYYYMMDD_HHMM.csv

Auth model
- You must provide a cookies file captured after logging into BigSpy in a normal browser
  (recommended: "Get cookies.txt" extension). DO NOT share this in chat; place it on disk.

Usage
  source .env
  python3 AELP2/tools/bigspy_auto_export.py \
    --cookies AELP2/secrets/bigspy_cookies.txt \
    --filters AELP2/config/bigspy_filters.yaml \
    --max 800

Notes
- The BigSpy UI changes over time; selectors here are defensive with fallbacks.
- If UI automation of filters fails, the script will still attempt to scrape whatever
  results are visible after login.
"""
import argparse, csv, json, os, re, sys, time, importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / 'AELP2' / 'vendor_imports'


def require(module: str, pip_name: str | None = None):
    """Import a module, installing it via pip if missing."""
    try:
        return importlib.import_module(module)
    except ImportError:
        sys.stderr.write(f"Installing missing dependency: {pip_name or module}\n")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name or module])
        return importlib.import_module(module)


# Import Playwright sync API explicitly (submodule) and yaml
playwright_sync = require('playwright.sync_api', 'playwright')
yaml = require('yaml', 'pyyaml')


@dataclass
class Filters:
    query: str = ''                    # semicolon-separated terms
    country: str = ''                  # e.g., United States
    language: str = 'English'
    networks: list[str] | None = None  # e.g., ["Facebook", "Instagram"]
    os_list: list[str] | None = None   # e.g., ["iOS", "Android"]
    dedup: str = 'Strict'              # Strict | AI | Advertiser
    sort: str = 'Popularity'           # Popularity | Recent
    only_new: bool = False


def load_filters(path: str|None) -> Filters:
    if not path:
        return Filters(
            query='antivirus; vpn; identity protection; credit monitoring; phishing; dark web; password manager; aura; behavioral health; anxiety; mental health',
            country='United States', language='English',
            networks=['Facebook', 'Instagram'], os_list=['iOS', 'Android'],
            dedup='Strict', sort='Popularity', only_new=False,
        )
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Filters(
        query=str(data.get('query', '')),
        country=str(data.get('country', '')),
        language=str(data.get('language', 'English')),
        networks=list(data.get('networks', ["Facebook", "Instagram"])),
        os_list=list(data.get('os', ["iOS", "Android"])),
        dedup=str(data.get('dedup', 'Strict')),
        sort=str(data.get('sort', 'Popularity')),
        only_new=bool(data.get('only_new', False)),
    )


def parse_cookies_file(path: Path) -> list[dict]:
    """Accept Netscape cookies.txt or a JSON list of cookie dicts.
    Returns Playwright cookie dicts.
    """
    raw = path.read_text(encoding='utf-8').strip()
    cookies: list[dict[str, Any]] = []
    if raw.startswith('['):
        # JSON cookies
        data = json.loads(raw)
        for c in data:
            if not isinstance(c, dict):
                continue
            domain = c.get('domain') or c.get('Domain')
            name = c.get('name') or c.get('Name')
            value = c.get('value') or c.get('Value')
            pathv = c.get('path', '/')
            secure = bool(c.get('secure', False))
            httpOnly = bool(c.get('httpOnly', False))
            expires = c.get('expires') or c.get('Expiration') or 0
            sameSite = c.get('sameSite', 'Lax')
            if domain and name and value:
                cookies.append({
                    'domain': domain.lstrip('.'), 'name': name, 'value': value,
                    'path': pathv or '/', 'secure': secure, 'httpOnly': httpOnly,
                    'expires': int(expires) if isinstance(expires, (int, float)) else 0,
                    'sameSite': sameSite if sameSite in ('Lax', 'None', 'Strict') else 'Lax',
                })
        return cookies
    # Netscape cookies.txt
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 7:
            continue
        domain, _flag, pathv, secure_flag, expires, name, value = parts[:7]
        cookies.append({
            'domain': domain.lstrip('.'), 'name': name, 'value': value,
            'path': pathv or '/', 'secure': (secure_flag.upper() == 'TRUE'),
            'httpOnly': False, 'expires': int(expires) if expires.isdigit() else 0,
            'sameSite': 'Lax',
        })
    return cookies


def add_cookies(context, cookies: list[dict]):
    # Add cookies for both root and www subdomain to be safe.
    if not cookies:
        return
    expanded = []
    for c in cookies:
        dom = c.get('domain') or ''
        if dom.startswith('.'):
            dom = dom.lstrip('.')
        base = c.copy(); base['domain'] = dom
        expanded.append(base)
        if not dom.startswith('www.'):
            www = c.copy(); www['domain'] = f"www.{dom}"
            expanded.append(www)
    context.add_cookies(expanded)


def visible(page, text: str):
    return page.locator(f'text={text}').first.is_visible()


def set_text_near_label(page, label_text: str, value: str):
    try:
        # Try role=form-like input under/near label text
        lab = page.locator(f'text="{label_text}"').first
        if not lab.count():
            return False
        # Search an input within the same section
        container = lab.locator('xpath=ancestor::*[self::div or self::section][1]')
        inp = container.locator('input, textarea').first
        if inp.count():
            inp.click()
            inp.fill(value)
            return True
    except Exception:
        return False
    return False


def click_radio_or_button(page, label_text: str, target_text: str) -> bool:
    try:
        sec = page.locator(f'text="{label_text}"').first
        if not sec.count():
            return False
        container = sec.locator('xpath=ancestor::*[self::div or self::section][1]')
        btn = container.locator(f'text="{target_text}"').first
        if btn.count():
            btn.click()
            return True
    except Exception:
        return False
    return False


def apply_filters(page, f: Filters):
    # BigSpy changes often; do best-effort targeting by visible labels.
    # Ads Text
    if f.query:
        set_text_near_label(page, 'Ads Text', f.query)
    # Networks
    if f.networks:
        for net in f.networks:
            click_radio_or_button(page, 'Networks', net)
    # Language
    if f.language:
        click_radio_or_button(page, 'Language', f.language)
    # Country/Region
    if f.country:
        set_text_near_label(page, 'Country/Region', f.country)
        # try selecting suggested item
        try:
            page.keyboard.press('Enter')
        except Exception:
            pass
    # OS
    if f.os_list:
        for osname in f.os_list:
            click_radio_or_button(page, 'OS', osname)
    # Dedup
    if f.dedup:
        click_radio_or_button(page, 'Strict Deduplication', 'Strict Deduplication')
    # Only New Ads
    if f.only_new:
        click_radio_or_button(page, 'Only New Ads', 'Only New Ads')
    # Sort
    if f.sort:
        # Open sort control if present
        try:
            page.locator('text=Sort By').first.click()
            page.locator(f'text={f.sort}').first.click()
        except Exception:
            pass
    # Submit/Search
    try:
        page.locator('text=Search').first.click()
    except Exception:
        # Some pages auto-apply; ignore
        pass


def scrape_cards(page, already: set[str]) -> list[dict]:
    rows = []
    # A very generic card heuristic: items that include Advertiser / Popularity / Duration etc.
    cards = page.locator('article, div.card, div:has-text("Popularity")').all()[:200]
    for c in cards:
        try:
            txt = c.inner_text(timeout=100)
        except Exception:
            continue
        # Attempt to recover key fields
        advertiser = None
        ad_text = None
        domain = None
        popularity = None
        last_seen = None
        m = re.search(r'Advertiser\s*\n?\s*(.+)', txt, re.I)
        if m:
            advertiser = m.group(1).strip().splitlines()[0]
        m = re.search(r'Ads?\s*Text\s*\n?\s*(.+)', txt, re.I)
        if m:
            ad_text = m.group(1).strip()
        m = re.search(r'Domain\s*\n?\s*(.+)', txt, re.I)
        if m:
            domain = m.group(1).strip().splitlines()[0]
        m = re.search(r'Popularity\s*\n?\s*(\d+)', txt, re.I)
        if m:
            popularity = m.group(1)
        m = re.search(r'Last\s*Seen\s*\n?\s*(\d{4}-\d{2}-\d{2})', txt, re.I)
        if m:
            last_seen = m.group(1)

        key = (advertiser or '') + '|' + (ad_text or '')[:64]
        if not ad_text or key in already:
            continue
        already.add(key)
        rows.append({
            'advertiser': advertiser or '',
            'ad_text': ad_text,
            'domain': domain or '',
            'popularity': popularity or '',
            'last_seen': last_seen or '',
        })
    return rows


def run(cookies_path: Path, filters_path: str|None, max_items: int, storage_path: str|None=None) -> Path:
    f = load_filters(filters_path)
    cookies = parse_cookies_file(cookies_path)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"bigspy_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    with playwright_sync.sync_playwright() as p:
        CHROME_UA = (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/128.0.0.0 Safari/537.36'
        )
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=CHROME_UA,
            viewport={'width': 1400, 'height': 900},
        )
        add_cookies(context, cookies)
        # Optional: pre-seed localStorage from a JSON blob exported in the browser
        storage: dict[str, str] = {}
        if storage_path:
            try:
                storage = json.loads(Path(storage_path).read_text())
                # Supported formats:
                # 1) {"key": "value", ...}
                # 2) [{"key": "...", "value": "..."}, ...]
                # 3) {"origin": "https://bigspy.com", "ls": {...}, "ss": {...}}
                ls_map: dict[str, str] = {}
                ss_map: dict[str, str] = {}
                if isinstance(storage, list):
                    ls_map = {str(d.get('key')): str(d.get('value')) for d in storage if isinstance(d, dict) and 'key' in d and 'value' in d}
                elif isinstance(storage, dict):
                    if 'ls' in storage or 'ss' in storage:
                        if isinstance(storage.get('ls'), dict):
                            ls_map = {str(k): str(v) for k, v in storage['ls'].items()}
                        if isinstance(storage.get('ss'), dict):
                            ss_map = {str(k): str(v) for k, v in storage['ss'].items()}
                    else:
                        ls_map = {str(k): str(v) for k, v in storage.items()}
                if ls_map or ss_map:
                    context.add_init_script(
                        f"""
                        try {{
                          const ls = {json.dumps(ls_map)};
                          const ss = {json.dumps(ss_map)};
                          for (const k in ls) {{ localStorage.setItem(k, ls[k]); }}
                          for (const k in ss) {{ sessionStorage.setItem(k, ss[k]); }}
                        }} catch (e) {{}}
                        """
                    )
            except Exception:
                pass
        page = context.new_page()

        # Warm the origin first (sets origin for localStorage/cookies), then go to /ads
        try:
            page.goto('https://bigspy.com', wait_until='domcontentloaded')
        except Exception:
            page.goto('https://www.bigspy.com', wait_until='domcontentloaded')
        page.wait_for_timeout(800)
        try:
            page.goto('https://bigspy.com/ads', wait_until='domcontentloaded')
        except Exception:
            page.goto('https://www.bigspy.com/ads', wait_until='domcontentloaded')
        # Give the site a moment to recognize cookies/session
        page.wait_for_timeout(1500)

        try:
            apply_filters(page, f)
        except Exception:
            pass

        # Wait for results area to appear
        page.wait_for_timeout(2500)

        # Scroll and gather
        seen: set[str] = set()
        rows: list[dict] = []
        last_len = 0
        for _ in range(40):
            rows.extend(scrape_cards(page, seen))
            rows = rows[:max_items]
            if len(rows) >= max_items or len(rows) == last_len:
                break
            last_len = len(rows)
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(600)

        # Persist CSV
        cols = ['advertiser', 'ad_text', 'domain', 'popularity', 'last_seen']
        with out.open('w', newline='', encoding='utf-8') as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in cols})

        # If nothing exported, dump a debug screenshot and HTML snapshot
        if not rows:
            try:
                debug_png = OUTDIR / f"bigspy_debug_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
                debug_html = OUTDIR / f"bigspy_debug_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
                page.screenshot(path=str(debug_png), full_page=True)
                debug_html.write_text(page.content())
            except Exception:
                pass

        context.close()
        browser.close()

    print(json.dumps({'exported': len(rows), 'out': str(out)}, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cookies', required=True, help='Path to cookies.txt (Netscape) or JSON cookies list')
    ap.add_argument('--filters', default=None, help='Path to YAML filters (optional)')
    ap.add_argument('--max', type=int, default=600, help='Max number of rows to scrape')
    ap.add_argument('--storage', default=None, help='Path to JSON localStorage export (optional)')
    args = ap.parse_args()

    cookies_path = Path(args.cookies)
    if not cookies_path.exists():
        sys.stderr.write(f"cookies file not found: {cookies_path}\n")
        sys.exit(2)

    # Ensure playwright browser available (install on demand)
    try:
        from playwright._impl._driver import compute_driver_executable
        _ = compute_driver_executable()
    except Exception:
        # playwright not installed browsers; install Chromium quietly
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'playwright', 'install', 'chromium', '--with-deps'])

    run(cookies_path, args.filters, args.max, args.storage)


if __name__ == '__main__':
    main()
