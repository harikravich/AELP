Headless BigSpy Export — How to provide auth (cookies)

Goal
- Let the headless scraper reuse your already-authenticated BigSpy session.

Why cookies
- BigSpy login is via OAuth; reproducing the entire Google flow headless is brittle and often blocked. Reusing your session cookies is the reliable path for a one-off export.

What I need from you (local, not via chat)
1) Export cookies for bigspy.com using a browser extension that can include HttpOnly cookies:
   - Recommended: “Get cookies.txt” for Chrome/Brave/Edge.
   - Log in to https://www.bigspy.com in your normal browser (ensure you can see ads results).
   - Click the extension → Export for the domain → save as `AELP2/secrets/bigspy_cookies.txt`.
   - This file uses Netscape cookies.txt format (what the script expects).

2) (Optional) Filters config
   - You can tweak `AELP2/config/bigspy_filters.yaml` to adjust search terms, country, networks, etc.

Run it
```bash
source .env
python3 AELP2/tools/bigspy_auto_export.py \
  --cookies AELP2/secrets/bigspy_cookies.txt \
  --filters AELP2/config/bigspy_filters.yaml \
  --max 800
```

Outputs
- CSV: `AELP2/vendor_imports/bigspy_export_YYYYMMDD_HHMM.csv`
- Then normalize and score:
```bash
python3 AELP2/tools/import_vendor_meta_creatives.py --src AELP2/vendor_imports
python3 AELP2/tools/build_features_from_creative_objects.py
python3 AELP2/tools/score_vendor_creatives.py
```
- Scores: `AELP2/reports/vendor_scores.json`

Security notes
- Do not paste cookies or tokens into chat.
- Keep `AELP2/secrets/` out of version control; it’s gitignored.

