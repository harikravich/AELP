# Meta Ad Library Activation Check — 2025-09-28

- Ran `python3 AELP2/tools/fetch_meta_adlibrary.py` with a freshly supplied token (Graph API Explorer, User Token).
- Result: all queries returned non‑200 (`OAuthException code 1`), so no live Ad Library rows were ingested. Fallback to cached Aura creative objects remains active.

## What actually unlocks the Ad Library API
- Ad Library access is gated at the **person level**, not via App Review permissions. You must complete the identity/location steps at the Ad Library API start page and generate a **User access token** from that same confirmed profile. System‑user tokens won’t satisfy this gate.

## Exact steps to complete (manual)
1) Confirm identity + location (the gate)
   - Log in to the Facebook profile you’ll use for access.
   - Visit the Ad Library API start page and complete identity + location confirmation (may require postal code verification in some countries). Once green, the page shows you as authorized.
2) Generate a User access token
   - Open Graph API Explorer, select `AURA Reporting App`, set "User or Page" to `User`, and add no special permissions for Ad Library (Marketing API scopes like `ads_read` are optional here).
   - Click "Generate Access Token" and ensure `Node: me` returns your name.
3) Configure env for the fetcher
   - Set `.env`: `META_ADLIBRARY_ACCESS_TOKEN=<that User token>`, `META_API_VERSION=v23.0`, `META_ADLIBRARY_COUNTRIES=GB` (GB/EU returns all ad categories; US returns only political/issue without `ad_type=POLITICAL_AND_ISSUE_ADS`).
4) Run the fetcher
   - `source .env && python3 AELP2/tools/fetch_meta_adlibrary.py`
   - Expect `count_adlibrary > 0` and `AELP2/competitive/ad_items_raw.json` to show nonzero `fetched`.

Notes
- Keep `META_ACCESS_TOKEN` (system‑user) for Marketing API/Insights only.
- When the gate isn’t complete, common responses include `code 10 / subcode 2332002` and `code 1` (generic). The fix is finishing identity confirmation and using a User token from that same profile.
