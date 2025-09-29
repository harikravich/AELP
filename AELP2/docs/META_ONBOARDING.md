Meta (Facebook) Ads Onboarding — OAuth and System User

Goal
- Enable read-only access to Meta Marketing API so we can ingest ad-level performance into BigQuery via `AELP2/pipelines/meta_to_bq.py`.

Two Paths
- Quick (OAuth user token): Fastest way to pull your own ad account’s data. Token lasts ~60 days; renew as needed.
- Production (Business System User): Server-to-server token tied to your Business Manager. Requires Business Manager and likely Business Verification + Ads Management Standard Access.

Prereqs
- You have an ad account ID (format `act_123...`; without `act_` the numeric ID also works in our scripts).
- Your user has at least Analyst access to the ad account.

Environment Variables
- `META_APP_ID`, `META_APP_SECRET`, `META_ACCESS_TOKEN`, `META_ACCOUNT_ID`
- Optional: `META_API_VERSION` (default `v21.0`), `AELP2_REDACT_TEXT=1` to store only name hashes.

Path A — Quick (OAuth User Token)
1) Create an app (developers.facebook.com → My Apps → Create App → Business type recommended).
2) Add “Marketing API” product to the app.
3) Generate a short-lived user token with `ads_read` permission:
   - Tools → Graph API Explorer → pick your app → Get User Access Token → select `ads_read` (and `business_management` if prompted) → Generate.
4) Exchange for a long-lived (~60-day) token:
   - `curl -G \
     'https://graph.facebook.com/v21.0/oauth/access_token' \
     --data-urlencode 'grant_type=fb_exchange_token' \
     --data-urlencode "client_id=$META_APP_ID" \
     --data-urlencode "client_secret=$META_APP_SECRET" \
     --data-urlencode "fb_exchange_token=<SHORT_LIVED_TOKEN>"`
   - Set `META_ACCESS_TOKEN` to the returned token.
5) Save creds (optional helper):
   - `META_APP_ID=... META_APP_SECRET=... META_ACCESS_TOKEN=... META_ACCOUNT_ID=... bash AELP2/scripts/platform_onboarding/init_meta.sh`
6) Test Insights:
   - `curl -s "https://graph.facebook.com/v21.0/act_${META_ACCOUNT_ID}/insights?access_token=${META_ACCESS_TOKEN}&level=ad&fields=date_start,ad_id,campaign_id,impressions,clicks,spend&time_range={\"since\":\"2025-09-01\",\"until\":\"2025-09-15\"}" | jq .`

Path B — Production (Business System User)
1) Create/Use Meta Business Manager; add your ad account as an asset.
2) App setup (Business type), add Marketing API.
3) In Business Settings → Users → System Users: create a System User and assign your app.
4) Assets → Assign your ad account to the System User with at least read/analyst access.
5) Generate a System User access token for the app with `ads_read` scope. Depending on your setup, tokens may be expiring or non-expiring; follow the UI options shown in Business Settings.
6) Store `META_ACCESS_TOKEN` and `META_ACCOUNT_ID` as above. If permissions errors occur, complete Business Verification and enable Ads Management Standard Access for your app.

Run the Ingestion
- Ensure BQ vars: `GOOGLE_CLOUD_PROJECT`, `BIGQUERY_TRAINING_DATASET`.
- Dry-run schema ensure: `python3 AELP2/pipelines/meta_to_bq.py --start 2025-09-01 --end 2025-09-01 --account $META_ACCOUNT_ID`
- Typical backfill: `python3 AELP2/pipelines/meta_to_bq.py --start 2025-08-01 --end 2025-09-15 --account $META_ACCOUNT_ID`

What We Ingest
- Ad-level daily rows with impressions, clicks, spend, CTR/CPC, and a best-effort conversions/revenue from `actions`/`action_values` (prefers lead/purchase types). Ad names are hashed if `AELP2_REDACT_TEXT=1`.

Troubleshooting
- `(#200) Permissions error` → Ensure the token is for a user/system user that has access to that ad account; for apps in Dev Mode, only app admins/developers/testers can call.
- `Unsupported get request` → Check the account id format (use `act_123...` or just the numeric id).
- Empty rows → Confirm date range has delivery; try `level=campaign` to sanity check quickly.
- Rate limits → The loader paginates with `paging.next`; if you hit rate caps, rerun for smaller windows.

