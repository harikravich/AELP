# Creative Publisher Spec (Google Ads RSA/PMax + YouTube attach)

Flow
1) Enqueue: `/api/control/creative/enqueue` inserts a row in `creative_publish_queue` with payload.
2) Publish: Cloud Run job `publish_google_creatives.py` pops `queued`, validates policy, creates assets/ads `PAUSED`, logs platform IDs.
3) Rollback: `/api/control/creative/rollback` pauses/removes by platform IDs.

Payloads (examples)
- RSA `{ type:'rsa', campaign_id, ad_group_id, headlines[], descriptions[], final_url }`
- PMax `{ type:'pmax_asset_group', campaign_id, text_assets[], image_assets[], video_assets[youtube_id], final_url }`
- Video `{ type:'video', youtube_id, campaign_id, asset_group_id }`

Policy Lint
- “Validate only” pass where possible; moderation on text; simple image checks; block on disapproval topics.

Flags & Safety
- Gates: `GATES_ENABLED=1`, `AELP2_ALLOW_BANDIT_MUTATIONS=1`, `ALLOW_REAL_CREATIVE=1` for live; default DRY + paused.

