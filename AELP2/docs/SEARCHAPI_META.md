SearchAPI.io — Meta Ad Library Fetcher

What this does
- Pulls Meta Ad Library results via SearchAPI.io’s self‑serve API.
- Uses your chips from `AELP2/config/bigspy_filters.yaml` (semicolon‑separated terms).
- Writes CSVs to `AELP2/vendor_imports/`, which our importer already supports.

Setup
- Sign up at searchapi.io → copy your API key.
- Add to `.env` on the VM: `SEARCHAPI_API_KEY=...`

Run
```bash
source .env
python3 AELP2/tools/fetch_searchapi_meta.py \
  --filters AELP2/config/bigspy_filters.yaml \
  --countries GB,US \
  --days 365 \
  --max-per-query 250

# then the normal pipeline
python3 AELP2/tools/import_vendor_meta_creatives.py --src AELP2/vendor_imports
python3 AELP2/tools/build_features_from_creative_objects.py
python3 AELP2/tools/score_vendor_creatives.py
```

Notes
- EU/UK (GB + EU codes) return all ad categories. US returns political/issue only via Ad Library.
- The fetcher batches one request per chip per country and paginates up to `--max-per-query`.
- CSV columns map to our importer’s expected fields: `ad_archive_id`, `title`, `ad_text`, `destination_url`, `page_id`, `platform`, etc.

