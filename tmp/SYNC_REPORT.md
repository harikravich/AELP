# AELP Local vs Remote Sync Report

**Date:** Thu Oct  2 20:06:22 EDT 2025
**Remote Source:** `merlin-l4-1:/home/harikravich_gmail_com/AELP` (GCP us-central1-c)
**Local Target:** Current directory

## Summary

- **Total remote entries:** 154 (excluding gitignore patterns)
- **Total local entries:** 2648 (excluding gitignore patterns)
- **Missing locally:** 150 files/directories
- **Extra locally:** 2644 files/directories (not on remote)

## EXCLUSIONS

**The following items are EXCLUDED from the copy plan:**
- **Log files** (`logs/`, `runs/*.pid`) - Training logs and process files
- **Downloaded creative assets** (`assets/meta_creatives/`, `assets/veo_videos/`, `assets/veo_balance/`) - Downloaded images/videos
- **Manifest files** (`assets/*_manifest.csv`) - CSV catalogs of downloaded assets

## Files/Directories to Copy

### Priority 1: Artifacts (40 items)

The `artifacts/` directory contains:
- **Creative features:** Meta creative features, Veo video features
- **Feature engineering:** Marketing CTR features, enhanced features, catalogs
- **Trained models:** CTR classifiers and rankers (~9MB total)
- **Predictions:** CTR scores, current running ads scores
- **Priors:** Thompson Sampling strategies
- **Validation results:** Forward holdout and ranking evaluation metrics

### Priority 2: Pipelines (57 items)

The `pipelines/` directory contains:
- **CTR pipeline:** Training and prediction scripts
- **Validation:** Forward holdout and metrics
- **Creative processing:** Feature extraction, manifest generation
- **Data processing:** Meta/GA4 unification, synthetic data
- **Additional utilities:** Bandit orchestration, feature engineering, etc.

### Priority 3: Scripts (26 items)

The `scripts/` directory contains utility scripts.

### Priority 4: Tools (7 items)

The `tools/` directory contains Meta API fetchers and creative utilities.

### Priority 5: Requirements (1 items)

GPU-specific requirements file.

### Priority 6: Root (14 items)

Root-level utility scripts and YOLOv8 model weights.

## Recommended Copy Commands

Copy files individually using `gcloud compute scp`:

```bash
# Copy commands for missing files

# Copy artifacts/ directory
gcloud compute scp --recurse --zone=us-central1-c \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/artifacts \
  ./

# Copy pipelines/ directory
gcloud compute scp --recurse --zone=us-central1-c \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/pipelines \
  ./

# Copy scripts/ directory
gcloud compute scp --recurse --zone=us-central1-c \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/scripts \
  ./

# Copy tools/ directory
gcloud compute scp --recurse --zone=us-central1-c \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/tools \
  ./

# Copy individual files
gcloud compute scp --zone=us-central1-c \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/runs \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/artifacts \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/assets \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/assets/meta_creatives \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/assets/veo_balance \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/assets/veo_balance_manifest.csv \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/assets/veo_videos \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/assets/veo_videos_manifest.csv \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/build_unique_today.py \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/current_link_report.py \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/current_running_report.py \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/generate_latest.py \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/logs \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/pipelines \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/requirements \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/requirements/requirements-gpu.txt \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/scripts \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/show_video_scores.py \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/tools \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/yolov8n.pt \
  ./

# Copy requirements
mkdir -p ./requirements
gcloud compute scp --zone=us-central1-c \
  merlin-l4-1:/home/harikravich_gmail_com/AELP/requirements/requirements-gpu.txt \
  ./requirements/

```

## File Size Estimates

Based on sampling (excluding creative assets and logs):
- **artifacts/models/**: ~9 MB (3 trained models)
- **artifacts/features/**: ~1 MB (Parquet feature files)
- **artifacts/predictions/**: ~100 KB (prediction scores)
- **pipelines/**: ~100-500 KB (Python scripts)
- **scripts/**: ~50-100 KB (utility scripts)
- **tools/**: ~50-100 KB (API integration tools)

**Total estimated download:** ~10-15 MB (much smaller without creative assets)

## Verification After Copy

After copying files, verify with:

```bash
# Check artifacts were copied
ls -lh ./artifacts/models/

# Check pipelines
ls -R ./pipelines/

# Check scripts and tools
ls ./scripts/
ls ./tools/

# Verify file counts
find ./artifacts -type f | wc -l  # Should be ~40
find ./pipelines -type f | wc -l  # Should be ~57
```
