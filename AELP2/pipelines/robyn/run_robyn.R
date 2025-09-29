#!/usr/bin/env Rscript
# Minimal Robyn runner stub. In a real run, this script would:
# 1) Read ads_campaign_daily from BigQuery (using bigrquery)
# 2) Fit a Robyn model and compute elasticities/response curve
# 3) Emit a small JSON summary to stdout; Python wrapper writes to BQ

args <- commandArgs(trailingOnly = TRUE)
start <- if (length(args) >= 1) args[[1]] else Sys.Date() - 90
end <- if (length(args) >= 2) args[[2]] else Sys.Date()
cat(jsonlite::toJSON(list(status="stub", start=as.character(start), end=as.character(end))))

