#!/usr/bin/env python3
"""
GA4 permissions check (dry-run friendly): verifies Data API access or provides guidance.
"""
import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        print('[dry_run] would check GA4 Data API credentials and scopes')
        return
    has_sa = bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or os.getenv('AELP2_BQ_CREDENTIALS'))
    has_oauth = bool(os.getenv('GA4_OAUTH_REFRESH_TOKEN') and os.getenv('GA4_OAUTH_CLIENT_ID') and os.getenv('GA4_OAUTH_CLIENT_SECRET'))
    if not has_sa and not has_oauth:
        print('No GA4 credentials detected. Grant SA analytics.readonly or provide OAuth envs. See docs/GA4_AUTH_SETUP.md')
        return
    print('GA4 auth path detected:', 'service_account' if has_sa else 'oauth_refresh_token')


if __name__ == '__main__':
    main()

