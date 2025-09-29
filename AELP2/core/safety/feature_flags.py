"""
Feature Flags SDK integration (GrowthBook/Unleash/Flagsmith compatible).

Server-side fetch with simple in-memory cache. If no provider configured,
falls back to environment variables and optional local JSON, same as
feature_gates.py.
"""
from __future__ import annotations

import os
import json
import time
import threading
from typing import Any, Dict

_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {'ts': 0, 'flags': {}}


def _fetch_flags() -> Dict[str, Any]:
    # GrowthBook
    gb_api = os.getenv('GROWTHBOOK_API_HOST')
    gb_key = os.getenv('GROWTHBOOK_SERVER_KEY')
    if gb_api and gb_key:
        try:
            import urllib.request
            req = urllib.request.Request(f"{gb_api}/api/features")
            req.add_header('Authorization', f"Bearer {gb_key}")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read().decode('utf-8'))
                return {k: v.get('defaultValue', False) for k, v in data.get('features', {}).items()}
        except Exception:
            pass
    # Flagsmith
    fs_env = os.getenv('FLAGSMITH_ENVIRONMENT_KEY')
    fs_host = os.getenv('FLAGSMITH_API') or 'https://edge.api.flagsmith.com/api/v1'
    if fs_env:
        try:
            import urllib.request
            req = urllib.request.Request(f"{fs_host}/flags/?environment={fs_env}")
            with urllib.request.urlopen(req, timeout=3) as r:
                arr = json.loads(r.read().decode('utf-8'))
                return {x['feature']['name']: x.get('enabled', False) for x in arr}
        except Exception:
            pass
    # Unleash
    un_url = os.getenv('UNLEASH_URL')
    un_key = os.getenv('UNLEASH_API_TOKEN')
    if un_url and un_key:
        try:
            import urllib.request
            req = urllib.request.Request(f"{un_url}/api/client/features")
            req.add_header('Authorization', un_key)
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read().decode('utf-8'))
                return {f['name']: f.get('enabled', False) for f in data.get('features', [])}
        except Exception:
            pass
    # Fallback: local JSON
    path = os.getenv('GATES_FLAGS_JSON')
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _get_flags_cached() -> Dict[str, Any]:
    now = time.time()
    with _cache_lock:
        if now - _cache.get('ts', 0) > int(os.getenv('AELP2_FLAGS_TTL_SEC', '60') or '60'):
            _cache['flags'] = _fetch_flags()
            _cache['ts'] = now
        return dict(_cache.get('flags', {}))


def is_enabled(flag: str) -> bool:
    flags = _get_flags_cached()
    if flag in flags:
        return bool(flags[flag])
    # Env fallback
    return os.getenv(flag, '0') == '1'

