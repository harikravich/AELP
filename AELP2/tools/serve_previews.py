#!/usr/bin/env python3
from __future__ import annotations
"""
Serve AELP2/outputs/finals on http://127.0.0.1:8080 for local/SSH-tunnel preview.
"""
import http.server, socketserver
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / 'AELP2' / 'outputs' / 'finals'

def main():
    os.chdir(DIR)
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(('127.0.0.1', 8080), Handler) as httpd:
        print(f"Serving {DIR} at http://127.0.0.1:8080")
        httpd.serve_forever()

if __name__=='__main__':
    DIR.mkdir(parents=True, exist_ok=True)
    main()

