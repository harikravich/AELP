#!/usr/bin/env python3
import sys
import re
from datetime import datetime, timezone


def mark(path: str, needle: str, status: str):
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    pattern = re.compile(rf"({re.escape(needle)}[\s\S]*?)(\n\n|\Z)")
    m = pattern.search(txt)
    if not m:
        print(f'warn: section not found: {needle}')
        return
    block = m.group(1)
    if f"Status: {status}" in block:
        return
    block = block.rstrip() + f"\n  - Status: {status} ({ts})\n"
    txt = txt[:m.start(1)] + block + txt[m.end(1):]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)


def main():
    if len(sys.argv) < 3:
        print('Usage: update_status.py <needle> <status>')
        sys.exit(1)
    path = 'AELP2/docs/TODO_RECONCILED.md'
    mark(path, sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()

