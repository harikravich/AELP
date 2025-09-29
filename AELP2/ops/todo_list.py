#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

TODO_PATH = os.path.join(os.path.dirname(__file__), '..', 'docs', 'TODO.md')


def parse_todo(md_text: str):
    section = None
    tasks = []
    sec_re = re.compile(r'^##\s+(.+?)\s*$')
    item_re = re.compile(r'^\s*-\s+(.*)$')
    # Status tokens appear commonly as ": Done", ": In Progress", ": Pending" or within parentheses
    status_re = re.compile(r'\b(Done|In Progress|Pending)\b', re.IGNORECASE)
    for line in md_text.splitlines():
        m = sec_re.match(line)
        if m:
            section = m.group(1).strip()
            continue
        m = item_re.match(line)
        if m:
            text = m.group(1).strip()
            sm = status_re.search(text)
            status = sm.group(1).title() if sm else 'Unknown'
            tasks.append({
                'section': section or 'Unsectioned',
                'text': text,
                'status': status,
            })
    return tasks


def load_md(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_outputs(tasks, out_json: str = None, out_md: str = None):
    if out_json:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'generated_at': datetime.utcnow().isoformat() + 'Z', 'tasks': tasks}, f, indent=2)
    if out_md:
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write('# Flattened TODO List\n\n')
            for t in tasks:
                f.write(f"- [{t['status']}] {t['section']} — {t['text']}\n")


def print_tasks(tasks, limit=None):
    count = 0
    for t in tasks:
        line = f"[{t['status']}] {t['section']} — {t['text']}"
        print(line)
        count += 1
        if limit and count >= limit:
            break
    print(f"\nTotal tasks: {len(tasks)}")


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', default=TODO_PATH, help='Path to TODO.md')
    p.add_argument('--out-json', default=os.path.join(os.path.dirname(TODO_PATH), 'TODO_LIST.json'))
    p.add_argument('--out-md', default=os.path.join(os.path.dirname(TODO_PATH), 'TODO_LIST.md'))
    p.add_argument('--watch', action='store_true', help='Continuously watch the TODO for changes')
    p.add_argument('--interval', type=float, default=5.0)
    p.add_argument('--max-loops', type=int, default=0, help='Max watch iterations (0 = infinite)')
    p.add_argument('--print-limit', type=int, default=0, help='Limit printed tasks (0 = all)')
    args = p.parse_args()

    try:
        md = load_md(args.path)
    except FileNotFoundError:
        print(f"ERROR: TODO not found at {args.path}", file=sys.stderr)
        sys.exit(1)

    def build_and_emit(src_text: str):
        tasks = parse_todo(src_text)
        write_outputs(tasks, args.out_json, args.out_md)
        print_tasks(tasks, None if args.print_limit <= 0 else args.print_limit)
        return tasks

    if not args.watch:
        build_and_emit(md)
        return

    last_hash = sha256(md)
    loops = 0
    tasks = build_and_emit(md)
    while True:
        time.sleep(args.interval)
        loops += 1
        if args.max_loops and loops >= args.max_loops:
            print("[watch] Reached max loops; exiting.")
            return
        try:
            cur = load_md(args.path)
        except Exception as e:
            print(f"[watch] Read error: {e}")
            continue
        h = sha256(cur)
        if h != last_hash:
            print(f"\n[watch] Detected change at {datetime.utcnow().isoformat()}Z; rebuilding...")
            tasks = build_and_emit(cur)
            last_hash = h


if __name__ == '__main__':
    main()

