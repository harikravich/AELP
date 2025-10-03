#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from context.loader import bundle_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = bundle_manifest()
    for rel in manifest['include']:
        src = Path(rel)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    print(f"Wrote context bundle with {len(manifest['include'])} files to {out}")


if __name__ == '__main__':
    main()

