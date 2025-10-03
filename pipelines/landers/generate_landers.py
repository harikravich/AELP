#!/usr/bin/env python3
"""
Generate static landers from templates + config, adding GA events.

Usage:
  python pipelines/landers/generate_landers.py --config pipelines/landers/sample_landers.yaml --outdir artifacts/landers
"""
from __future__ import annotations
import argparse
import yaml  # type: ignore
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape  # type: ignore


def render_lander(env, spec: dict) -> str:
    template = env.get_template('lander.html.j2')
    return template.render(**spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    specs = yaml.safe_load(Path(args.config).read_text())
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / 'templates')),
        autoescape=select_autoescape(['html', 'xml'])
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for s in specs['landers']:
        html = render_lander(env, s)
        fname = f"{s['slug']}.html"
        (outdir / fname).write_text(html)
        print(f"Wrote {fname}")


if __name__ == '__main__':
    main()

