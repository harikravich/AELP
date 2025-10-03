#!/usr/bin/env python3
from __future__ import annotations
import os, sys

def mask(v: str) -> str:
    if not v: return 'MISSING'
    return v[:4] + '…' + v[-4:]

def main():
    ok = True
    oa = os.getenv('OPENAI_API_KEY')
    an = os.getenv('ANTHROPIC_API_KEY')
    gm = os.getenv('GEMINI_API_KEY')
    print('Keys:')
    print('  OPENAI_API_KEY   =', mask(oa))
    print('  ANTHROPIC_API_KEY=', mask(an))
    print('  GEMINI_API_KEY   =', mask(gm))

    # Optional: issue low-cost capability calls (model list) to verify access
    # Skip if keys missing
    try:
        if an:
            from anthropic import Anthropic
            ac = Anthropic(api_key=an)
            models = list(ac.models.list())
            print(f"Anthropic OK: {len(models)} models visible")
        else:
            print('Anthropic skipped (no key)')
    except Exception as e:
        ok = False
        print('Anthropic error:', e)

    try:
        if oa:
            # OpenAI: simple test without sending a chat (list models)
            from openai import OpenAI
            oc = OpenAI(api_key=oa)
            _ = list(oc.models.list())
            print('OpenAI OK: models list returned')
        else:
            print('OpenAI skipped (no key)')
    except Exception as e:
        ok = False
        print('OpenAI error:', e)

    try:
        if gm:
            import google.generativeai as genai
            genai.configure(api_key=gm)
            ms = genai.list_models()
            print('Gemini OK: model list returned')
        else:
            print('Gemini skipped (no key)')
    except Exception as e:
        ok = False
        print('Gemini error:', e)

    if not ok:
        sys.exit(1)

if __name__ == '__main__':
    main()

