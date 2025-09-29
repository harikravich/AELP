#!/usr/bin/env python3
from __future__ import annotations
"""
Prefect flow (optional) to orchestrate: build assets -> assemble themes -> log DNA.
If Prefect is not installed, prints a CLI sequence instead.
"""
import shutil, subprocess

def have_prefect():
    return shutil.which('prefect') is not None

def cli_sequence():
    cmds = [
        'python3 AELP2/tools/build_proof_assets.py',
        'python3 AELP2/tools/assemble_two_screens.py',
        'python3 AELP2/tools/assemble_spot_the_tell.py',
        'python3 AELP2/tools/log_creative_dna.py'
    ]
    for c in cmds:
        print('>>', c)
        subprocess.run(c, shell=True, check=True)

def main():
    if not have_prefect():
        print('Prefect not found; running CLI sequence...')
        cli_sequence()
        return
    from prefect import flow, task

    @task
    def build_assets():
        subprocess.run('python3 AELP2/tools/build_proof_assets.py', shell=True, check=True)

    @task
    def assemble():
        subprocess.run('python3 AELP2/tools/assemble_two_screens.py', shell=True, check=True)
        subprocess.run('python3 AELP2/tools/assemble_spot_the_tell.py', shell=True, check=True)

    @task
    def log_dna():
        subprocess.run('python3 AELP2/tools/log_creative_dna.py', shell=True, check=True)

    @flow(name='creative-batch')
    def creative_batch():
        build_assets(); assemble(); log_dna()

    creative_batch()

if __name__=='__main__':
    main()

