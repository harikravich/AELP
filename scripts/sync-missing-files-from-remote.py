#!/usr/bin/env python3
"""
AELP Local vs Remote Sync Analysis Tool

This script compares the local repository with the remote GCP instance and generates
a comprehensive sync report with copy commands, excluding logs and creative assets.

Usage:
    python3 sync_analysis.py

Output:
    - SYNC_REPORT.md: Complete analysis and copy commands
    - copy_list.txt: List of all missing files
    - local_files.txt: Local file listing
    - remote_files.txt: Remote file listing
"""

import subprocess
import sys
import os
from pathlib import Path

class AELPSyncAnalyzer:
    def __init__(self):
        self.local_files = set()
        self.remote_files = set()
        self.gitignore_patterns = self._load_gitignore_patterns()
        
    def _load_gitignore_patterns(self):
        """Load patterns from .gitignore file."""
        patterns = set()
        gitignore_path = Path('.gitignore')
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Convert gitignore patterns to simpler patterns for filtering
                        if line.endswith('/'):
                            patterns.add(line.rstrip('/'))
                        else:
                            patterns.add(line)
        
        return patterns
    
    def _should_exclude(self, path):
        """Check if a file/directory should be excluded based on gitignore patterns."""
        path_str = str(path)
        
        # Check against gitignore patterns
        for pattern in self.gitignore_patterns:
            if pattern in path_str or path_str.endswith(pattern):
                return True
        
        # Additional exclusions for sync analysis
        exclude_patterns = [
            '__pycache__',
            'venv/',
            'node_modules/',
            '.next/',
            'dist/',
            '.venv/',
            'secrets/',
            'outputs/',
            'vendor_imports/',
            'reports/',
            '.pytest_cache/',
            '.log',
            '.vscode/',
            '.idea/',
            '.DS_Store',
            '.pkl',
            '.h5',
            '.pth',
            'checkpoints/',
            'wandb/',
            'd3rlpy_logs/',
            'venv-',
            # Additional exclusions for sync
            'logs/',
            'assets/meta_creatives/',
            'assets/veo_videos/',
            'assets/veo_balance/',
            'assets/*_manifest.csv',
            'runs/',
        ]
        
        for pattern in exclude_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def get_local_files(self):
        """Get list of local files, excluding gitignore patterns."""
        print("Scanning local files...")
        
        try:
            result = subprocess.run(
                ['find', '.', '-type', 'f', '-o', '-type', 'd'],
                capture_output=True, text=True, cwd='.'
            )
            
            if result.returncode == 0:
                all_files = set(line.strip() for line in result.stdout.split('\n') if line.strip())
                
                # Filter out excluded patterns
                self.local_files = {
                    f for f in all_files 
                    if not self._should_exclude(f) and not f.startswith('./.git/')
                }
                
                print(f"Found {len(self.local_files)} local files (after exclusions)")
            else:
                print(f"Error scanning local files: {result.stderr}")
                
        except Exception as e:
            print(f"Error getting local files: {e}")
    
    def get_remote_files(self):
        """Get list of remote files, excluding gitignore patterns."""
        print("Scanning remote files...")
        
        try:
            cmd = [
                'gcloud', 'compute', 'ssh', 'merlin-l4-1', 
                '--zone=us-central1-c',
                '--command=find /home/harikravich_gmail_com/AELP -type f -o -type d'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                all_files = set()
                for line in result.stdout.split('\n'):
                    if line.strip():
                        # Convert remote path to local relative path
                        if '/home/harikravich_gmail_com/AELP/' in line:
                            rel_path = './' + line.split('/home/harikravich_gmail_com/AELP/')[-1]
                            all_files.add(rel_path)
                
                # Filter out excluded patterns
                self.remote_files = {
                    f for f in all_files 
                    if not self._should_exclude(f)
                }
                
                print(f"Found {len(self.remote_files)} remote files (after exclusions)")
            else:
                print(f"Error scanning remote files: {result.stderr}")
                
        except Exception as e:
            print(f"Error getting remote files: {e}")
    
    def analyze_differences(self):
        """Analyze differences between local and remote."""
        missing_locally = self.remote_files - self.local_files
        extra_locally = self.local_files - self.remote_files
        common_files = self.local_files & self.remote_files
        
        return {
            'missing_locally': missing_locally,
            'extra_locally': extra_locally,
            'common_files': common_files,
            'total_remote': len(self.remote_files),
            'total_local': len(self.local_files)
        }
    
    def categorize_missing_files(self, missing_files):
        """Categorize missing files by directory."""
        categories = {}
        
        for file_path in missing_files:
            path_parts = file_path.lstrip('./').split('/')
            
            if len(path_parts) == 1:
                category = 'ROOT'
            else:
                category = path_parts[0]
            
            if category not in categories:
                categories[category] = []
            categories[category].append(file_path)
        
        return categories
    
    def generate_copy_commands(self, missing_files):
        """Generate copy commands for missing files."""
        categories = self.categorize_missing_files(missing_files)
        
        commands = []
        commands.append("# Copy commands for missing files")
        commands.append("")
        
        # Priority directories to copy entirely
        priority_dirs = ['artifacts', 'pipelines', 'scripts', 'tools']
        
        for dir_name in priority_dirs:
            if dir_name in categories:
                commands.append(f"# Copy {dir_name}/ directory")
                commands.append(f"gcloud compute scp --recurse --zone=us-central1-c \\")
                commands.append(f"  merlin-l4-1:/home/harikravich_gmail_com/AELP/{dir_name} \\")
                commands.append(f"  ./")
                commands.append("")
        
        # Individual files
        individual_files = []
        for category, files in categories.items():
            if category not in priority_dirs:
                individual_files.extend(files)
        
        if individual_files:
            commands.append("# Copy individual files")
            commands.append("gcloud compute scp --zone=us-central1-c \\")
            
            for i, file_path in enumerate(sorted(individual_files)):
                remote_path = f"merlin-l4-1:/home/harikravich_gmail_com/AELP/{file_path.lstrip('./')}"
                if i == len(individual_files) - 1:
                    commands.append(f"  {remote_path} \\")
                else:
                    commands.append(f"  {remote_path} \\")
            
            commands.append("  ./")
            commands.append("")
        
        # Requirements
        if 'requirements' in categories:
            commands.append("# Copy requirements")
            commands.append("mkdir -p ./requirements")
            commands.append("gcloud compute scp --zone=us-central1-c \\")
            commands.append("  merlin-l4-1:/home/harikravich_gmail_com/AELP/requirements/requirements-gpu.txt \\")
            commands.append("  ./requirements/")
            commands.append("")
        
        return '\n'.join(commands)
    
    def generate_report(self, analysis):
        """Generate comprehensive sync report."""
        missing_files = analysis['missing_locally']
        categories = self.categorize_missing_files(missing_files)
        
        report = []
        report.append("# AELP Local vs Remote Sync Report")
        report.append("")
        report.append("**Date:** " + subprocess.run(['date'], capture_output=True, text=True).stdout.strip())
        report.append("**Remote Source:** `merlin-l4-1:/home/harikravich_gmail_com/AELP` (GCP us-central1-c)")
        report.append("**Local Target:** Current directory")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total remote entries:** {analysis['total_remote']} (excluding gitignore patterns)")
        report.append(f"- **Total local entries:** {analysis['total_local']} (excluding gitignore patterns)")
        report.append(f"- **Missing locally:** {len(missing_files)} files/directories")
        report.append(f"- **Extra locally:** {len(analysis['extra_locally'])} files/directories (not on remote)")
        report.append("")
        
        # Exclusions
        report.append("## EXCLUSIONS")
        report.append("")
        report.append("**The following items are EXCLUDED from the copy plan:**")
        report.append("- **Log files** (`logs/`, `runs/*.pid`) - Training logs and process files")
        report.append("- **Downloaded creative assets** (`assets/meta_creatives/`, `assets/veo_videos/`, `assets/veo_balance/`) - Downloaded images/videos")
        report.append("- **Manifest files** (`assets/*_manifest.csv`) - CSV catalogs of downloaded assets")
        report.append("")
        
        # Files to copy
        report.append("## Files/Directories to Copy")
        report.append("")
        
        priority_order = ['artifacts', 'pipelines', 'scripts', 'tools', 'requirements', 'ROOT']
        
        for i, category in enumerate(priority_order, 1):
            if category in categories:
                files = categories[category]
                report.append(f"### Priority {i}: {category.title()} ({len(files)} items)")
                report.append("")
                
                if category == 'artifacts':
                    report.append("The `artifacts/` directory contains:")
                    report.append("- **Creative features:** Meta creative features, Veo video features")
                    report.append("- **Feature engineering:** Marketing CTR features, enhanced features, catalogs")
                    report.append("- **Trained models:** CTR classifiers and rankers (~9MB total)")
                    report.append("- **Predictions:** CTR scores, current running ads scores")
                    report.append("- **Priors:** Thompson Sampling strategies")
                    report.append("- **Validation results:** Forward holdout and ranking evaluation metrics")
                elif category == 'pipelines':
                    report.append("The `pipelines/` directory contains:")
                    report.append("- **CTR pipeline:** Training and prediction scripts")
                    report.append("- **Validation:** Forward holdout and metrics")
                    report.append("- **Creative processing:** Feature extraction, manifest generation")
                    report.append("- **Data processing:** Meta/GA4 unification, synthetic data")
                    report.append("- **Additional utilities:** Bandit orchestration, feature engineering, etc.")
                elif category == 'scripts':
                    report.append("The `scripts/` directory contains utility scripts.")
                elif category == 'tools':
                    report.append("The `tools/` directory contains Meta API fetchers and creative utilities.")
                elif category == 'requirements':
                    report.append("GPU-specific requirements file.")
                elif category == 'ROOT':
                    report.append("Root-level utility scripts and YOLOv8 model weights.")
                
                report.append("")
        
        # Copy commands
        report.append("## Recommended Copy Commands")
        report.append("")
        report.append("Copy files individually using `gcloud compute scp`:")
        report.append("")
        report.append("```bash")
        report.append(self.generate_copy_commands(missing_files))
        report.append("```")
        report.append("")
        
        # File size estimates
        report.append("## File Size Estimates")
        report.append("")
        report.append("Based on sampling (excluding creative assets and logs):")
        report.append("- **artifacts/models/**: ~9 MB (3 trained models)")
        report.append("- **artifacts/features/**: ~1 MB (Parquet feature files)")
        report.append("- **artifacts/predictions/**: ~100 KB (prediction scores)")
        report.append("- **pipelines/**: ~100-500 KB (Python scripts)")
        report.append("- **scripts/**: ~50-100 KB (utility scripts)")
        report.append("- **tools/**: ~50-100 KB (API integration tools)")
        report.append("")
        report.append("**Total estimated download:** ~10-15 MB (much smaller without creative assets)")
        report.append("")
        
        # Verification
        report.append("## Verification After Copy")
        report.append("")
        report.append("After copying files, verify with:")
        report.append("")
        report.append("```bash")
        report.append("# Check artifacts were copied")
        report.append("ls -lh ./artifacts/models/")
        report.append("")
        report.append("# Check pipelines")
        report.append("ls -R ./pipelines/")
        report.append("")
        report.append("# Check scripts and tools")
        report.append("ls ./scripts/")
        report.append("ls ./tools/")
        report.append("")
        report.append("# Verify file counts")
        report.append("find ./artifacts -type f | wc -l  # Should be ~40")
        report.append("find ./pipelines -type f | wc -l  # Should be ~57")
        report.append("```")
        report.append("")
        
        return '\n'.join(report)
    
    def save_files(self, analysis):
        """Save analysis results to files."""
        # Save file lists
        with open('tmp/local_files.txt', 'w') as f:
            for file_path in sorted(self.local_files):
                f.write(f"{file_path}\n")
        
        with open('tmp/remote_files.txt', 'w') as f:
            for file_path in sorted(self.remote_files):
                f.write(f"{file_path}\n")
        
        # Save missing files list
        missing_files = analysis['missing_locally']
        with open('tmp/copy_list.txt', 'w') as f:
            f.write("FILES TO COPY:\n")
            f.write("=" * 80 + "\n")
            for file_path in sorted(missing_files):
                f.write(f"{file_path.lstrip('./')}\n")
        
        # Save report
        report = self.generate_report(analysis)
        with open('tmp/SYNC_REPORT.md', 'w') as f:
            f.write(report)
        
        print(f"Analysis complete!")
        print(f"- Report saved to: tmp/SYNC_REPORT.md")
        print(f"- File lists saved to: tmp/local_files.txt, tmp/remote_files.txt")
        print(f"- Copy list saved to: tmp/copy_list.txt")
    
    def run(self):
        """Run the complete sync analysis."""
        print("AELP Sync Analysis Tool")
        print("=" * 50)
        
        # Ensure tmp directory exists
        os.makedirs('tmp', exist_ok=True)
        
        # Get file listings
        self.get_local_files()
        self.get_remote_files()
        
        if not self.local_files or not self.remote_files:
            print("Error: Could not retrieve file listings")
            return
        
        # Analyze differences
        print("\nAnalyzing differences...")
        analysis = self.analyze_differences()
        
        print(f"\nAnalysis Results:")
        print(f"- Missing locally: {len(analysis['missing_locally'])}")
        print(f"- Extra locally: {len(analysis['extra_locally'])}")
        print(f"- Common files: {len(analysis['common_files'])}")
        
        # Save results
        self.save_files(analysis)
        
        # Show summary
        missing_files = analysis['missing_locally']
        if missing_files:
            print(f"\nTop missing directories:")
            categories = self.categorize_missing_files(missing_files)
            for category, files in sorted(categories.items()):
                if len(files) > 5:  # Only show directories with many files
                    print(f"  {category}/: {len(files)} items")

def main():
    analyzer = AELPSyncAnalyzer()
    analyzer.run()

if __name__ == '__main__':
    main()
