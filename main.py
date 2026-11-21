"""
Run the whole pipeline in order: step01 → step12.

Usage:
    python main.py        # run every step
    python main.py 6      # resume from step06 (earlier outputs must already exist)
"""

import os
import subprocess
import sys
from glob import glob

for folder in ["data", "figures", "latents", "logs", "models", "weights"]:
    os.makedirs(folder, exist_ok=True)

start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
steps = sorted(glob("step*.py"))

for script in steps:
    if int(script[4:6]) < start:
        continue
    print(f"\n{'=' * 20} {script} {'=' * 20}", flush=True)
    subprocess.run([sys.executable, script], check=True)   # stops on the first failing step

print("\nAll steps finished.")
