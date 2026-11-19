"""Run Yahoo stock research only. Options collection is retired.

Legacy worker entrypoint retained for existing Railway configurations.
Each collector retains its own idempotency, readiness and last-good protections.
"""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = 0
    scripts = ["stock_research_refresh.py"]
    for script in scripts:
        print("Starting " + script, flush=True)
        code = subprocess.run([sys.executable, str(ROOT / "jobs" / script)], cwd=ROOT).returncode
        print(script + " exited " + str(code), flush=True)
        if code:
            result = max(result, abs(code))
    return result


if __name__ == "__main__":
    sys.exit(main())
