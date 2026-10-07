"""Run options then stock research serially on the existing Theta cron worker.

Separate processes release vendor sessions and memory between collectors.
Each collector retains its own idempotency, readiness and last-good protections.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = 0
    for script in ("options_flow_refresh.py", "theta_stock_refresh.py"):
        print("Starting " + script, flush=True)
        code = subprocess.run([sys.executable, str(ROOT / "jobs" / script)], cwd=ROOT).returncode
        print(script + " exited " + str(code), flush=True)
        if code:
            result = max(result, abs(code))
    return result


if __name__ == "__main__":
    sys.exit(main())
