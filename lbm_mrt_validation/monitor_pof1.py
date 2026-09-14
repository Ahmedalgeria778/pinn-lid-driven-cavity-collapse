# -*- coding: utf-8 -*-
"""
Monitor POF1 relaunch runs every MONITOR_INTERVAL seconds.
Appends one status line to pof1_converged/monitor.log each pass:
  UTC time | py_alive | rows_in_csv | completed cases | elapsed_min
"""
import os
import csv
import time
import datetime
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pof1_converged")
CSV_PATH = os.path.join(OUT, "pof1_results_converged.csv")
META_DIR = os.path.join(OUT, "meta")
LOG_PATH = os.path.join(OUT, "monitor.log")

CASES = [
    "uniform_Re500_N256", "sin_pi_Re500_N256", "sin_2pi_Re500_N256", "pinn_Re500_N256",
    "sin_2pi_Re500_N512",
    "uniform_Re1000_N256", "sin_pi_Re1000_N256", "sin_2pi_Re1000_N256", "pinn_Re1000_N256",
]

MONITOR_INTERVAL = 1200.0  # seconds


def count_python_running():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process -Name python -ErrorAction SilentlyContinue | Measure-Object).Count"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip() or 0)
    except Exception:
        return -1


def read_done():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted({f"{r['profile']}_Re{r['Re']}_N{r['N']}" for r in rows})


def main():
    start = time.time()
    with open(LOG_PATH, "a") as log:
        while True:
            npy = count_python_running()
            done = read_done()
            elapsed = (time.time() - start) / 60.0
            missing = [c for c in CASES if c not in done]
            line = ("{ts} | py={npy} | rows={nrows} | done={ndone}/9 | "
                    "elapsed={el:.0f}min | missing={missing}").format(
                ts=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                npy=npy, nrows=len(done), ndone=len(done), el=elapsed, missing=";".join(missing))
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            time.sleep(MONITOR_INTERVAL)


if __name__ == "__main__":
    main()