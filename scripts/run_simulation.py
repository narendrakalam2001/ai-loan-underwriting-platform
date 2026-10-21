# ============================================================
# RUN SIMULATION — runner script
# ============================================================
# Usage:
#   python scripts/run_simulation.py
#   python scripts/run_simulation.py --n 30 --scenario risky
#   python scripts/run_simulation.py --scenario all
# ============================================================

import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from simulation.data_simulator import simulate_applications, simulate_all_scenarios

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate loan applications against the live API")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--scenario", choices=["safe", "risky", "random", "all"], default="random")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    if args.scenario == "all":
        simulate_all_scenarios(n_per_scenario=max(1, args.n // 3))
    else:
        simulate_applications(n=args.n, scenario=args.scenario, delay=args.delay)
