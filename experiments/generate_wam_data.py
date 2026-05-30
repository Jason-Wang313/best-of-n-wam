from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.multi_env_suite import generate_all_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    summary = generate_all_datasets(args.states, args.rollouts, args.seed)
    print(f"generated WAM datasets for {len(summary['datasets'])} env/split entries")


if __name__ == "__main__":
    main()
