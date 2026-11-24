"""Reproduce every table and figure from the raw survey extracts.

    uv run python run_all.py          # full run, a few minutes on a laptop
    uv run python run_all.py --fast   # quick check with fewer tuning iterations
"""

import argparse

from src import build_sample, covariate_blocks, ml_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full replication pipeline.")
    parser.add_argument("--fast", action="store_true", help="fewer hyperparameter search iterations")
    fast = parser.parse_args().fast

    print("== Step 1: build the analysis sample ==")
    build_sample.main()
    print("\n== Step 2: nested covariate-block logits ==")
    covariate_blocks.main()
    print("\n== Step 3: model comparison and variable importance ==")
    ml_models.main(fast=fast)
    print("\nDone. Tables are in output/tables/, figures in output/figures/.")


if __name__ == "__main__":
    main()
