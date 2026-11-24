"""Reproduce every table and figure from the raw survey extracts.

uv run python run_all.py          # full run, a few minutes on a laptop
uv run python run_all.py --fast   # quick check with fewer tuning iterations
"""

import argparse

from src import build_sample, codebook, covariate_blocks, descriptives, ml_models, robustness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full replication pipeline.")
    parser.add_argument(
        "--fast", action="store_true", help="fewer hyperparameter search iterations"
    )
    fast = parser.parse_args().fast

    print("== Step 1: build the analysis sample and codebook ==")
    build_sample.main()
    codebook.main()
    print("\n== Step 2: descriptive statistics ==")
    descriptives.main()
    print("\n== Step 3: nested covariate-block logits ==")
    covariate_blocks.main()
    print("\n== Step 4: robustness by wave and without leaners ==")
    robustness.main()
    print("\n== Step 5: model comparison and variable importance ==")
    ml_models.main(fast=fast)
    print("\nDone. Tables are in output/tables/, figures in output/figures/.")


if __name__ == "__main__":
    main()
