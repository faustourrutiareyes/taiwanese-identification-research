"""Paths and constants shared by every step of the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
OUTPUT_DIR = ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
SAMPLE_PATH = OUTPUT_DIR / "data" / "analysis_sample.csv"

# Survey waves of the Taiwan Social Change Survey used in the analysis.
WAVES = (2012, 2013, 2014)

# Taiwan's ROC calendar: ROC year = Gregorian year - 1911.
ROC_OFFSET = 1911

SEED = 42
N_FOLDS = 5


def ensure_output_dirs() -> None:
    for path in (TABLE_DIR, FIGURE_DIR, LOG_DIR, SAMPLE_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
