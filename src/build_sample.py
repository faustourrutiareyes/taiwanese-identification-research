"""Step 1: build the pooled 2012 to 2014 analysis sample.

Reads the three Taiwan Social Change Survey extracts in Data/, harmonizes the
answer labels with the rules in recode.py, and writes one tidy file with
English category names to output/data/analysis_sample.csv. A log of how many
respondents each step keeps is written to output/logs/sample_construction.md.

Run from the repository root:  python -m src.build_sample
"""

from __future__ import annotations

import pandas as pd

from src import recode
from src.config import (
    DATA_DIR,
    LOG_DIR,
    ROC_OFFSET,
    SAMPLE_PATH,
    WAVES,
    ensure_output_dirs,
)


def load_wave(year: int) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / f"{year}_cleaned.csv", dtype=str)
    frame["wave"] = year
    return frame


def harmonize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map one wave's raw answers to the common English coding."""
    out = pd.DataFrame(index=raw.index)
    out["respondent_id"] = raw["id"]
    out["wave"] = raw["wave"]

    # Birth year is recorded in the ROC calendar (e.g. 75 = 1986).
    birth_year_roc = pd.to_numeric(raw["birth_year"], errors="coerce")
    out["age"] = raw["wave"] - (birth_year_roc + ROC_OFFSET)
    out["female"] = raw["gender"].str.startswith("女").astype(int)

    out["party"] = recode.apply_rules(raw["political_party"], recode.PARTY_RULES, "Other or none")
    out["identity"] = recode.apply_rules(
        raw["taiwanese_or_chinese"], recode.IDENTITY_RULES, "Other or no answer"
    )
    out["independence"] = recode.apply_rules(
        raw["taiwan_indep"], recode.INDEPENDENCE_RULES, "No answer"
    )
    out["vote_2012"] = recode.apply_rules(raw["pre_ele_101"], recode.VOTE_2012_RULES, "No answer")
    out["vote_2008"] = recode.apply_rules(raw["pre_ele_97"], recode.VOTE_2008_RULES, "No answer")
    if raw["wave"].iat[0] == 2014:
        # 2014's pre_ele_97 asks about a referendum ballot, not the 2008 vote.
        out["vote_2008"] = "Not asked"

    out["education"] = recode.apply_rules(raw["edu"], recode.EDUCATION_RULES, "Other")
    out["marital_status"] = recode.apply_rules(raw["marriage"], recode.MARRIAGE_RULES, "Other")
    out["income_personal"] = raw["month_income"].map(recode.income_midpoint)
    out["income_household"] = raw["mothe_house_income"].map(recode.income_midpoint)
    return out


def build_sample() -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Return the pooled sample and a log of (step, respondents kept)."""
    pooled = pd.concat([harmonize(load_wave(year)) for year in WAVES], ignore_index=True)
    log = [("All respondents, 2012 to 2014 waves", len(pooled))]

    pooled = pooled[pooled["age"].notna()].copy()
    log.append(("With a valid birth year", len(pooled)))

    pooled["age"] = pooled["age"].astype(int)
    pooled["dpp"] = (pooled["party"] == "DPP").astype(int)
    pooled["partisan"] = pooled["party"].isin(["DPP", "KMT"]).astype(int)
    log.append(("Identify with or lean toward the DPP or KMT", int(pooled["partisan"].sum())))

    # Income is often refused; keep those respondents and flag the gap.
    for column in ("income_personal", "income_household"):
        pooled[f"{column}_missing"] = pooled[column].isna().astype(int)
    return pooled, log


def write_log(sample: pd.DataFrame, log: list[tuple[str, int]]) -> None:
    partisans = sample[sample["partisan"] == 1]
    lines = [
        "# Sample construction",
        "",
        "| Step | Respondents |",
        "| --- | ---: |",
        *[f"| {step} | {count:,} |" for step, count in log],
        "",
        "## Partisan sample by wave",
        "",
        "| Wave | DPP | KMT | Share DPP |",
        "| --- | ---: | ---: | ---: |",
    ]
    for wave, group in partisans.groupby("wave"):
        dpp = int(group["dpp"].sum())
        kmt = len(group) - dpp
        lines.append(f"| {wave} | {dpp:,} | {kmt:,} | {dpp / len(group):.1%} |")
    lines += [
        "",
        "## Missing income",
        "",
        f"- Personal income missing (refused or unknown): "
        f"{partisans['income_personal_missing'].mean():.1%} of partisans",
        f"- Household income missing (refused or unknown): "
        f"{partisans['income_household_missing'].mean():.1%} of partisans",
        "",
    ]
    (LOG_DIR / "sample_construction.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> pd.DataFrame:
    ensure_output_dirs()
    sample, log = build_sample()
    sample.to_csv(SAMPLE_PATH, index=False)
    write_log(sample, log)
    for step, count in log:
        print(f"{step}: {count:,}")
    print(f"Wrote {SAMPLE_PATH.relative_to(SAMPLE_PATH.parents[2])}")
    return sample


if __name__ == "__main__":
    main()
