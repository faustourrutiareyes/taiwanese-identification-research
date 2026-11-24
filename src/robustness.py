"""Robustness: does the identity result hold wave by wave and without leaners?

Re-estimates model (3) of covariate_blocks.py (socioeconomic controls,
national identity and independence stance) separately for each survey wave,
and on the pooled sample without "leaners" (respondents who only lean toward
a party, an option offered in 2013 and 2014 but not 2012). For each sample
it reports the main average marginal effects and the cross-validated AUC of
the socioeconomic-only model and of model (3).

Outputs: output/tables/robustness.{tex,md}

Run from the repository root:  python -m src.robustness
"""

from __future__ import annotations

import re

import pandas as pd

from src.config import N_FOLDS, TABLE_DIR, ensure_output_dirs
from src.covariate_blocks import (
    BLOCKS,
    cross_validated_predictions,
    fit_block,
    fold_aucs,
    load_partisans,
    stars,
)

SES_ONLY = BLOCKS["(1) Socioeconomic"]
MODEL_3 = BLOCKS["(3) + Independence"]

ROWS = [
    ("C(identity, Treatment('Both'))[T.Taiwanese]", "Identifies as Taiwanese (vs. both)"),
    ("C(identity, Treatment('Both'))[T.Chinese]", "Identifies as Chinese (vs. both)"),
    (
        "C(independence, Treatment('Status quo'))[T.Independence]",
        "Favors independence (vs. status quo)",
    ),
    (
        "C(independence, Treatment('Status quo'))[T.Unification]",
        "Favors unification (vs. status quo)",
    ),
]


def samples(data: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, bool]]:
    """Name -> (sample, whether wave dummies apply)."""
    out = {"Pooled (baseline)": (data, True)}
    for wave in sorted(data["wave"].unique()):
        out[str(wave)] = (data[data["wave"] == wave].reset_index(drop=True), False)
    out["Excluding leaners"] = (data[data["leaner"] == 0].reset_index(drop=True), True)
    return out


CATEGORICAL = ["identity", "independence", "education", "marital_status"]


def drop_perfect_predictors(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop categories in which every respondent backs the same party.

    Such a category predicts the outcome perfectly, so its coefficient is not
    identified and the logit fails to converge. As Stata's logit does, the
    affected respondents are dropped and the omission is reported.
    """
    notes = []
    for column in CATEGORICAL:
        for level, group in sample.groupby(column):
            if group["dpp"].nunique() == 1:
                party = "DPP" if group["dpp"].iat[0] == 1 else "KMT"
                notes.append(f"{column} = {level} (all {len(group)} {party})")
                sample = sample[sample[column] != level]
    return sample.reset_index(drop=True), notes


def without_wave(terms: list[str]) -> list[str]:
    return [term for term in terms if term != "C(wave)"]


def cv_auc(data: pd.DataFrame, terms: list[str]) -> tuple[float, float]:
    aucs = fold_aucs(data, cross_validated_predictions(data, terms))
    return aucs.mean(), aucs.std()


def estimate(data: pd.DataFrame) -> pd.DataFrame:
    columns, dropped = {}, {}
    for name, (sample, pooled) in samples(data).items():
        sample, notes = drop_perfect_predictors(sample)
        if notes:
            dropped[name] = notes
        ses = SES_ONLY if pooled else without_wave(SES_ONLY)
        full = MODEL_3 if pooled else without_wave(MODEL_3)
        margins = fit_block(sample, full).get_margeff(at="overall", dummy=True).summary_frame()
        column = {}
        for key, label in ROWS:
            if key not in margins.index:  # category dropped as a perfect predictor
                column[label], column[f"{label} se"] = "n.i.", ""
                continue
            row = margins.loc[key]
            column[label] = f"{row['dy/dx']:.3f}{stars(row['Pr(>|z|)'])}"
            column[f"{label} se"] = f"({row['Std. Err.']:.3f})"
        ses_auc, full_auc = cv_auc(sample, ses), cv_auc(sample, full)
        column["AUC, socioeconomic only"] = f"{ses_auc[0]:.3f} ({ses_auc[1]:.3f})"
        column["AUC, model (3)"] = f"{full_auc[0]:.3f} ({full_auc[1]:.3f})"
        column["Observations"] = f"{len(sample):,}"
        columns[name] = column
    table = pd.DataFrame(columns)
    table.attrs["dropped"] = dropped
    return table


def write_tables(table: pd.DataFrame) -> None:
    names = list(table.columns)
    md = ["| | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Robustness of model (3): by survey wave and excluding party leaners}",
        "\\label{tab:robustness}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(names) + "}",
        "\\toprule",
        " & " + " & ".join(names) + " \\\\",
        "\\midrule",
    ]
    for label in table.index:
        shown = "" if label.endswith(" se") else label
        cells = list(table.loc[label])
        if label == "AUC, socioeconomic only":
            tex.append("\\midrule")
            tex.append(
                f"\\multicolumn{{{len(names) + 1}}}{{l}}{{\\textit{{Out-of-sample AUC ({N_FOLDS}-fold CV)}}}} \\\\"
            )
        md.append(f"| {shown} | " + " | ".join(cells) + " |")
        tex_cells = [re.sub(r"(\*+)$", r"$^{\1}$", cell) for cell in cells]
        tex.append(f"{shown} & " + " & ".join(tex_cells) + " \\\\")
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{minipage}{0.95\\linewidth}\\footnotesize",
        "\\vspace{4pt}Notes: Average marginal effects on the probability of DPP (versus KMT)",
        "identification from model (3), with HC1 standard errors in parentheses. All columns control for",
        "age, sex, education, marital status and income; pooled columns add wave fixed effects.",
        "Leaners are respondents who chose ``leans blue'' or ``leans green'', offered only in 2013 and",
        "2014. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.",
        *[
            f"n.i.: not identified. Dropped as perfect predictors in {name}: {'; '.join(notes)}."
            for name, notes in table.attrs["dropped"].items()
        ],
        "\\end{minipage}",
        "\\end{table}",
    ]
    md += [
        "",
        *[
            f"n.i.: not identified. Dropped as perfect predictors in {name}: {'; '.join(notes)}."
            for name, notes in table.attrs["dropped"].items()
        ],
    ]
    (TABLE_DIR / "robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABLE_DIR / "robustness.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> pd.DataFrame:
    ensure_output_dirs()
    table = estimate(load_partisans())
    write_tables(table)
    print(table.to_string())
    return table


if __name__ == "__main__":
    main()
