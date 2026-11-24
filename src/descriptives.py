"""Summary statistics: how do DPP and KMT supporters differ?

Reports the mean of each covariate (shares for categories) among DPP and
KMT partisans, the difference, and a two-sided Welch t-test p-value.

Outputs: output/tables/descriptives.{tex,md}

Run from the repository root:  python -m src.descriptives
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import ttest_ind

from src.config import TABLE_DIR, ensure_output_dirs
from src.covariate_blocks import load_partisans

CATEGORIES = {
    "education": [
        "Primary or less",
        "Junior high",
        "Senior high",
        "Junior college",
        "University",
        "Graduate",
    ],
    "identity": ["Taiwanese", "Both", "Chinese", "Other or no answer"],
    "independence": ["Independence", "Status quo", "Unification", "No answer"],
    "vote_2012": ["Tsai (DPP)", "Ma (KMT)", "Soong or blank", "Did not vote", "No answer"],
}
SECTION_TITLES = {
    "education": "Education (share)",
    "identity": "National identity (share)",
    "independence": "Independence stance (share)",
    "vote_2012": "2012 presidential vote (share)",
}


def indicator_columns(data: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """Return (section, label, values) for every row of the table."""
    rows: list[tuple[str, str, pd.Series]] = [
        ("", "Age (years)", data["age"]),
        ("", "Female (share)", data["female"]),
        (
            "",
            "Personal income (NT\\$10k/month)",
            data["income_personal"].where(data["income_personal_missing"] == 0),
        ),
        (
            "",
            "Household income (NT\\$10k/month)",
            data["income_household"].where(data["income_household_missing"] == 0),
        ),
    ]
    for variable, levels in CATEGORIES.items():
        for level in levels:
            rows.append((SECTION_TITLES[variable], level, (data[variable] == level).astype(float)))
    return rows


def build_table(data: pd.DataFrame) -> pd.DataFrame:
    dpp = data["dpp"] == 1
    records = []
    for section, label, values in indicator_columns(data):
        a, b = values[dpp].dropna(), values[~dpp].dropna()
        records.append(
            {
                "section": section,
                "variable": label,
                "dpp": a.mean(),
                "kmt": b.mean(),
                "difference": a.mean() - b.mean(),
                "p_value": ttest_ind(a, b, equal_var=False).pvalue,
            }
        )
    table = pd.DataFrame(records)
    table.attrs["n_dpp"], table.attrs["n_kmt"] = int(dpp.sum()), int((~dpp).sum())
    return table


def fmt(value: float) -> str:
    return f"{value:.2f}" if abs(value) >= 10 else f"{value:.3f}"


def write_tables(table: pd.DataFrame) -> None:
    n_dpp, n_kmt = table.attrs["n_dpp"], table.attrs["n_kmt"]
    md = [
        f"| | DPP (n={n_dpp:,}) | KMT (n={n_kmt:,}) | Difference | p-value |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Characteristics of DPP and KMT partisans, TSCS 2012--2014}",
        "\\label{tab:descriptives}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f" & DPP & KMT & Difference & $p$-value \\\\ & ($n={n_dpp:,}$) & ($n={n_kmt:,}$) & & \\\\",
        "\\midrule",
    ]
    current = None
    for row in table.itertuples():
        if row.section and row.section != current:
            current = row.section
            md.append(f"| *{current}* | | | | |")
            tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{current}}}}} \\\\")
        indent = "\\quad " if row.section else ""
        cells = [fmt(row.dpp), fmt(row.kmt), fmt(row.difference), f"{row.p_value:.3f}"]
        md.append(f"| {row.variable.replace(chr(92), '')} | " + " | ".join(cells) + " |")
        tex.append(f"{indent}{row.variable} & " + " & ".join(cells) + " \\\\")
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{minipage}{0.9\\linewidth}\\footnotesize",
        "\\vspace{4pt}Notes: Means, or shares for categories, among respondents who identify with or lean",
        "toward each party. Income means exclude refusals and ``don't know''. $p$-values are from",
        "two-sided Welch $t$-tests of equal means.",
        "\\end{minipage}",
        "\\end{table}",
    ]
    (TABLE_DIR / "descriptives.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABLE_DIR / "descriptives.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> pd.DataFrame:
    ensure_output_dirs()
    table = build_table(load_partisans())
    write_tables(table)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


if __name__ == "__main__":
    main()
