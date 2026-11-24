"""Step 2: how much does each block of covariates explain?

Estimates four nested logit models of DPP (versus KMT) identification among
partisans, adding one block of covariates at a time:

    1. Socioeconomic and demographic (age, sex, education, marital status,
       personal and household income, survey wave)
    2. + National self-identification (Taiwanese / Chinese / both)
    3. + Stance on independence versus unification
    4. + Vote in the 2012 presidential election

For each model it reports average marginal effects with heteroskedasticity-
robust standard errors, McFadden's pseudo R-squared, and the out-of-sample
AUC from stratified 5-fold cross-validation. The question is "identity or
class?", so the comparison of interest is how much fit each block adds.

Outputs: output/tables/covariate_blocks.{tex,md} and
         output/figures/roc_covariate_blocks.png

Run from the repository root:  python -m src.covariate_blocks
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from src.config import FIGURE_DIR, N_FOLDS, SAMPLE_PATH, SEED, TABLE_DIR, ensure_output_dirs

SES_TERMS = [
    "age_decades",
    "female",
    "C(education, Treatment('Senior high'))",
    "C(marital_status, Treatment('Married or cohabiting'))",
    "income_personal",
    "income_personal_missing",
    "income_household",
    "income_household_missing",
    "C(wave)",
]
IDENTITY_TERM = "C(identity, Treatment('Both'))"
INDEPENDENCE_TERM = "C(independence, Treatment('Status quo'))"
VOTE_TERM = "C(vote_2012, Treatment('Ma (KMT)'))"

BLOCKS = {
    "(1) Socioeconomic": SES_TERMS,
    "(2) + Identity": SES_TERMS + [IDENTITY_TERM],
    "(3) + Independence": SES_TERMS + [IDENTITY_TERM, INDEPENDENCE_TERM],
    "(4) + 2012 vote": SES_TERMS + [IDENTITY_TERM, INDEPENDENCE_TERM, VOTE_TERM],
}

# Rows shown in the table, in order: (design-matrix column, label).
# Marital status, wave dummies and income-missing flags are controls.
DISPLAY_ROWS = [
    ("age_decades", "Age (10 years)"),
    ("female", "Female"),
    ("C(education, Treatment('Senior high'))[T.Primary or less]", "Primary or less"),
    ("C(education, Treatment('Senior high'))[T.Junior high]", "Junior high"),
    ("C(education, Treatment('Senior high'))[T.Junior college]", "Junior college"),
    ("C(education, Treatment('Senior high'))[T.University]", "University"),
    ("C(education, Treatment('Senior high'))[T.Graduate]", "Graduate degree"),
    ("income_personal", "Personal income (NT\\$10k/month)"),
    ("income_household", "Household income (NT\\$10k/month)"),
    ("C(identity, Treatment('Both'))[T.Taiwanese]", "Identifies as Taiwanese"),
    ("C(identity, Treatment('Both'))[T.Chinese]", "Identifies as Chinese"),
    ("C(identity, Treatment('Both'))[T.Other or no answer]", "Identity: other or no answer"),
    ("C(independence, Treatment('Status quo'))[T.Independence]", "Favors independence"),
    ("C(independence, Treatment('Status quo'))[T.Unification]", "Favors unification"),
    ("C(independence, Treatment('Status quo'))[T.No answer]", "Independence: no answer"),
    ("C(vote_2012, Treatment('Ma (KMT)'))[T.Tsai (DPP)]", "Voted Tsai (DPP) in 2012"),
    ("C(vote_2012, Treatment('Ma (KMT)'))[T.Soong or blank]", "Voted Soong or blank"),
    ("C(vote_2012, Treatment('Ma (KMT)'))[T.Did not vote]", "Did not vote"),
    ("C(vote_2012, Treatment('Ma (KMT)'))[T.No answer]", "Vote: no answer"),
]
GROUP_HEADERS = {
    "Primary or less": "Education (vs. senior high)",
    "Identifies as Taiwanese": "National identity (vs. both)",
    "Favors independence": "Independence stance (vs. status quo)",
    "Voted Tsai (DPP) in 2012": "2012 presidential vote (vs. Ma)",
}


def load_partisans() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)
    partisans = sample[
        (sample["partisan"] == 1)
        & (sample["education"] != "Other")
        & (sample["marital_status"] != "Other")
    ].copy()
    partisans["age_decades"] = partisans["age"] / 10
    for column in ("income_personal", "income_household"):
        partisans[column] = partisans[column].fillna(0.0)  # flagged by *_missing
    return partisans.reset_index(drop=True)


def formula(terms: list[str]) -> str:
    return "dpp ~ " + " + ".join(terms)


def fit_block(data: pd.DataFrame, terms: list[str]):
    return smf.logit(formula(terms), data=data).fit(disp=False, cov_type="HC1")


def cross_validated_predictions(data: pd.DataFrame, terms: list[str]) -> np.ndarray:
    """Out-of-fold predicted probabilities from stratified K-fold CV."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    predictions = np.empty(len(data))
    for train_idx, test_idx in folds.split(data, data["dpp"]):
        model = smf.logit(formula(terms), data=data.iloc[train_idx]).fit(disp=False)
        predictions[test_idx] = model.predict(data.iloc[test_idx])
    return predictions


def fold_aucs(data: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return np.array(
        [
            roc_auc_score(data["dpp"].iloc[test_idx], predictions[test_idx])
            for _, test_idx in folds.split(data, data["dpp"])
        ]
    )


def stars(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""


def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every block; return AME estimates, fit statistics and OOF predictions."""
    effects, fit_rows, oof = {}, [], {}
    for name, terms in BLOCKS.items():
        result = fit_block(data, terms)
        margins = result.get_margeff(at="overall", dummy=True).summary_frame()
        effects[name] = margins.rename(columns={"dy/dx": "ame", "Std. Err.": "se", "Pr(>|z|)": "p"})
        oof[name] = cross_validated_predictions(data, terms)
        aucs = fold_aucs(data, oof[name])
        fit_rows.append(
            {
                "model": name,
                "n": int(result.nobs),
                "pseudo_r2": result.prsquared,
                "cv_auc": aucs.mean(),
                "cv_auc_sd": aucs.std(),
            }
        )
    return effects, pd.DataFrame(fit_rows), oof


def table_cells(effects: dict, fit: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Rows of (label, cells) shared by the LaTeX and Markdown writers."""
    rows: list[tuple[str, list[str]]] = []
    for column, label in DISPLAY_ROWS:
        if label in GROUP_HEADERS:
            rows.append((f"__header__{GROUP_HEADERS[label]}", []))
        estimates, errors = [], []
        for name in BLOCKS:
            frame = effects[name]
            if column in frame.index:
                row = frame.loc[column]
                estimates.append(f"{row['ame']:.3f}{stars(row['p'])}")
                errors.append(f"({row['se']:.3f})")
            else:
                estimates.append("")
                errors.append("")
        rows.append((label, estimates))
        rows.append(("", errors))
    rows.append(("__rule__", []))
    rows.append(("Marital status and wave controls", ["Yes"] * len(BLOCKS)))
    rows.append(("Observations", [f"{n:,}" for n in fit["n"]]))
    rows.append(("McFadden pseudo $R^2$", [f"{v:.3f}" for v in fit["pseudo_r2"]]))
    rows.append(
        (
            f"Out-of-sample AUC ({N_FOLDS}-fold CV)",
            [f"{m:.3f} ({s:.3f})" for m, s in zip(fit["cv_auc"], fit["cv_auc_sd"])],
        )
    )
    return rows


def write_latex(rows: list[tuple[str, list[str]]]) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Nested logit models of DPP versus KMT identification: average marginal effects}",
        "\\label{tab:covariate_blocks}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(BLOCKS) + "}",
        "\\toprule",
        " & " + " & ".join(BLOCKS) + " \\\\",
        "\\midrule",
    ]
    for label, cells in rows:
        if label.startswith("__header__"):
            lines.append(f"\\multicolumn{{{len(BLOCKS) + 1}}}{{l}}{{\\textit{{{label[10:]}}}}} \\\\")
        elif label == "__rule__":
            lines.append("\\midrule")
        else:
            cells = [re.sub(r"(\*+)$", r"$^{\1}$", cell) for cell in cells]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{minipage}{0.95\\linewidth}\\footnotesize",
        "\\vspace{4pt}Notes: Sample is respondents to the 2012--2014 Taiwan Social Change Survey who",
        "identify with or lean toward the DPP or KMT. The outcome is 1 for the DPP. Entries are average",
        "marginal effects (discrete changes for indicators) with heteroskedasticity-robust (HC1) standard",
        "errors in parentheses. Income missing indicators are included. AUC is the mean (standard",
        "deviation) across folds of out-of-fold predictions. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.",
        "\\end{minipage}",
        "\\end{table}",
    ]
    (TABLE_DIR / "covariate_blocks.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(rows: list[tuple[str, list[str]]]) -> None:
    lines = [
        "| | " + " | ".join(BLOCKS) + " |",
        "| --- |" + " ---: |" * len(BLOCKS),
    ]
    for label, cells in rows:
        if label.startswith("__header__"):
            lines.append(f"| *{label[10:]}* |" + " |" * len(BLOCKS))
        elif label == "__rule__":
            continue
        else:
            clean = label.replace("\\$", "$").replace("$R^2$", "R²")
            lines.append(f"| {clean} | " + " | ".join(cells) + " |")
    (TABLE_DIR / "covariate_blocks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_roc(data: pd.DataFrame, oof: dict, fit: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for (name, predictions), auc in zip(oof.items(), fit["cv_auc"]):
        fpr, tpr, _ = roc_curve(data["dpp"], predictions)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC curves, DPP vs. KMT identification")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_covariate_blocks.png", dpi=200)
    plt.close(fig)


def main() -> pd.DataFrame:
    ensure_output_dirs()
    data = load_partisans()
    effects, fit, oof = summarize(data)
    rows = table_cells(effects, fit)
    write_latex(rows)
    write_markdown(rows)
    plot_roc(data, oof, fit)
    print(fit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return fit


if __name__ == "__main__":
    main()
