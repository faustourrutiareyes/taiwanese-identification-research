"""Step 3: how well can flexible models predict party identification?

Uses the full covariate set of model (4) in covariate_blocks.py and compares
a logit with tree-based learners on one stratified 80/20 train/test split:

    * a single decision tree, unpruned and pruned by cost-complexity with
      5-fold CV (one-standard-error rule)
    * random forest, gradient boosting, XGBoost and CatBoost, each tuned by
      randomized search with 5-fold CV on the training set only

The test set is touched once, for the final scores. Variable importance is
measured by grouped permutation on the test set: all dummy columns that
come from one survey question are shuffled together, so a question with
many categories is not split across many small importances.

Outputs: output/tables/model_comparison.{csv,tex,md} and
         output/figures/permutation_importance.png

Run from the repository root:  python -m src.ml_models   (add --fast for a
quick run with fewer search iterations)
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import FIGURE_DIR, N_FOLDS, SEED, TABLE_DIR, ensure_output_dirs
from src.covariate_blocks import load_partisans

# Survey questions used as predictors, grouped as in covariate_blocks.py.
CATEGORICAL = ["education", "marital_status", "identity", "independence", "vote_2012", "wave"]
NUMERIC = [
    "age",
    "female",
    "income_personal",
    "income_personal_missing",
    "income_household",
    "income_household_missing",
]
VARIABLE_LABELS = {
    "age": "Age",
    "female": "Sex",
    "education": "Education",
    "marital_status": "Marital status",
    "income_personal": "Personal income",
    "income_household": "Household income",
    "identity": "National identity",
    "independence": "Independence stance",
    "vote_2012": "2012 presidential vote",
    "wave": "Survey wave",
}


def design_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode predictors; map each survey question to its columns."""
    X = pd.get_dummies(data[NUMERIC + CATEGORICAL].astype({"wave": str}), columns=CATEGORICAL)
    X = X.astype(float)
    groups: dict[str, list[str]] = {}
    for variable in VARIABLE_LABELS:
        groups[variable] = [
            column
            for column in X.columns
            if column == variable or column.startswith(f"{variable}_")
        ]
    return X, groups


def search(estimator, distributions: dict, n_iter: int, cv) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator,
        distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=SEED,
        n_jobs=-1,
    )


def pruned_tree(X_train: pd.DataFrame, y_train: pd.Series, cv) -> DecisionTreeClassifier:
    """Cost-complexity pruning, choosing the simplest tree within 1 SE of the best."""
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X_train, y_train)
    alphas = np.unique(path.ccp_alphas)[:-1]  # the largest alpha leaves only the root
    scores = []
    for alpha in alphas:
        tree = DecisionTreeClassifier(random_state=SEED, ccp_alpha=float(alpha))
        fold_scores = cross_val_score(tree, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        scores.append((alpha, fold_scores.mean(), fold_scores.std() / np.sqrt(len(fold_scores))))
    scores = pd.DataFrame(scores, columns=["alpha", "mean", "se"])
    best = scores.loc[scores["mean"].idxmax()]
    eligible = scores[scores["mean"] >= best["mean"] - best["se"]]
    alpha = float(eligible["alpha"].max())
    return DecisionTreeClassifier(random_state=SEED, ccp_alpha=alpha)


def candidate_models(n_iter: int, cv, X_train, y_train) -> dict:
    return {
        "Logit (model 4 covariates)": LogisticRegression(C=1e6, max_iter=10_000),
        "Decision tree (unpruned)": DecisionTreeClassifier(random_state=SEED),
        "Decision tree (pruned, CV)": pruned_tree(X_train, y_train, cv),
        "Random forest": search(
            RandomForestClassifier(random_state=SEED),
            {
                "n_estimators": [200, 400, 700],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", None],
            },
            n_iter,
            cv,
        ),
        "Gradient boosting": search(
            GradientBoostingClassifier(random_state=SEED),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": loguniform(0.01, 0.3),
                "max_depth": [1, 2, 3, 4],
                "subsample": [0.6, 0.8, 1.0],
            },
            n_iter,
            cv,
        ),
        "XGBoost": search(
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
            {
                "n_estimators": [100, 200, 400],
                "learning_rate": loguniform(0.01, 0.3),
                "max_depth": [2, 3, 4, 6],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
            },
            n_iter,
            cv,
        ),
        "CatBoost": search(
            CatBoostClassifier(random_state=SEED, verbose=0, allow_writing_files=False, thread_count=1),
            {
                "iterations": [200, 400],
                "learning_rate": loguniform(0.01, 0.3),
                "depth": [3, 4, 6],
                "l2_leaf_reg": [1, 3, 5, 7],
            },
            n_iter,
            cv,
        ),
    }


def grouped_permutation_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series, groups: dict, repeats: int = 30
) -> pd.DataFrame:
    """Mean drop in test AUC when all columns of one question are shuffled together."""
    rng = np.random.default_rng(SEED)
    baseline = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    rows = []
    for variable, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X_test.copy()
            order = rng.permutation(len(X_test))
            shuffled[columns] = X_test[columns].to_numpy()[order]
            drops.append(baseline - roc_auc_score(y_test, model.predict_proba(shuffled)[:, 1]))
        rows.append({"variable": VARIABLE_LABELS[variable], "auc_drop": np.mean(drops), "sd": np.std(drops)})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


def write_tables(results: pd.DataFrame) -> None:
    results.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    md = ["| Model | Test accuracy | Test F1 (DPP) | Test AUC |", "| --- | ---: | ---: | ---: |"]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Out-of-sample performance on the held-out 20\\% test set}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & Accuracy & F1 (DPP) & AUC \\\\",
        "\\midrule",
    ]
    for row in results.itertuples():
        md.append(f"| {row.model} | {row.accuracy:.3f} | {row.f1_dpp:.3f} | {row.auc:.3f} |")
        tex.append(f"{row.model} & {row.accuracy:.3f} & {row.f1_dpp:.3f} & {row.auc:.3f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABLE_DIR / "model_comparison.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def plot_importance(importance: pd.DataFrame, model_name: str) -> None:
    ordered = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(ordered["variable"], ordered["auc_drop"], xerr=ordered["sd"], color="#3b6ea5")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Drop in test AUC when the variable is shuffled")
    ax.set_title(f"Grouped permutation importance, {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permutation_importance.png", dpi=200)
    plt.close(fig)


def main(fast: bool = False) -> pd.DataFrame:
    ensure_output_dirs()
    data = load_partisans()
    X, groups = design_matrix(data)
    y = data["dpp"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, model in candidate_models(5 if fast else 30, cv, X_train, y_train).items():
        model = clone(model).fit(X_train, y_train)
        best = getattr(model, "best_estimator_", model)
        probability = best.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "f1_dpp": f1_score(y_test, prediction),
                "auc": roc_auc_score(y_test, probability),
            }
        )
        fitted[name] = best
        print(f"{name:<30} accuracy {rows[-1]['accuracy']:.3f}  AUC {rows[-1]['auc']:.3f}")

    results = pd.DataFrame(rows)
    write_tables(results)

    best_name = results.loc[results["auc"].idxmax(), "model"]
    importance = grouped_permutation_importance(fitted[best_name], X_test, y_test, groups)
    importance.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)
    plot_importance(importance, best_name)
    print(f"\nGrouped permutation importance ({best_name}):")
    print(importance.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="fewer search iterations")
    main(fast=parser.parse_args().fast)
