# Identity versus Class in the Taiwanese Electorate

[![Tests and reproducibility](https://github.com/faustourrutiareyes/taiwanese-identification-research/actions/workflows/tests.yml/badge.svg)](https://github.com/faustourrutiareyes/taiwanese-identification-research/actions/workflows/tests.yml)

Does national self-identification or socioeconomic position better explain which of Taiwan's two main parties a voter supports? Using three waves (2012 to 2014) of the Taiwan Social Change Survey, this project compares nested models of DPP versus KMT identification among 1,705 partisans.

**Main result.** Socioeconomic and demographic variables barely separate DPP from KMT supporters (out-of-sample AUC 0.60). Adding national identity raises the AUC to 0.75, and adding the respondent's stance on independence raises it to 0.81. Identifying as "Taiwanese" rather than "both Taiwanese and Chinese" is associated with a 42 percentage point higher probability of DPP identification, holding class and demographics fixed. The identity gap holds in each survey wave separately (22 to 38 points in model 3) and when party leaners are excluded. Past presidential vote then pushes the AUC to 0.97, and flexible machine learning models (random forest, gradient boosting, XGBoost, CatBoost) do no better than a logit with the same covariates.

| Covariates | McFadden pseudo R² | Out-of-sample AUC (5-fold CV) |
| --- | ---: | ---: |
| (1) Socioeconomic and demographic | 0.035 | 0.604 |
| (2) + National identity | 0.177 | 0.752 |
| (3) + Independence stance | 0.241 | 0.808 |
| (4) + 2012 presidential vote | 0.713 | 0.970 |

![Out-of-fold ROC curves by covariate block](output/figures/roc_covariate_blocks.png)

Full tables: [descriptive statistics](output/tables/descriptives.md), [nested logits with marginal effects](output/tables/covariate_blocks.md), [robustness by wave](output/tables/robustness.md), [model comparison](output/tables/model_comparison.md). LaTeX versions sit next to each.

## Reproducing the results

The code needs Python 3.14 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync                          # create the environment from uv.lock
uv run python run_all.py         # rebuild every table, figure and the codebook (about a minute)
uv run python run_all.py --fast  # quicker check with fewer tuning iterations
uv run pytest                    # tests for the recoding rules and the sample
```

Each step can also be run on its own, for example `uv run python -m src.covariate_blocks`. All randomness (CV folds, train/test split, hyperparameter search) is seeded in `src/config.py`. On every push, GitHub Actions runs the linter and tests, rebuilds the regression tables and the codebook from the raw data, and fails if they differ from the committed versions.

## Repository layout

| Path | What it does |
| --- | --- |
| `src/recode.py` | Every coding decision, written as ordered (pattern, label) rules that map the Chinese answer labels of each wave to common English categories |
| `src/build_sample.py` | Step 1: loads the three waves, harmonizes them and writes the pooled analysis sample plus a sample construction log |
| `src/codebook.py` | Step 1: writes [`docs/codebook.md`](docs/codebook.md), every raw answer with its recoded category and counts by wave |
| `src/descriptives.py` | Step 2: characteristics of DPP and KMT partisans with difference-in-means tests |
| `src/covariate_blocks.py` | Step 3: four nested logits with average marginal effects, robust standard errors and cross-validated AUC; writes the main table and the ROC figure |
| `src/robustness.py` | Step 4: model (3) re-estimated for each wave and without party leaners |
| `src/ml_models.py` | Step 5: logit versus tuned tree ensembles on a held-out test set, plus grouped permutation importance |
| `src/config.py` | Paths, seeds and constants |
| `run_all.py` | Runs every step in order |
| `tests/` | Unit tests for the recoding rules and checks on the constructed sample |
| `Data/` | Survey extracts for 2012, 2013 and 2014 (converted from Stata to CSV) |
| `output/tables/` | Markdown and LaTeX tables: `descriptives`, `covariate_blocks` (main table), `robustness`, `model_comparison`, plus `permutation_importance.csv` |
| `output/figures/` | `roc_covariate_blocks.png`, `permutation_importance.png` |
| `output/logs/` | `sample_construction.md`: respondents kept at each step, by wave |
| `notebooks/` | The original exploratory notebook and a PDF of it, kept for the record (its rendered version is [online](https://faustourrutiareyes.github.io/taiwanese-identification-research/) via `index.html`) |

## Data

The data come from the Taiwan Social Change Survey, run by the Center for Survey Research at Academia Sinica. The 2012, 2013 and 2014 waves are used because they all ask respondents whether they consider themselves Taiwanese, Chinese, or both. The original Stata files were converted to CSV by a classmate; everything else in this repository is my own work.

## Coding decisions worth knowing

The [codebook](docs/codebook.md) lists every raw answer and the category it was coded to. The decisions that matter most:

- **Outcome.** Respondents who identify with or lean toward the DPP (民進黨, 泛綠) are coded 1 and those who identify with or lean toward the KMT (國民黨, 泛藍) are coded 0. Everyone else (no party, small parties, refusals) is outside the estimation sample. The 2012 questionnaire has no "lean" option, so all models include wave fixed effects, and a robustness check drops leaners.
- **Independence stance.** The six-point independence/unification scale is collapsed to direction: independence (now or eventually), unification (now or eventually), and status quo (indefinitely, or "decide later", an option only 2014 offers).
- **2012 vote non-answers.** "Forgot whether I voted" and "refused to say whether I voted" contain the Chinese words for "did not vote", so they are matched first and coded as no answer.
- **2008 presidential vote is not used.** In 2014 the variable that holds the 2008 vote in other waves records whether the respondent took a referendum ballot instead, so it cannot be compared across waves. The 2012 vote is asked consistently and is used instead.
- **Religion is not used.** It was not asked in 2013 (99.8% missing).
- **Income.** Bracketed monthly income is converted to bracket midpoints in NT$10,000, with an indicator for refusals and "don't know" (19% of partisans for household income).
- **Weights.** Estimates are unweighted.

## Interpretation and limitations

The models describe and predict; they do not identify causal effects. Past vote is best read as a summary of earlier partisanship (and is itself shaped by identity), so model (4) shows how much of identity's association with party runs through prior behavior rather than an independent effect of voting. The sample covers 2012 to 2014 only, so it says nothing about later shifts such as the 2016 DPP victory.

## Differences from the original notebook

`src/` is a cleaned, scripted version of `notebooks/exploratory_analysis.ipynb`. Rebuilding it fixed several issues, so its numbers differ from the notebook's:

- The notebook's independence recode mapped "status quo, then independence" and "status quo, then unification" to "status quo"; they are now coded by direction.
- The notebook's gender variable became missing for 2012 because the category list used different labels; it is now coded for all waves.
- The notebook used the 2014 referendum-ballot question as if it were the 2008 vote, and included township dummies; both are dropped.
- The notebook coded "forgot whether I voted" as not voting; it is now coded as no answer.
- Model comparisons now use a stratified split, AUC alongside accuracy, and tuning on the training set only. Held-out accuracy is about 92% for both the logit and the tree ensembles (the notebook reported 94.7% for XGBoost).
