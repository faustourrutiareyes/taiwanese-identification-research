"""Checks on the recoding rules and the constructed sample."""

import math

import pandas as pd
import pytest

from src import recode
from src.build_sample import build_sample


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("國民黨(跳答h7)", "KMT"),
        ("泛藍", "KMT"),
        ("民進黨", "DPP"),
        ("泛綠", "DPP"),
        ("親民黨", "Other or none"),
        ("都差不多", "Other or none"),
    ],
)
def test_party_rules(raw, expected):
    assert recode.apply_rules(pd.Series([raw]), recode.PARTY_RULES, "Other or none")[0] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("維持現狀,看情形再決定獨立或統一", "Status quo"),
        ("永遠維持現狀", "Status quo"),
        ("維持現狀,以後走向獨立", "Independence"),
        ("儘快宣布獨立", "Independence"),
        ("維持現狀,以後走向統一", "Unification"),
        ("儘快與中國大陸統一", "Unification"),
        ("不知道", "No answer"),
    ],
)
def test_independence_rules(raw, expected):
    labels = recode.apply_rules(pd.Series([raw]), recode.INDEPENDENCE_RULES, "No answer")
    assert labels[0] == expected


def test_identity_rules_do_not_treat_neither_as_both():
    raw = pd.Series(["兩者都是:是中國人也是台灣人", "都是", "都不是,請說明", "台灣人"])
    labels = recode.apply_rules(raw, recode.IDENTITY_RULES, "Other or no answer").tolist()
    assert labels == ["Both", "Both", "Other or no answer", "Taiwanese"]


def test_graduate_degree_is_not_coded_as_university():
    raw = pd.Series(["碩士(續答b2)", "大學", "技術學院、科大"])
    labels = recode.apply_rules(raw, recode.EDUCATION_RULES, "Other").tolist()
    assert labels == ["Graduate", "University", "University"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2-3萬元以下", 2.5), ("1萬元以下", 0.5), ("30萬元以上", 37.5), ("無收入", 0.0), ("沒有工作", 0.0)],
)
def test_income_midpoints(raw, expected):
    assert recode.income_midpoint(raw) == expected


def test_income_refusal_is_missing():
    assert math.isnan(recode.income_midpoint("拒答"))


def test_sample():
    sample, _ = build_sample()
    assert sample["age"].between(18, 100).all()
    assert set(sample["wave"]) == {2012, 2013, 2014}
    assert (sample.loc[sample["wave"] == 2014, "vote_2008"] == "Not asked").all()
    partisans = sample[sample["partisan"] == 1]
    assert set(partisans["party"]) == {"DPP", "KMT"}
    assert len(partisans) == 1707
