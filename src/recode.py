"""Harmonize the Chinese-language response labels of the three survey waves.

The 2012, 2013 and 2014 questionnaires ask the same questions but label the
answers differently (for example "男" versus "男性", or "國民黨(跳答h7)" with a
skip-pattern note attached). Each recode below is a list of (pattern, label)
rules applied in order: the first regular expression that matches the cleaned
answer decides the label, and anything unmatched falls into a default bucket.
Keeping the rules as data makes every coding decision visible in one place.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

Rules = list[tuple[str, str]]

# Questionnaire routing notes such as "(跳答h7)" (skip to h7) or "(續答b2)"
# (continue to b2) are interviewer instructions, not part of the answer.
_ROUTING_NOTE = re.compile(r"\((?:跳答|續答)[^)]*\)")


def strip_routing(value: object) -> object:
    if not isinstance(value, str):
        return value
    return _ROUTING_NOTE.sub("", value).strip()


def apply_rules(series: pd.Series, rules: Rules, default: str) -> pd.Series:
    """Label each answer with the first matching rule, else `default`."""
    cleaned = series.map(strip_routing)

    def label(value: object) -> str:
        if isinstance(value, str):
            for pattern, name in rules:
                if re.search(pattern, value):
                    return name
        return default

    return cleaned.map(label)


# Party identification. 2013 and 2014 add "leans blue / leans green"
# (泛藍 / 泛綠) options that 2012 lacks; leaners are grouped with the party.
PARTY_RULES: Rules = [
    (r"民進黨|泛綠", "DPP"),
    (r"國民黨|泛藍", "KMT"),
]

# National self-identification. Both orderings of "both Taiwanese and
# Chinese" (2014 asks them separately) are merged.
IDENTITY_RULES: Rules = [
    (r"^台灣人$", "Taiwanese"),
    (r"^中國人$", "Chinese"),
    (r"^(都是|兩者都是.*)$", "Both"),
]

# Independence versus unification, the standard six-point Taiwanese scale
# collapsed to three directions. 2014 adds "status quo, decide later"
# (看情形再決定), which mentions both outcomes, so it is matched first.
INDEPENDENCE_RULES: Rules = [
    (r"看情形|永遠維持現狀", "Status quo"),
    (r"統一", "Unification"),
    (r"獨立", "Independence"),
]

# Vote in the January 2012 presidential election, asked in all three waves.
VOTE_2012_RULES: Rules = [
    (r"蔡英文", "Tsai (DPP)"),
    (r"馬英九", "Ma (KMT)"),
    (r"宋楚瑜|投廢票|拒領", "Soong or blank"),
    (r"沒有去投票|未滿20歲|沒有投票權|沒有總統投票權", "Did not vote"),
]

# Vote in the 2008 presidential election. In 2014 this variable records
# whether the respondent took a referendum ballot instead, so it is only
# usable for 2012 and 2013 (see build_sample.py).
VOTE_2008_RULES: Rules = [
    (r"謝長廷", "Hsieh (DPP)"),
    (r"馬英九", "Ma (KMT)"),
    (r"投廢票|拒領", "Blank"),
    (r"沒有去投票|未滿20歲|沒有投票權|沒有總統投票權", "Did not vote"),
]

# Highest education completed, grouped by level. Graduate degrees are
# matched before "大學" so that a master's is not coded as university.
EDUCATION_RULES: Rules = [
    (r"碩士|博士", "Graduate"),
    (r"大學|技術學院|科大|軍警官校", "University"),
    (r"二專|三專|五專|空中行專|商專|專科班|專修班|士官學校", "Junior college"),
    (r"高中|高職", "Senior high"),
    (r"國\(初\)中|初職", "Junior high"),
    (r"小學|不識字|自修|私塾", "Primary or less"),
]
EDUCATION_ORDER = [
    "Primary or less",
    "Junior high",
    "Senior high",
    "Junior college",
    "University",
    "Graduate",
    "Other",
]

MARRIAGE_RULES: Rules = [
    (r"已婚|同居", "Married or cohabiting"),
    (r"單身", "Never married"),
    (r"離婚|分居|配偶去世", "Divorced, separated or widowed"),
]

_INCOME_BAND = re.compile(r"^(\d+)-(\d+)萬元以下$")
_INCOME_BELOW = re.compile(r"^(\d+)萬元以下$")
_INCOME_ABOVE = re.compile(r"^(\d+)萬元以上$")
_NO_INCOME = re.compile(r"沒有工作|沒有收入|無收入")


def income_midpoint(value: object) -> float:
    """Convert an income bracket to its midpoint in NT$10,000 per month.

    "2-3萬元以下" (20,000 to 30,000) becomes 2.5, "1萬元以下" becomes 0.5,
    and an open top bracket such as "30萬元以上" is top-coded at 1.25 times
    its floor. No job or no income is 0; refusals and "don't know" are NaN.
    """
    if not isinstance(value, str):
        return np.nan
    value = strip_routing(value)
    if _NO_INCOME.search(value):
        return 0.0
    if match := _INCOME_BAND.match(value):
        low, high = map(float, match.groups())
        return (low + high) / 2
    if match := _INCOME_BELOW.match(value):
        return float(match.group(1)) / 2
    if match := _INCOME_ABOVE.match(value):
        return float(match.group(1)) * 1.25
    return np.nan
