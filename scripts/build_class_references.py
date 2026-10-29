"""Build the team-authored reference cards for the real class labs (MU_L01 ... MU_L10).

STATUS: team-authored reference, not faculty-issued. No faculty solutions or rubrics were
available. Every task text below was reconstructed by the team from what the class notebooks
attempt. Expected outputs are taken from executing the reference solution (or, where noted,
from closed-form values), never from student work.

Usage:
    python scripts/build_class_references.py --data-dir <folder with kerala.csv,
        population_by_county.xlsx, train.csv> [--kernel python3]

Writes data/class/references/<LAB>/{template.ipynb, reference_solution.ipynb,
reference_explanations.json, expected_outputs.json, rubric.json}.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "class" / "references"
STATUS = "team-authored reference, not faculty-issued"
BANNER = (f"> **{STATUS.capitalize()}.** Written by Group 3 for the Managing Uncertainty labs. "
          "It is not the faculty handout or the faculty solution.")


def C(term, *variants):
    return {"term": term, "variants": list(variants)}


# Each lab: title, tasks [(task_id, prompt, solution_code, explanation_prompt or None)],
# explanations {task: (text, concepts)}, expected {task: spec}, alignment, data files.
LABS: dict[str, dict] = {}

LABS["MU_L01"] = dict(
    title="Lab 1: Car rental billing (Python control flow and functions)",
    data=[],
    tasks=[
        ("T1", "A rental company charges per day: SUV Rs 1000, Sedan Rs 500, Hatchback Rs 250. "
               "Rentals longer than 10 days get a 10% discount. A refundable deposit of Rs 500 is "
               "collected up front and deducted from the amount payable at return. Read the car type "
               "and number of days and print the bill (the reference fixes the inputs instead of using input()).",
         '''RATES = {"SUV": 1000, "SEDAN": 500, "HATCHBACK": 250}
DEPOSIT = 500

def rental_bill(car_type, days):
    gross = RATES[car_type.upper()] * days
    discount = 0.10 * gross if days > 10 else 0.0
    payable = gross - discount - DEPOSIT
    return gross, discount, payable

car_type, days = "SUV", 11   # in class this came from input()
gross, discount, payable = rental_bill(car_type, days)
print(f"Car: {car_type}, days: {days}")
print(f"Gross: {gross:.0f}, discount: {discount:.0f}, payable after deposit: {payable:.0f}")''', None),
        ("T2", "For one month (4 weeks) the SUV is rented on Mon/Wed/Fri, the Sedan on Tue/Thu/Sat and "
               "the Hatchback on Sunday. Using functions, print the bill for each car and the total.",
         '''schedule = {"SUV": 3 * 4, "SEDAN": 3 * 4, "HATCHBACK": 1 * 4}
total = 0
for car, d in schedule.items():
    g, disc, pay = rental_bill(car, d)
    total += pay
    print(f"{car}: {d} days, gross {g:.0f}, discount {disc:.0f}, payable {pay:.0f}")
print(f"Total payable for the month: {total:.0f}")''',
         "Explain why the SUV and Sedan bills get the discount but the Hatchback bill does not, "
         "and how the deposit is treated."),
    ],
    explanations={"T2": (
        "The SUV and the Sedan are each rented for 12 days, which is more than the 10 day threshold, "
        "so a 10% discount is applied to their gross amount. The Hatchback is rented for only 4 days, "
        "so no discount applies. The refundable deposit of 500 is deducted from each payable amount. "
        "Writing the bill as a function lets the same rule be reused for every car.",
        [C("more than 10 days", "10 days", "threshold", "greater than 10"),
         C("10% discount", "10 percent", "ten percent", "discount"),
         C("deposit deducted", "deposit", "refundable"),
         C("function reuse", "function", "reuse")])},
    expected={"T2": {"value": "10300 4900", "match": "numeric", "tol": 0.5}},
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15},
)

LABS["MU_L02"] = dict(
    title="Lab 2: Numpy and Pandas exercises",
    data=["population_by_county.xlsx"],
    template_from_handout=True,   # blank class handout found in the dataset is used as the template
    tasks=[
        ("T1", "Numpy exercises Q1-Q19 (arrays, identity matrix, random numbers, linspace, indexing "
               "on mat = np.arange(1,26).reshape(5,5), sum, standard deviation, column sums).",
         '''import numpy as np
print(np.zeros(10)); print(np.ones(10) * 5); print(np.ones(10))
print(np.arange(10, 51, 2)); print(np.arange(10, 51))
print(np.eye(3)); print(np.arange(9).reshape(3, 3))
rng = np.random.default_rng(0)
print(rng.random()); print(rng.standard_normal(25))
print(np.linspace(0, 1, 20))
mat = np.arange(1, 26).reshape(5, 5)
print(mat[:3, :2]); print(mat[3, 4]); print(mat[:3, 1]); print(mat[-1]); print(mat[-2:])
print("sum:", mat.sum())
print("std:", round(float(mat.std()), 4))
print("column sums:", mat.sum(axis=0))''',
         "Explain the difference between np.arange and np.linspace."),
        ("T2", "Pandas exercises Q1-Q12 on population_by_county.xlsx (head, columns, number of states, "
               "most common county names, most populated counties and states, counties above 1 million, "
               "names without 'County', percent change 2010-2017).",
         '''import pandas as pd
pop = pd.read_excel("population_by_county.xlsx")
print(pop.head()); print(pop.columns.tolist())
print("states:", pop["State"].nunique())
print(pop["State"].unique()[:5])
print(pop["County"].value_counts().head())
print(pop.sort_values("2010Census", ascending=False).head())
print(pop.groupby("State")["2010Census"].sum().sort_values(ascending=False).head())
print("counties over 1 million:", (pop["2010Census"] > 1_000_000).sum())
print("names without 'County':", (~pop["County"].str.contains("County")).sum())
pop["PercentChange"] = 100 * (pop["2017PopEstimate"] - pop["2010Census"]) / pop["2010Census"]
st = pop.groupby("State")[["2010Census", "2017PopEstimate"]].sum()
st["PercentChange"] = 100 * (st["2017PopEstimate"] - st["2010Census"]) / st["2010Census"]
print(st.sort_values("PercentChange", ascending=False).head())''',
         "Explain how you counted the number of states and why value_counts answers Q6."),
    ],
    explanations={
        "T1": ("np.arange steps from a start value by a fixed step size and excludes the stop value. "
               "np.linspace returns a fixed number of evenly spaced points and includes both end points.",
               [C("step size", "step"), C("number of points", "number of samples", "how many points"),
                C("end point included", "includes the end", "inclusive", "endpoint")]),
        "T2": ("nunique counts the distinct values of the State column, so it gives the number of states "
               "(51 including the District of Columbia). value_counts counts how often each county name "
               "occurs and sorts by frequency, so its head gives the most common names.",
               [C("unique values", "nunique", "distinct"), C("frequency", "value_counts", "count of each"),
                C("sorted", "sort", "descending")])},
    expected={"T1": {"value": "325 7.2111", "match": "numeric", "tol": 0.001},
              "T2": {"value": "51", "match": "numeric", "tol": 0}},
    alignment={"mode": "loose", "generic_q": False, "min_similarity": 0.15,
               "markers": [["numpy exercises", "T1"], ["pandas exercises", "T2"],
                           ["^import pandas", "T2"], ["read_excel", "T2"]]},
)

LABS["MU_L03"] = dict(
    title="Lab 3: Descriptive statistics (mean, median, mode, histograms)",
    data=[],
    tasks=[
        ("T1", "For data = [1, 3, 4, 5, 7, 9, 2] compute the mean by hand (sum / n) and with the statistics module.",
         '''import statistics
data = [1, 3, 4, 5, 7, 9, 2]
mean_manual = sum(data) / len(data)
print("manual mean:", round(mean_manual, 4))
print("statistics.mean:", round(statistics.mean(data), 4))''', None),
        ("T2", "Plot a histogram of data, and a histogram of 200 values drawn from a normal distribution "
               "with mean 100 and standard deviation 10.",
         '''import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.hist(data, bins=5); plt.title("data"); plt.show()
sample = np.random.default_rng(0).normal(100, 10, 200)
plt.hist(sample, bins=20); plt.title("normal(100, 10), n=200"); plt.show()
print("sample mean:", round(float(sample.mean()), 2), "sample sd:", round(float(sample.std(ddof=1)), 2))''',
         "Describe the shape of the normal histogram and how it relates to the mean and standard deviation."),
        ("T3", "Compute the median of data by sorting (odd and even n), and check with statistics.median.",
         '''s = sorted(data); n = len(s)
median_manual = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
print("manual median:", median_manual)
print("statistics.median:", statistics.median(data))''', None),
        ("T4", "Find the mode of [1, 3, 4, 5, 7, 9, 1, 7]. It has two modes.",
         '''from collections import Counter
d2 = [1, 3, 4, 5, 7, 9, 1, 7]
counts = Counter(d2); top = max(counts.values())
print("modes:", sorted(k for k, v in counts.items() if v == top))
print("statistics.multimode:", statistics.multimode(d2))''',
         "Why does statistics.mode return a single value here, and why is multimode the better answer?"),
    ],
    explanations={
        "T2": ("The normal histogram is roughly bell shaped and symmetric around the mean of 100. Most values "
               "lie within one standard deviation (90 to 110) and almost all within three (70 to 130). "
               "With only 200 values the bars are uneven.",
               [C("bell shaped", "bell", "bell curve"), C("symmetric", "symmetrical"),
                C("centred at the mean", "mean of 100", "centered", "around 100"),
                C("spread set by standard deviation", "standard deviation", "spread", "within one")]),
        "T4": ("Both 1 and 7 occur twice, so the data are bimodal. statistics.mode returns only the first "
               "mode it meets (1), so it hides the second one. multimode returns every value with the "
               "highest count.",
               [C("bimodal", "two modes", "more than one mode"), C("highest frequency", "occur twice", "most often"),
                C("multimode", "all modes")])},
    expected={"T1": {"value": "4.4286", "match": "numeric", "tol": 0.001},
              "T3": {"value": "4", "match": "numeric", "tol": 0},
              "T4": {"value": "1 7", "match": "numeric", "tol": 0}},
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15,
               "markers": [["hist", "T2"], ["median", "T3"], ["mode", "T4"]]},
)

LABS["MU_L04"] = dict(
    title="Lab 4: Dice simulation (empirical vs theoretical probability)",
    data=[],
    tasks=[
        ("T1", "Simulate N rolls of one fair die. Print the frequency and relative frequency of each face "
               "and compare with 1/6.",
         '''import numpy as np
rng = np.random.default_rng(42)
N = 10_000
rolls = rng.integers(1, 7, size=N)
for face in range(1, 7):
    f = int((rolls == face).sum())
    print(face, f, round(f / N, 4), "theoretical", round(1 / 6, 4))''', None),
        ("T2", "Simulate N rolls of two dice. For each sum 2-12 print the empirical probability next to the "
               "theoretical one, and plot both.",
         '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sums = rng.integers(1, 7, size=N) + rng.integers(1, 7, size=N)
emp, theo = [], []
for s in range(2, 13):
    emp.append((sums == s).mean()); theo.append((6 - abs(s - 7)) / 36)
    print(s, round(emp[-1], 4), round(theo[-1], 4))
plt.bar(range(2, 13), emp, alpha=0.6, label="empirical"); plt.plot(range(2, 13), theo, "ro-", label="theoretical")
plt.legend(); plt.show()''',
         "Explain why 7 is the most likely sum and why the empirical probabilities approach the theoretical ones."),
    ],
    explanations={"T2": (
        "Seven can be made in six of the 36 equally likely outcomes, more than any other sum, so it has "
        "probability 6/36. By the law of large numbers the relative frequencies get closer to the "
        "theoretical probabilities as the number of rolls grows.",
        [C("36 outcomes", "36", "equally likely"), C("seven most likely", "7", "six ways"),
         C("law of large numbers", "large number of trials", "more rolls", "converge")])},
    expected={},   # random outputs: no fixed values to compare
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15},
)

LABS["MU_L05"] = dict(
    title="Lab 5: Exploratory data analysis on the Titanic training data",
    data=["train.csv"],
    data_note="train.csv was not in the class dataset; the public Kaggle Titanic training file (891 rows) was used.",
    tasks=[
        ("T1", "Load train.csv, show its shape and missing values, and count survivors and non-survivors.",
         '''import pandas as pd
df = pd.read_csv("train.csv")
print(df.shape)
print(df.isnull().sum())
print(df["Survived"].value_counts())''', None),
        ("T2", "Plot count plots / bar charts for Sex, Pclass, Parch and Embarked.",
         '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for col in ["Sex", "Pclass", "Parch", "Embarked"]:
    df[col].value_counts().plot(kind="bar", title=col); plt.show()
print(df["Pclass"].value_counts())''', None),
        ("T3", "Tabulate survival by passenger class and by sex, and the survival rate of each group.",
         '''print(pd.crosstab(df["Pclass"], df["Survived"]))
print(pd.crosstab(df["Sex"], df["Survived"]))
print(df.groupby("Sex")["Survived"].mean().round(4))
print(df.groupby("Pclass")["Survived"].mean().round(4))''',
         "What do the tables say about who was more likely to survive?"),
    ],
    explanations={"T3": (
        "Women survived far more often than men (about 74% against 19%), and first class passengers "
        "survived more often than third class passengers (about 63% against 24%). Survival depends on "
        "sex and class, so these are conditional probabilities that differ from the overall rate of 38%.",
        [C("women more likely", "female", "women"), C("first class higher", "first class", "pclass 1", "class 1"),
         C("third class lower", "third class", "class 3"), C("conditional", "rate", "depends on")])},
    expected={"T1": {"value": "549 342", "match": "numeric", "tol": 0},
              "T2": {"value": "491 216 184", "match": "numeric", "tol": 0}},
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15},
)

LABS["MU_L06"] = dict(
    title="Lab 6: Conditional probability with Kerala rainfall and floods",
    data=["kerala.csv"],
    tasks=[
        ("T1", "Load kerala.csv. Encode FLOODS as 1/0 and add indicator columns for June and July rainfall above 500 mm.",
         '''import pandas as pd
df = pd.read_csv("kerala.csv")
df.columns = df.columns.str.strip()
df["F"] = (df["FLOODS"] == "YES").astype(int)
df["JUN_GT_500"] = (df["JUN"] > 500).astype(int)
df["JUL_GT_500"] = (df["JUL"] > 500).astype(int)
print(df.shape, df["F"].sum())''', None),
        ("T2", "Cross-tabulate floods against June rain above 500 mm. Compute P(F), P(J), P(F and J) and P(F | J).",
         '''n = len(df)
print(pd.crosstab(df["F"], df["JUN_GT_500"]))
pF = df["F"].mean(); pJ = df["JUN_GT_500"].mean()
pFJ = ((df["F"] == 1) & (df["JUN_GT_500"] == 1)).mean()
print("P(F) =", round(pF, 4), "P(J) =", round(pJ, 4), "P(F and J) =", round(pFJ, 4))
print("P(F | J) =", round(pFJ / pJ, 4))''',
         "Does heavy June rain change the chance of a flood year? Compare P(F | J) with P(F)."),
        ("T3", "Repeat for July rain above 500 mm, and also compute P(July > 500 | F) using Bayes' rule.",
         '''pJul = df["JUL_GT_500"].mean()
pFJul = ((df["F"] == 1) & (df["JUL_GT_500"] == 1)).mean()
print(pd.crosstab(df["F"], df["JUL_GT_500"]))
print("P(F | JUL) =", round(pFJul / pJul, 4))
print("P(JUL | F) =", round(pFJul / pF, 4), "via Bayes:", round((pFJul / pJul) * pJul / pF, 4))''', None),
    ],
    explanations={"T2": (
        "P(F | J) is the probability of a flood year given that June rain exceeded 500 mm. It is about "
        "0.58, a little higher than the unconditional P(F) of about 0.51, so heavy June rain raises the "
        "chance of floods only slightly. The events are not independent, but June rain alone is a weak "
        "predictor.",
        [C("conditional probability", "given that", "p(f|j)", "conditional"),
         C("compare with unconditional", "p(f)", "unconditional", "base rate"),
         C("independence", "independent", "not independent"),
         C("slightly higher", "higher", "increases", "raises")])},
    expected={"T2": {"value": "0.5806", "match": "numeric", "tol": 0.001},
              "T3": {"value": "0.5938 0.95", "match": "numeric", "tol": 0.001}},
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15,
               "markers": [["jul", "T3"]]},
)

LABS["MU_L07_08"] = dict(
    title="Labs 7-8: Binomial and Poisson distributions",
    data=[],
    tasks=[
        ("T1", "X ~ Binomial(n=12, p=0.45). Find P(X = 8), P(X > 4) and P(X <= 10).",
         '''from scipy.stats import binom, poisson
n, p = 12, 0.45
print("P(X=8)  =", round(binom.pmf(8, n, p), 6))
print("P(X>4)  =", round(1 - binom.cdf(4, n, p), 6))
print("P(X<=10)=", round(binom.cdf(10, n, p), 6))''', None),
        ("T2", "For n = 10 trials find: (a) P(X = 4) with p = 0.3, (b) P(X = 0) with p = 0.55, "
               "(c) P(X = 2) with p = 0.15, (d) P(X >= 8) with p = 0.3.",
         '''print("a", round(binom.pmf(4, 10, 0.3), 6))
print("b", round(binom.pmf(0, 10, 0.55), 6))
print("c", round(binom.pmf(2, 10, 0.15), 6))
print("d", round(1 - binom.cdf(7, 10, 0.3), 6))''', None),
        ("T3", "Arrivals follow a Poisson distribution with mean 4.2. Find P(X <= 2), P(X >= 5) and P(X = 8).",
         '''lam = 4.2
print("P(X<=2) =", round(poisson.cdf(2, lam), 6))
print("P(X>=5) =", round(1 - poisson.cdf(4, lam), 6))
print("P(X=8)  =", round(poisson.pmf(8, lam), 6))''',
         "When is the Poisson distribution an appropriate model, and why is P(X >= 5) computed as 1 - P(X <= 4)?"),
    ],
    explanations={"T3": (
        "The Poisson distribution models the number of events in a fixed interval when events occur "
        "independently at a constant average rate. P(X >= 5) is the complement of P(X <= 4), so it is "
        "1 minus the cumulative distribution function at 4. P(X = 8) needs the probability mass function, "
        "not the cdf.",
        [C("fixed interval", "interval", "per unit time"), C("independent events", "independently", "independent"),
         C("constant rate", "average rate", "mean rate", "lambda"), C("complement", "1 -", "1 minus"),
         C("pmf versus cdf", "pmf", "mass function", "cdf")])},
    expected={"T1": {"value": "0.076165 0.695568 0.99892", "match": "numeric", "tol": 0.0001},
              "T2": {"value": "0.200121 0.000341 0.275897 0.00159", "match": "numeric", "tol": 0.0001},
              "T3": {"value": "0.210238 0.410173 0.036011", "match": "numeric", "tol": 0.0001}},
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15, "markers": [["poisson", "T3"]]},
)

_REJECT = "(?<!fail to )(?<!do not )(?<!don't )(?<!not )reject"
LABS["MU_L09"] = dict(
    title="Lab 9: One-sample t-tests",
    data=[],
    tasks=[
        ("T1", "A machine should fill 12 oz. A sample of n = 10 has mean 11.8 and s = 0.15. Test H0: mu = 12 "
               "against a two-tailed alternative at alpha = 0.05.",
         '''import math
from scipy import stats

def one_sample_t(xbar, mu, s, n, alpha, tails):
    t = (xbar - mu) / (s / math.sqrt(n))
    df = n - 1
    if tails == 2:
        crit = stats.t.ppf(1 - alpha / 2, df); reject = abs(t) > crit
    else:
        crit = stats.t.ppf(1 - alpha, df); reject = t > crit
    print(f"t = {t:.4f}, critical = {crit:.4f}, df = {df}")
    print("Reject the null hypothesis" if reject else "Fail to reject the null hypothesis")
    return t

one_sample_t(11.8, 12, 0.15, 10, 0.05, 2)''', None),
        ("T2", "A new method claims to raise the mean above 100. A sample of n = 25 has mean 130 and s = 15. "
               "Test at alpha = 0.05, one-tailed.",
         '''one_sample_t(130, 100, 15, 25, 0.05, 1)''', None),
        ("T3", "Scores [62, 92, 75, 68, 83, 95]. Test whether the mean exceeds 70 at the 90% confidence level (one-tailed).",
         '''import numpy as np
scores = np.array([62, 92, 75, 68, 83, 95])
t3 = one_sample_t(scores.mean(), 70, scores.std(ddof=1), len(scores), 0.10, 1)
print(stats.ttest_1samp(scores, 70, alternative="greater"))''',
         "State the hypotheses, the decision and what it means in context."),
    ],
    explanations={"T3": (
        "H0: the mean score is 70; H1: the mean score is greater than 70. The t statistic is larger than "
        "the one-tailed critical value at alpha = 0.10 (equivalently the p-value is below 0.10), so we reject "
        "the null hypothesis and conclude the mean score is above 70. The sample uses the sample standard "
        "deviation with n - 1 degrees of freedom.",
        [C("null hypothesis", "h0", "null"), C("alternative hypothesis", "h1", "alternative", "greater than 70"),
         C("critical value or p-value", "critical value", "p-value", "p value"),
         C("degrees of freedom", "n - 1", "df"), C("reject", "rejected")])},
    expected={"T1": {"value": _REJECT, "match": "regex"},
              "T2": {"value": _REJECT, "match": "regex"},
              "T3": {"value": _REJECT, "match": "regex"}},
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15},
)

LABS["MU_L10"] = dict(
    title="Lab 10: Two-sample t-tests",
    data=[],
    tasks=[
        ("T1", "Two groups of n = 20 have means 44 and 28 and standard deviations 30 and 20. Test whether the "
               "difference of means exceeds 2 at alpha = 0.05 (one-tailed, pooled variance).",
         '''import math
from scipy import stats

def two_sample_t(x1, x2, s1, s2, n1, n2, delta, alpha, tails):
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df
    t = (x1 - x2 - delta) / math.sqrt(sp2 * (1 / n1 + 1 / n2))
    if tails == 2:
        crit = stats.t.ppf(1 - alpha / 2, df); reject = abs(t) > crit
    else:
        crit = stats.t.ppf(1 - alpha, df); reject = t > crit
    print(f"t = {t:.4f}, critical = {crit:.4f}, df = {df}")
    print("Reject the null hypothesis" if reject else "Fail to reject the null hypothesis")
    return t

two_sample_t(44, 28, 30, 20, 20, 20, 2, 0.05, 1)''', None),
        ("T2", "Two machines: means 200 and 250, standard deviations 20 and 25, n = 25 each. Test for a "
               "difference at alpha = 0.01 (two-tailed).",
         '''two_sample_t(200, 250, 20, 25, 25, 25, 0, 0.01, 2)''',
         "Explain the decision, and why the degrees of freedom are n1 + n2 - 2."),
    ],
    explanations={"T2": (
        "The absolute t statistic (about 7.8) is far above the two-tailed critical value at alpha = 0.01, "
        "so we reject the null hypothesis that the machines have equal means. With a pooled variance two "
        "means are estimated from the data, so the degrees of freedom are n1 + n2 - 2 = 48.",
        [C("reject null", "reject"), C("critical value", "critical", "p-value", "p value"),
         C("pooled variance", "pooled"), C("degrees of freedom", "n1 + n2 - 2", "48", "df")])},
    expected={"T1": {"value": _REJECT, "match": "regex"},
              "T2": {"value": "-7.8087", "match": "numeric", "tol": 0.001}},
    alignment={"mode": "loose", "generic_q": True, "min_similarity": 0.15},
)


# Reference-author style choices that students may legitimately do differently.
NOT_REQUIRED = ["use", "default_rng", "show", "legend", "bar", "plot", "Counter", "multimode",
                "ttest_1samp", "integers", "random", "astype", "array", "std", "round", "contains"]


def _nb(cells, title):
    nb = new_notebook(cells=cells)
    nb.metadata["reference_status"] = STATUS
    nb.metadata["title"] = title
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def build_lab(lab: str, spec: dict, data_dir: Path, kernel: str, handout: Path | None) -> None:
    d = OUT / lab
    d.mkdir(parents=True, exist_ok=True)
    head = [new_markdown_cell(f"# {spec['title']}\n\n{BANNER}" +
                              (f"\n\nData note: {spec['data_note']}" if spec.get("data_note") else ""))]
    tmpl, sol = list(head), list(head)
    for tid, prompt, code, explain in spec["tasks"]:
        header = f"## Task {tid[1:]}\n\n{prompt}"
        tmpl += [new_markdown_cell(header), new_code_cell("# your code here")]
        sol += [new_markdown_cell(header), new_code_cell(code)]
        if explain:
            tmpl += [new_markdown_cell(f"**Explain:** {explain}\n\n_Your answer here._")]
            sol += [new_markdown_cell(f"**Explain:** {explain}\n\n{spec['explanations'][tid][0]}")]
    if spec.get("template_from_handout") and handout and handout.is_file():
        t = nbformat.read(str(handout), as_version=4)
        t.nbformat_minor = 5   # handout is v4.4; new cells carry ids
        for i, c in enumerate(t.cells):
            c.setdefault("id", f"handout-{i}")
        t.cells.insert(0, new_markdown_cell(
            f"{BANNER}\n\nThe cells below are the blank class handout as found in the dataset "
            "(an unfilled student copy). It is used only to recognise prompt text in submissions."))
        t.metadata["reference_status"] = STATUS
        nbformat.write(t, str(d / "template.ipynb"))
    else:
        nbformat.write(_nb(tmpl, spec["title"]), str(d / "template.ipynb"))

    sol_nb = _nb(sol, spec["title"])
    from nbclient import NotebookClient
    with tempfile.TemporaryDirectory() as tmp:
        for f in spec["data"]:
            shutil.copy(data_dir / f, Path(tmp) / f)
        NotebookClient(sol_nb, timeout=120, kernel_name=kernel, resources={"metadata": {"path": tmp}}).execute()
    for c in sol_nb.cells:   # drop image payloads: keeps the reference small and diff-able
        for o in c.get("outputs", []):
            if "data" in o:
                o["data"] = {k: v for k, v in o["data"].items() if k == "text/plain"}
    nbformat.write(sol_nb, str(d / "reference_solution.ipynb"))

    stamp = {"_status": STATUS, "_lab": lab, "_title": spec["title"]}
    (d / "reference_explanations.json").write_text(json.dumps({**stamp, "tasks": {
        t: {"text": txt, "concepts": cs} for t, (txt, cs) in spec["explanations"].items()}}, indent=2))
    (d / "expected_outputs.json").write_text(json.dumps({**stamp, "tasks": spec["expected"]}, indent=2))
    rubric = {**stamp, "code": 8, "explanation": 8, "documentation": 4, "alignment": spec["alignment"],
              "not_required": NOT_REQUIRED,
              "criteria": {
                  "code": "Each task attempted, runs without error, uses the required functions, and its "
                          "printed output matches expected_outputs.json.",
                  "explanation": "Markdown or comment explanation covers the reference concepts for tasks "
                                 "that ask to explain.",
                  "documentation": "Headings or comments label each task; outputs are shown; no stale errors."},
              "tasks": {t: p for t, p, _, _ in spec["tasks"]}}
    (d / "rubric.json").write_text(json.dumps(rubric, indent=2))
    print(f"[ok] {lab}: {len(spec['tasks'])} tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--handout", type=Path, help="blank MU Lab 2 handout notebook (template for MU_L02)")
    ap.add_argument("--labs", nargs="*")
    a = ap.parse_args()
    for lab, spec in LABS.items():
        if not a.labs or lab in a.labs:
            build_lab(lab, spec, a.data_dir, a.kernel, a.handout)


if __name__ == "__main__":
    main()
