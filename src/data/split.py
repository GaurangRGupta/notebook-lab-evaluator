"""Notebook-level, group-preserving train/validation/test split.

Unit of splitting = notebook. All cells inherit their notebook's split.
Groups (never split across sets):
  * group_key: same student (salted hash) for class data, same repo/author for public data
  * near-duplicate clusters: copied / forked notebooks are merged into one group

Target proportions are approximately 60/20/20. Because whole groups are assigned,
the realised proportions can deviate from the target, especially on small datasets;
the realised proportions are always reported, never assumed.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def merge_near_duplicates(ids: list[str], texts: list[str], groups: list[str], threshold: float = 0.95) -> list[str]:
    """Union-find: notebooks whose student content has TF-IDF cosine >= threshold share a group."""
    parent = {g: g for g in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nonempty = [i for i, t in enumerate(texts) if t.strip()]
    if len(nonempty) > 1:
        m = TfidfVectorizer().fit_transform([texts[i] for i in nonempty])
        sim = cosine_similarity(m)
        for a in range(len(nonempty)):
            for b in range(a + 1, len(nonempty)):
                if sim[a, b] >= threshold:
                    ga, gb = find(groups[nonempty[a]]), find(groups[nonempty[b]])
                    if ga != gb:
                        parent[gb] = ga
    return [find(g) for g in groups]


def grouped_split(ids, groups, strata=None, target=(0.6, 0.2, 0.2), seed=42) -> dict:
    """Returns {"assignment": {id: split}, "realised": {...}, "method": str}.

    Method: StratifiedGroupKFold with k = round(1 / test_fraction) folds; one fold -> test,
    one -> validation, the rest -> train. Falls back to two GroupShuffleSplits when there
    are too few groups for k folds. Neither method guarantees exact proportions."""
    ids, groups = list(ids), list(groups)
    strata = list(strata) if strata is not None else ["all"] * len(ids)
    k = max(2, round(1 / target[2]))
    n_groups = len(set(groups))
    idx = np.arange(len(ids))
    split = {}
    if n_groups >= k:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = [test for _, test in sgkf.split(idx, strata, groups)]
        n_val_folds = max(1, round(target[1] * k))
        for i in folds[0]:
            split[ids[i]] = "test"
        for f in folds[1:1 + n_val_folds]:
            for i in f:
                split[ids[i]] = "val"
        for i in idx:
            split.setdefault(ids[i], "train")
        method = f"StratifiedGroupKFold(k={k}, seed={seed}), strata=assignment/source"
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=target[2], random_state=seed)
        rest, test = next(gss.split(idx, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=target[1] / (target[0] + target[1]), random_state=seed)
        tr, va = next(gss2.split(rest, groups=[groups[i] for i in rest]))
        for i in test:
            split[ids[i]] = "test"
        for i in rest[va]:
            split[ids[i]] = "val"
        for i in rest[tr]:
            split[ids[i]] = "train"
        method = f"GroupShuffleSplit fallback (only {n_groups} groups), seed={seed}"
    check_no_group_leakage(split, dict(zip(ids, groups)))
    n = len(ids)
    realised = {s: round(sum(v == s for v in split.values()) / n, 3) for s in ("train", "val", "test")}
    return {"assignment": split, "realised": realised, "target": dict(zip(("train", "val", "test"), target)),
            "method": method, "n_notebooks": n, "n_groups": n_groups}


def check_no_group_leakage(split: dict, group_of: dict):
    seen = {}
    for nid, s in split.items():
        g = group_of[nid]
        if g in seen and seen[g] != s:
            raise AssertionError(f"group {g} appears in both {seen[g]} and {s}")
        seen[g] = s
