"""Generate SYNTHETIC DEVELOPMENT DATA for the Week 2 pipeline.

!!! Everything written by this script is synthetic, authored by the team to test
!!! the pipeline. It is NOT the project dataset, NOT real student work, and results
!!! computed on it are NOT evidence of how the evaluator performs on real submissions.

Creates, under data/synthetic/:
  raw/<LAB>/SYN_Sxx_<LAB>.ipynb      10 notebooks (2 labs x 5 quality profiles), really executed
  references/<LAB>/template.ipynb    blank lab handout
  references/<LAB>/reference_solution.ipynb
  references/<LAB>/reference_explanations.json
  references/<LAB>/expected_outputs.json   derived by executing the reference solution
  references/<LAB>/rubric.json
  annotations/raw_annotations.csv    SIMULATED labels from 3 fictitious annotators
  manifest.csv, class_list.txt       (class list contains only fictitious names)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
BANNER = ("> **SYNTHETIC DEVELOPMENT DATA.** Team-authored fixture for testing the Group 3 "
          "Jupyter Notebook Lab Evaluator pipeline. Not a real student submission.")
NAME_TEMPLATE = "**Name:** ____\n\n**Roll No:** ____"

# ============================================================== LAB_SYN1: TF-IDF
SYN1 = {
    "title": "# LAB_SYN1: TF-IDF and Cosine Similarity",
    "given_code": ('# Given corpus (do not modify)\n'
                   'docs = ["the cat sat on the mat",\n'
                   '        "the dog sat on the log",\n'
                   '        "cats and dogs are pets"]'),
    "tasks": [
        {"id": "T1", "prompt": "## Task 1: Tokenise each document (lowercase, whitespace split) and print the vocabulary size.",
         "explain": None},
        {"id": "T2", "prompt": "## Task 2: Build a TF-IDF matrix with scikit-learn's `TfidfVectorizer` and print its shape.",
         "explain": "### Explain: What does TF-IDF measure, and why does it down-weight words such as 'the'?"},
        {"id": "T3", "prompt": "## Task 3: Compute the cosine similarity between document 0 and document 1, rounded to 4 decimals.",
         "explain": "### Explain: What does cosine similarity measure, and why is it preferred over Euclidean distance for text?"},
    ],
    "reference": {
        "T1": ('tokens = [d.lower().split() for d in docs]\n'
               'vocab = sorted({t for doc in tokens for t in doc})\n'
               'print("Vocabulary size:", len(vocab))'),
        "T2": ('from sklearn.feature_extraction.text import TfidfVectorizer\n'
               'vectorizer = TfidfVectorizer()\n'
               'X = vectorizer.fit_transform(docs)\n'
               'print("TF-IDF matrix shape:", X.shape)'),
        "T3": ('from sklearn.metrics.pairwise import cosine_similarity\n'
               'sim = cosine_similarity(X[0], X[1])[0, 0]\n'
               'print("Cosine similarity:", round(sim, 4))'),
    },
    "ref_expl": {
        "T2": {"text": ("TF-IDF (term frequency-inverse document frequency) weights a word by how often it occurs "
                        "in a document (term frequency) multiplied by its inverse document frequency, the log of the "
                        "number of documents divided by the number of documents that contain the word. Words such as "
                        "'the' appear in most documents, so their IDF is low and they are down-weighted, while rare, "
                        "discriminative words receive a higher weight."),
               "concepts": [
                   {"term": "term frequency", "variants": ["tf", "how often", "frequency of the word", "frequency of a word"]},
                   {"term": "inverse document frequency", "variants": ["idf"]},
                   {"term": "logarithm", "variants": ["log"]},
                   {"term": "common words down-weighted", "variants": ["down-weighted", "down-weight", "downweight",
                        "lower weight", "low weight", "less weight", "less important", "penalise", "penalize"]},
                   {"term": "rare words weighted higher", "variants": ["rare", "discriminative", "distinctive", "unique words"]},
               ]},
        "T3": {"text": ("Cosine similarity is the cosine of the angle between two vectors: their dot product divided by "
                        "the product of their norms (magnitudes). It depends only on the direction of the vectors, so it "
                        "is insensitive to document length, whereas Euclidean distance grows with vector magnitude and "
                        "would make two documents on the same topic but of different lengths look far apart."),
               "concepts": [
                   {"term": "angle", "variants": ["angle between"]},
                   {"term": "dot product", "variants": ["inner product"]},
                   {"term": "norm", "variants": ["norms", "magnitude", "magnitudes", "normalised", "normalized"]},
                   {"term": "document length", "variants": ["length", "longer documents", "size of the document"]},
                   {"term": "direction", "variants": ["orientation"]},
               ]},
    },
}

# ============================================================== LAB_SYN2: bigram LM
SYN2 = {
    "title": "# LAB_SYN2: Bigram Language Model with Add-One Smoothing",
    "given_code": ('# Given training corpus (do not modify)\n'
                   'corpus = ["<s> the cat sat </s>",\n'
                   '          "<s> the cat ran </s>",\n'
                   '          "<s> the dog sat </s>"]'),
    "tasks": [
        {"id": "T1", "prompt": "## Task 1: Count unigrams and bigrams in the corpus and print count(the, cat).",
         "explain": None},
        {"id": "T2", "prompt": "## Task 2: Implement add-one (Laplace) smoothed bigram probability and print P(cat | the), rounded to 4 decimals.",
         "explain": "### Explain: Why is smoothing needed in an n-gram language model?"},
        {"id": "T3", "prompt": "## Task 3: Compute the perplexity of the sentence `<s> the dog ran </s>` under your smoothed model, rounded to 3 decimals.",
         "explain": "### Explain: What does perplexity measure, and how should its value be interpreted?"},
    ],
    "reference": {
        "T1": ('from collections import Counter\n'
               'unigrams, bigrams = Counter(), Counter()\n'
               'for sent in corpus:\n'
               '    toks = sent.split()\n'
               '    unigrams.update(toks)\n'
               '    bigrams.update(zip(toks[:-1], toks[1:]))\n'
               'print("count(the, cat) =", bigrams[("the", "cat")])'),
        "T2": ('V = len(unigrams)\n'
               'def p_add_one(w_prev, w):\n'
               '    return (bigrams[(w_prev, w)] + 1) / (unigrams[w_prev] + V)\n'
               'print("P(cat | the) =", round(p_add_one("the", "cat"), 4))'),
        "T3": ('import math\n'
               'test = "<s> the dog ran </s>".split()\n'
               'log_prob = sum(math.log(p_add_one(a, b)) for a, b in zip(test[:-1], test[1:]))\n'
               'perplexity = math.exp(-log_prob / (len(test) - 1))\n'
               'print("Perplexity:", round(perplexity, 3))'),
    },
    "ref_expl": {
        "T2": {"text": ("Maximum likelihood estimates assign zero probability to any bigram that never appeared in the "
                        "training data, so a sentence containing one unseen bigram gets probability zero. Add-one "
                        "(Laplace) smoothing adds one to every bigram count and adds the vocabulary size V to the "
                        "denominator, redistributing probability mass from seen to unseen events so that no probability is zero."),
               "concepts": [
                   {"term": "zero probability", "variants": ["probability of zero", "probability zero", "0 probability"]},
                   {"term": "unseen bigrams", "variants": ["unseen", "never seen", "never appeared", "not in the training"]},
                   {"term": "add one", "variants": ["add-one", "laplace", "plus one", "adding one", "adds one"]},
                   {"term": "vocabulary size", "variants": ["vocabulary", "v"]},
                   {"term": "probability mass", "variants": ["redistribute", "redistributes", "redistributing", "mass"]},
               ]},
        "T3": {"text": ("Perplexity is the exponentiated average negative log-probability per word, i.e. the inverse "
                        "probability of the test sentence normalised by the number of words. It measures how surprised "
                        "the model is by the text; a lower perplexity means the model assigns higher probability to the "
                        "test data and is therefore a better language model."),
               "concepts": [
                   {"term": "inverse probability", "variants": ["negative log", "negative log-probability", "inverse of the probability"]},
                   {"term": "normalised per word", "variants": ["per word", "number of words", "normalised", "normalized", "average"]},
                   {"term": "lower is better", "variants": ["lower perplexity", "lower is better", "lower value", "smaller perplexity"]},
                   {"term": "surprise", "variants": ["surprised", "uncertainty", "uncertain", "branching factor"]},
               ]},
    },
}

LABS = {"LAB_SYN1": SYN1, "LAB_SYN2": SYN2}

# ------------------------------------------------------------------ student profiles
# each entry: task_id -> {"code": str, "md": str | None}; plus name cell text & post-processing
STUDENTS = {
    ("LAB_SYN1", "SYN_S01"): {
        "profile": "excellent: correct code, docstrings/comments, complete correct explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": ('def tokenize(doc):\n'
                        '    """Lowercase a document and split it on whitespace."""\n'
                        '    return doc.lower().split()\n\n'
                        '# the vocabulary is the set of unique tokens across all documents\n'
                        'tokens = [tokenize(d) for d in docs]\n'
                        'vocab = sorted({t for doc in tokens for t in doc})\n'
                        'print("Vocabulary size:", len(vocab))')},
        "T2": {"code": ('from sklearn.feature_extraction.text import TfidfVectorizer\n\n'
                        '# fit the vectorizer on the corpus; rows = documents, columns = vocabulary terms\n'
                        'vectorizer = TfidfVectorizer()\n'
                        'X = vectorizer.fit_transform(docs)\n'
                        'print("TF-IDF matrix shape:", X.shape)'),
               "md": ("TF-IDF combines **term frequency** (how often a word occurs in a document) with **inverse "
                      "document frequency**, the log of the total number of documents over the number of documents "
                      "containing the word. A word like *the* occurs in almost every document, so its IDF is close to "
                      "zero and it is down-weighted; rare, distinctive words such as *pets* get a higher weight and "
                      "therefore dominate the document vector.")},
        "T3": {"code": ('from sklearn.metrics.pairwise import cosine_similarity\n\n'
                        'def doc_similarity(i, j):\n'
                        '    """Cosine similarity between rows i and j of the TF-IDF matrix."""\n'
                        '    return cosine_similarity(X[i], X[j])[0, 0]\n\n'
                        'print("Cosine similarity:", round(doc_similarity(0, 1), 4))'),
               "md": ("Cosine similarity is the cosine of the angle between two document vectors: the dot product "
                      "divided by the product of their norms. Because it only looks at the direction of the vectors, "
                      "a long and a short document about the same topic can still be similar. Euclidean distance "
                      "depends on magnitude, so document length would distort it.")},
    },
    ("LAB_SYN1", "SYN_S02"): {
        "profile": "correct code, no comments, weak/vague explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": 'vocab = set(" ".join(docs).split())\nprint(len(vocab))'},
        "T2": {"code": 'from sklearn.feature_extraction.text import TfidfVectorizer\nv = TfidfVectorizer()\nX = v.fit_transform(docs)\nX.shape',
               "md": "TF-IDF is a very popular technique in NLP. It is used to convert text into numbers."},
        "T3": {"code": 'from sklearn.metrics.pairwise import cosine_similarity\nprint(round(cosine_similarity(X[0], X[1])[0][0], 4))',
               "md": "Cosine similarity tells how similar two documents are. It is better."},
    },
    ("LAB_SYN1", "SYN_S03"): {
        "profile": "syntax error (T1), runtime NameError (T3), partly correct explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": ('# tokenise every document\n'
                        'vocab = set()\n'
                        'for d in docs\n'
                        '    vocab.update(d.split())\n'
                        'print("Vocabulary size:", len(vocab))')},
        "T2": {"code": ('from sklearn.feature_extraction.text import TfidfVectorizer\n'
                        '# build tf-idf features\n'
                        'X = TfidfVectorizer().fit_transform(docs)\n'
                        'print("TF-IDF matrix shape:", X.shape)'),
               "md": ("TF-IDF multiplies the term frequency by the inverse document frequency. Words that are in "
                      "every document like 'the' get less weight.")},
        "T3": {"code": ('from sklearn.metrics.pairwise import cosine_similarity\n'
                        '# compare the first two documents\n'
                        'sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])\n'
                        'print(round(sim[0][0], 4))'),
               "md": "Cosine similarity measures the angle between two vectors, so it does not depend on how long the documents are."},
    },
    ("LAB_SYN1", "SYN_S04"): {
        "profile": "output mismatches (T1, T3), wrong construct (CountVectorizer), conceptually wrong explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": ('# count the words\n'
                        'words = []\n'
                        'for d in docs:\n'
                        '    words.extend(d.split())\n'
                        'print("Vocabulary size:", len(words))')},
        "T2": {"code": ('from sklearn.feature_extraction.text import CountVectorizer\n'
                        'cv = CountVectorizer()\n'
                        'X = cv.fit_transform(docs)\n'
                        'print("TF-IDF matrix shape:", X.shape)'),
               "md": ("TF-IDF gives a higher weight to words that appear in many documents, because words like 'the' "
                      "are the most important for the meaning of a document. It counts the frequency of each word.")},
        "T3": {"code": ('import numpy as np\n'
                        '# similarity = dot product of the two rows\n'
                        'a = X[0].toarray()[0]\n'
                        'b = X[1].toarray()[0]\n'
                        'print("Cosine similarity:", np.dot(a, b))'),
               "md": "Cosine similarity is the Euclidean distance between the two vectors; a smaller distance means the documents are more similar."},
    },
    ("LAB_SYN1", "SYN_S05"): {
        "profile": "minimal: no explanations, no comments, T2 never executed, out-of-order execution, contains fake PII",
        "name": "**Name:** Fake Student Alpha\n\n**Roll No:** J099\n\nContact: fake.student.alpha@example.com",
        "T1": {"code": 'v=set(" ".join(docs).split())\nprint(len(v))'},
        "T2": {"code": 'from sklearn.feature_extraction.text import TfidfVectorizer\nX=TfidfVectorizer().fit_transform(docs)\nprint(X.shape)',
               "md": "-", "unexecuted": True},
        "T3": {"code": ('from sklearn.metrics.pairwise import cosine_similarity\n'
                        'from sklearn.feature_extraction.text import TfidfVectorizer\n'
                        'X=TfidfVectorizer().fit_transform(docs)\n'
                        'print(round(cosine_similarity(X[0],X[1])[0,0],4))'),
               "md": "Your answer here"},
        "shuffle_counts": True,
    },
    # ------------------------------------------------------------------ LAB_SYN2
    ("LAB_SYN2", "SYN_S01"): {
        "profile": "excellent: correct code, docstrings/comments, complete correct explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": ('from collections import Counter\n\n'
                        'def count_ngrams(sentences):\n'
                        '    """Return unigram and bigram Counters for a list of tokenised sentences."""\n'
                        '    uni, bi = Counter(), Counter()\n'
                        '    for sent in sentences:\n'
                        '        toks = sent.split()\n'
                        '        uni.update(toks)\n'
                        '        # consecutive token pairs are the bigrams\n'
                        '        bi.update(zip(toks[:-1], toks[1:]))\n'
                        '    return uni, bi\n\n'
                        'unigrams, bigrams = count_ngrams(corpus)\n'
                        'print("count(the, cat) =", bigrams[("the", "cat")])')},
        "T2": {"code": ('V = len(unigrams)  # vocabulary size, including <s> and </s>\n\n'
                        'def p_add_one(w_prev, w):\n'
                        '    """Laplace-smoothed P(w | w_prev)."""\n'
                        '    return (bigrams[(w_prev, w)] + 1) / (unigrams[w_prev] + V)\n\n'
                        'print("P(cat | the) =", round(p_add_one("the", "cat"), 4))'),
               "md": ("With maximum likelihood estimation, any bigram that was never seen in training gets zero "
                      "probability, so a whole test sentence containing it would get probability zero. Add-one "
                      "(Laplace) smoothing adds one to every count and adds the vocabulary size V to the denominator, "
                      "which redistributes a little probability mass to unseen bigrams.")},
        "T3": {"code": ('import math\n\n'
                        'def perplexity(sentence):\n'
                        '    """Perplexity of a sentence under the add-one bigram model."""\n'
                        '    toks = sentence.split()\n'
                        '    log_p = sum(math.log(p_add_one(a, b)) for a, b in zip(toks[:-1], toks[1:]))\n'
                        '    return math.exp(-log_p / (len(toks) - 1))\n\n'
                        'print("Perplexity:", round(perplexity("<s> the dog ran </s>"), 3))'),
               "md": ("Perplexity is the inverse probability of the test sentence normalised by the number of words "
                      "(the exponent of the average negative log-probability per word). It tells us how surprised "
                      "the model is; a lower perplexity means a better language model.")},
    },
    ("LAB_SYN2", "SYN_S02"): {
        "profile": "correct code with good comments/docstrings and an IPython magic, but weak explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": ('from collections import Counter\n'
                        '# unigram and bigram counters\n'
                        'unigrams, bigrams = Counter(), Counter()\n'
                        'for sent in corpus:\n'
                        '    toks = sent.split()  # "#" inside strings must not count: "a#b"\n'
                        '    unigrams.update(toks)\n'
                        '    bigrams.update(zip(toks[:-1], toks[1:]))  # pairs of adjacent tokens\n'
                        'print("count(the, cat) =", bigrams[("the", "cat")])')},
        "T2": {"code": ('%time V = len(unigrams)\n'
                        'def p_add_one(w_prev, w):\n'
                        '    """Add-one smoothed conditional probability P(w | w_prev)."""\n'
                        '    # numerator: bigram count plus one\n'
                        '    # denominator: history count plus vocabulary size\n'
                        '    return (bigrams[(w_prev, w)] + 1) / (unigrams[w_prev] + V)\n'
                        'print("P(cat | the) =", round(p_add_one("the", "cat"), 4))'),
               "md": "Smoothing is used to smooth the probabilities of the model."},
        "T3": {"code": ('import math\n'
                        'def perplexity(sentence):\n'
                        '    """Return perplexity of a whitespace-tokenised sentence."""\n'
                        '    toks = sentence.split()\n'
                        '    # sum log-probabilities to avoid underflow\n'
                        '    log_p = sum(math.log(p_add_one(a, b)) for a, b in zip(toks[:-1], toks[1:]))\n'
                        '    return math.exp(-log_p / (len(toks) - 1))\n'
                        'print("Perplexity:", round(perplexity("<s> the dog ran </s>"), 3))'),
               "md": "Perplexity is a metric used for language models."},
    },
    ("LAB_SYN2", "SYN_S03"): {
        "profile": "runtime NameError (T2), syntax error (T3), mostly correct explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": ('from collections import Counter\n'
                        'unigrams, bigrams = Counter(), Counter()\n'
                        'for sent in corpus:\n'
                        '    toks = sent.split()\n'
                        '    unigrams.update(toks)\n'
                        '    bigrams.update(zip(toks[:-1], toks[1:]))\n'
                        'print("count(the, cat) =", bigrams[("the", "cat")])')},
        "T2": {"code": ('V = len(unigrams)\n'
                        '# add-one smoothing\n'
                        'def p_add_one(w_prev, w):\n'
                        '    return (bigram_counts[(w_prev, w)] + 1) / (unigrams[w_prev] + V)\n'
                        'print("P(cat | the) =", round(p_add_one("the", "cat"), 4))'),
               "md": ("Without smoothing, unseen bigrams get zero probability. Add-one smoothing adds one to each "
                      "count so nothing is zero.")},
        "T3": {"code": ('import math\n'
                        'test = "<s> the dog ran </s>".split()\n'
                        'log_prob = sum(math.log(p_add_one(a, b)) for a, b in zip(test[:-1], test[1:])\n'
                        'print("Perplexity:", math.exp(-log_prob / (len(test) - 1)))'),
               "md": "Perplexity measures how well the model predicts the sentence. Lower perplexity is better."},
    },
    ("LAB_SYN2", "SYN_S04"): {
        "profile": "no smoothing (output mismatch), downstream ValueError, conceptually wrong explanations",
        "name": "**Name:** [synthetic]\n\n**Roll No:** [synthetic]",
        "T1": {"code": ('from collections import Counter\n'
                        'unigrams, bigrams = Counter(), Counter()\n'
                        'for sent in corpus:\n'
                        '    toks = sent.split()\n'
                        '    unigrams.update(toks)\n'
                        '    bigrams.update(zip(toks[:-1], toks[1:]))\n'
                        'print("count(the, cat) =", bigrams[("the", "cat")])')},
        "T2": {"code": ('# probability of w given the previous word\n'
                        'def p(w_prev, w):\n'
                        '    return bigrams[(w_prev, w)] / unigrams[w_prev]\n'
                        'print("P(cat | the) =", round(p("the", "cat"), 4))'),
               "md": "Smoothing makes the probabilities of frequent words larger so the model is more confident."},
        "T3": {"code": ('import math\n'
                        'test = "<s> the dog ran </s>".split()\n'
                        'lp = sum(math.log(p(a, b)) for a, b in zip(test[:-1], test[1:]))\n'
                        'print("Perplexity:", round(math.exp(-lp / (len(test) - 1)), 3))'),
               "md": "A higher perplexity means the model is better because it can handle more words."},
    },
    ("LAB_SYN2", "SYN_S05"): {
        "profile": "correct code, no comments, no explanations, contains fake PII",
        "name": "**Name:** Fake Student Beta\n\n**Roll No:** J098\n\nContact: fake.student.beta@example.com",
        "T1": {"code": ('from collections import Counter\nu,b=Counter(),Counter()\nfor s in corpus:\n'
                        '    t=s.split();u.update(t);b.update(zip(t[:-1],t[1:]))\nprint(b[("the","cat")])')},
        "T2": {"code": 'V=len(u)\np=lambda x,y:(b[(x,y)]+1)/(u[x]+V)\nprint(round(p("the","cat"),4))', "md": ""},
        "T3": {"code": ('import math\nt="<s> the dog ran </s>".split()\n'
                        'print(round(math.exp(-sum(math.log(p(x,y)) for x,y in zip(t[:-1],t[1:]))/(len(t)-1)),3))'),
               "md": ""},
    },
}

# SIMULATED annotations from three fictitious annotators (A1, A2, A3).
# code 0-8, explanation 0-8, documentation 0-4. Authored by hand to exercise the
# agreement/aggregation code, including disagreements > 1 mark.
SIM_ANNOTATIONS = {
    ("LAB_SYN1", "SYN_S01"): [(8, 8, 4), (8, 7, 4), (7, 8, 4)],
    ("LAB_SYN1", "SYN_S02"): [(7, 2, 1), (8, 2, 1), (7, 3, 0)],
    ("LAB_SYN1", "SYN_S03"): [(3, 5, 2), (4, 5, 3), (2, 6, 2)],
    ("LAB_SYN1", "SYN_S04"): [(2, 1, 2), (3, 1, 2), (2, 2, 1)],
    ("LAB_SYN1", "SYN_S05"): [(5, 0, 0), (6, 0, 0), (4, 0, 1)],
    ("LAB_SYN2", "SYN_S01"): [(8, 8, 4), (8, 8, 3), (8, 7, 4)],
    ("LAB_SYN2", "SYN_S02"): [(8, 2, 4), (7, 3, 4), (8, 2, 3)],
    ("LAB_SYN2", "SYN_S03"): [(3, 6, 2), (2, 6, 2), (3, 5, 2)],
    ("LAB_SYN2", "SYN_S04"): [(2, 1, 2), (2, 0, 2), (1, 1, 2)],
    ("LAB_SYN2", "SYN_S05"): [(8, 0, 0), (7, 0, 0), (8, 0, 0)],
}


# ------------------------------------------------------------------ builders
def _md(s):
    return nbformat.v4.new_markdown_cell(s)


def _code(s):
    return nbformat.v4.new_code_cell(s)


def build_cells(lab: dict, fill: dict | None, name_text: str) -> list:
    """fill=None -> blank template; otherwise task_id -> {"code","md"}."""
    cells = [_md(lab["title"] + "\n\n" + BANNER), _md(name_text), _code(lab["given_code"])]
    for t in lab["tasks"]:
        cells.append(_md(t["prompt"]))
        if fill is not None:
            c = _code(fill[t["id"]]["code"])
            if fill[t["id"]].get("unexecuted"):
                c.metadata["_unexecuted"] = True
            cells.append(c)
        else:
            cells.append(_code("# your code here"))
        if t["explain"]:
            cells.append(_md(t["explain"]))
            cells.append(_md(fill[t["id"]].get("md", "") if fill is not None else "Your answer here"))
    return cells


def new_nb(cells) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["synthetic_development_data"] = True
    return nb


def execute(nb, workdir: Path) -> nbformat.NotebookNode:
    skip = [i for i, c in enumerate(nb.cells) if c.metadata.get("_unexecuted")]
    NotebookClient(nb, timeout=120, kernel_name="python3", allow_errors=True,
                   resources={"metadata": {"path": str(workdir)}}).execute()
    for i in skip:  # simulate a cell the student never ran
        nb.cells[i].outputs = []
        nb.cells[i].execution_count = None
    for c in nb.cells:
        c.metadata.pop("_unexecuted", None)
        c.metadata.pop("execution", None)
    return nb


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for lab_id, lab in LABS.items():
        ref_dir = OUT / "references" / lab_id
        raw_dir = OUT / "raw" / lab_id
        ref_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # template (handout)
        nbformat.write(new_nb(build_cells(lab, None, NAME_TEMPLATE)), ref_dir / "template.ipynb")

        # reference solution: executed, used to derive expected outputs
        ref_fill = {t["id"]: {"code": lab["reference"][t["id"]],
                              "md": lab["ref_expl"].get(t["id"], {}).get("text", "")} for t in lab["tasks"]}
        ref_nb = execute(new_nb(build_cells(lab, ref_fill, NAME_TEMPLATE)), ref_dir)
        nbformat.write(ref_nb, ref_dir / "reference_solution.ipynb")

        expected = {}
        for t in lab["tasks"]:
            for c in ref_nb.cells:
                if c.cell_type == "code" and c.source == lab["reference"][t["id"]]:
                    text = "".join(o.get("text", "") for o in c.outputs if o.output_type == "stream").strip()
                    if not text or any(o.output_type == "error" for o in c.outputs):
                        sys.exit(f"reference {lab_id}/{t['id']} produced no output or an error")
                    expected[t["id"]] = {"value": text, "match": "numeric", "tol": 1e-3}
        (ref_dir / "expected_outputs.json").write_text(json.dumps(
            {"_status": "SYNTHETIC_DEVELOPMENT_DATA", "tasks": expected}, indent=2))
        (ref_dir / "reference_explanations.json").write_text(json.dumps(
            {"_status": "SYNTHETIC_DEVELOPMENT_DATA", "tasks": lab["ref_expl"]}, indent=2))
        (ref_dir / "rubric.json").write_text(json.dumps(
            {"_status": "SYNTHETIC_DEVELOPMENT_DATA (max marks from Week 1 proposal example)",
             "code": 8, "explanation": 8, "documentation": 4}, indent=2))

        # student notebooks
        for (lid, sid), spec in STUDENTS.items():
            if lid != lab_id:
                continue
            fill = {t["id"]: spec[t["id"]] for t in lab["tasks"]}
            nb = execute(new_nb(build_cells(lab, fill, spec["name"])), raw_dir)
            if spec.get("shuffle_counts"):  # simulate out-of-order execution
                code_cells = [c for c in nb.cells if c.cell_type == "code" and c.execution_count]
                counts = [c.execution_count for c in code_cells]
                for c, n in zip(code_cells, [counts[0], counts[-1] + 3, counts[1]] + counts[3:]):
                    c.execution_count = n
                    for o in c.outputs:
                        if "execution_count" in o:
                            o["execution_count"] = n
            fname = f"{sid}_{lab_id}.ipynb"
            nbformat.write(nb, raw_dir / fname)
            manifest_rows.append({
                "raw_path": f"raw/{lab_id}/{fname}", "assignment_id": lab_id, "group_key": sid,
                "source": "synthetic", "license": "team-authored synthetic fixture",
                "synthetic": True, "profile": spec["profile"]})
            print("wrote", lab_id, fname)

    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    (OUT / "annotations").mkdir(exist_ok=True)
    with open(OUT / "annotations" / "raw_annotations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["raw_path", "annotator", "code", "explanation", "documentation", "status"])
        for (lab_id, sid), rows in SIM_ANNOTATIONS.items():
            for a, (c, e, d) in zip(["A1", "A2", "A3"], rows):
                w.writerow([f"raw/{lab_id}/{sid}_{lab_id}.ipynb", a, c, e, d, "SIMULATED"])

    (OUT / "class_list.txt").write_text(
        "# SYNTHETIC: fictitious names only, used to test the anonymiser\nFake Student Alpha\nFake Student Beta\n")
    (OUT / "README_SYNTHETIC.md").write_text(
        "# SYNTHETIC DEVELOPMENT DATA\n\nEverything in this folder was authored by the team to test the "
        "Week 2 pipeline. It is not the project dataset and not real student work. Annotations are SIMULATED. "
        "Metrics computed on it demonstrate that the code runs; they are not evidence of evaluator performance.\n")


if __name__ == "__main__":
    main()
