"""SYNTHETIC DEVELOPMENT SET v2 (Week 5). Design and label rules: docs/week5_protocol.md.

!!! Synthetic, team-authored notebooks. Labels are RULE-DERIVED from planted properties. !!!
!!! Not student work; results on it are not evidence about real submissions.             !!!
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
import re
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "synthetic"
OUT = ROOT / "data" / "synthetic_v2"
SEED = 2026

spec = importlib.util.spec_from_file_location("fx", ROOT / "scripts" / "make_synthetic_fixtures.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)

# ----------------------------------------------------------------------------- code variants
# "#! " marks a comment line kept according to the student's comment level.
# TOKENIZE(x)/TOKENS(x) become a helper call or inline code depending on the helper-function trait.
COUNTS = ("from collections import Counter\nU, B = Counter(), Counter()\nfor s in corpus:\n"
          "    t = TOKENS(s)\n    U.update(t)\n    B.update(zip(t[:-1], t[1:]))\n")
CODE = {
    "LAB_SYN1": {
        "T1": {
            "correct_a": "#! build the vocabulary from lowercase whitespace tokens\ntokens = [TOKENIZE(d) for d in docs]\n#! unique tokens across all documents\nvocab = sorted({t for doc in tokens for t in doc})\nprint(\"Vocabulary size:\", len(vocab))",
            "correct_b": "vocab = set()\nfor d in docs:\n    #! add this document's tokens to the vocabulary\n    vocab.update(TOKENIZE(d))\nprint(\"Vocabulary size:\", len(vocab))",
            "syntax": "vocab = set()\nfor d in docs\n    #! add tokens\n    vocab.update(TOKENIZE(d))\nprint(\"Vocabulary size:\", len(vocab))",
            "runtime": "#! collect unique tokens\nvocab = {t for d in documents for t in TOKENIZE(d)}\nprint(\"Vocabulary size:\", len(vocab))",
            "wrong": "#! count the words\nwords = []\nfor d in docs:\n    words.extend(TOKENIZE(d))\nprint(\"Vocabulary size:\", len(words))",
        },
        "T2": {
            "correct_a": "from sklearn.feature_extraction.text import TfidfVectorizer\n#! fit TF-IDF on the corpus: rows are documents, columns are terms\nvectorizer = TfidfVectorizer()\nX = vectorizer.fit_transform(docs)\nprint(\"TF-IDF matrix shape:\", X.shape)",
            "correct_b": "from sklearn.feature_extraction.text import TfidfVectorizer as TV\n#! document-term matrix with tf-idf weights\nX = TV().fit_transform(docs)\nX.shape",
            "syntax": "from sklearn.feature_extraction.text import TfidfVectorizer\n#! build the matrix\nX = TfidfVectorizer().fit_transform(docs\nprint(\"TF-IDF matrix shape:\", X.shape)",
            "runtime": "from sklearn.feature_extraction.text import TfidfVectorizer\n#! build the matrix\nX = TfidfVectorizer().fit_transform(doc)\nprint(\"TF-IDF matrix shape:\", X.shape)",
            "wrong": "from sklearn.feature_extraction.text import CountVectorizer\n#! term counts\nX = CountVectorizer().fit_transform(docs)\nprint(\"TF-IDF matrix shape:\", X.shape)",
        },
        "T3": {
            "correct_a": "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.metrics.pairwise import cosine_similarity\n#! recompute tf-idf so this cell is self-contained\nM = TfidfVectorizer().fit_transform(docs)\n#! cosine between the first two documents\nprint(\"Cosine similarity:\", round(cosine_similarity(M[0], M[1])[0, 0], 4))",
            "correct_b": "import numpy as np\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nA = TfidfVectorizer().fit_transform(docs).toarray()\n#! dot product divided by the product of the norms\nsim = A[0] @ A[1] / (np.linalg.norm(A[0]) * np.linalg.norm(A[1]))\nprint(\"Cosine similarity:\", round(sim, 4))",
            "syntax": "from sklearn.metrics.pairwise import cosine_similarity\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nM = TfidfVectorizer().fit_transform(docs)\nprint(\"Cosine similarity:\" round(cosine_similarity(M[0], M[1])[0, 0], 4))",
            "runtime": "from sklearn.metrics.pairwise import cosine_similarity\n#! compare documents 0 and 1\nprint(\"Cosine similarity:\", round(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0], 4))",
            "wrong": "from sklearn.feature_extraction.text import CountVectorizer\nimport numpy as np\n#! similarity as a dot product of counts\nC = CountVectorizer().fit_transform(docs).toarray()\nprint(\"Cosine similarity:\", np.dot(C[0], C[1]))",
        },
    },
    "LAB_SYN2": {
        "T1": {
            "correct_a": "from collections import Counter\n#! unigram and bigram counts\nunigrams, bigrams = Counter(), Counter()\nfor sent in corpus:\n    toks = TOKENS(sent)\n    unigrams.update(toks)\n    #! adjacent token pairs are bigrams\n    bigrams.update(zip(toks[:-1], toks[1:]))\nprint(\"count(the, cat) =\", bigrams[(\"the\", \"cat\")])",
            "correct_b": "#! count bigrams with a plain dictionary\ncounts = {}\nfor sent in corpus:\n    toks = TOKENS(sent)\n    for a, b in zip(toks, toks[1:]):\n        counts[(a, b)] = counts.get((a, b), 0) + 1\nprint(\"count(the, cat) =\", counts[(\"the\", \"cat\")])",
            "syntax": "counts = {}\nfor sent in corpus\n    toks = TOKENS(sent)\n    for a, b in zip(toks, toks[1:]):\n        counts[(a, b)] = counts.get((a, b), 0) + 1\nprint(\"count(the, cat) =\", counts[(\"the\", \"cat\")])",
            "runtime": "#! look up the bigram count\nprint(\"count(the, cat) =\", bigram_counts[(\"the\", \"cat\")])",
            "wrong": "from collections import Counter\n#! pairs of tokens\nbigrams = Counter()\nfor sent in corpus:\n    toks = TOKENS(sent)\n    bigrams.update(zip(toks[:-2], toks[2:]))\nprint(\"count(the, cat) =\", bigrams[(\"the\", \"cat\")])",
        },
        "T2": {
            "correct_a": COUNTS + "#! add-one smoothing: (count + 1) / (history count + V)\nV = len(U)\ndef p_add_one(w_prev, w):\n    return (B[(w_prev, w)] + 1) / (U[w_prev] + V)\nprint(\"P(cat | the) =\", round(p_add_one(\"the\", \"cat\"), 4))",
            "correct_b": COUNTS + "#! Laplace estimate for one bigram\np = (B[(\"the\", \"cat\")] + 1) / (U[\"the\"] + len(U))\nprint(\"P(cat | the) =\", round(p, 4))",
            "syntax": COUNTS + "V = len(U)\np = (B[(\"the\", \"cat\")] + 1 / (U[\"the\"] + V)\nprint(\"P(cat | the) =\", round(p, 4))",
            "runtime": COUNTS + "#! smoothed probability\np = (B[(\"the\", \"cat\")] + 1) / (U[\"the\"] + vocab_size)\nprint(\"P(cat | the) =\", round(p, 4))",
            "wrong": COUNTS + "#! probability of cat given the\np = B[(\"the\", \"cat\")] / U[\"the\"]\nprint(\"P(cat | the) =\", round(p, 4))",
        },
        "T3": {
            "correct_a": COUNTS + "import math\nV = len(U)\n#! perplexity = exp of the average negative log-probability per bigram\ntest = \"<s> the dog ran </s>\".split()\nlp = sum(math.log((B[(a, b)] + 1) / (U[a] + V)) for a, b in zip(test[:-1], test[1:]))\nprint(\"Perplexity:\", round(math.exp(-lp / (len(test) - 1)), 3))",
            "correct_b": COUNTS + "V = len(U)\ntest = \"<s> the dog ran </s>\".split()\nprob = 1.0\nfor a, b in zip(test[:-1], test[1:]):\n    #! multiply smoothed bigram probabilities\n    prob *= (B[(a, b)] + 1) / (U[a] + V)\nprint(\"Perplexity:\", round(prob ** (-1 / (len(test) - 1)), 3))",
            "syntax": COUNTS + "import math\nV = len(U)\ntest = \"<s> the dog ran </s>\".split()\nlp = sum(math.log((B[(a, b)] + 1) / (U[a] + V)) for a, b in zip(test[:-1], test[1:])\nprint(\"Perplexity:\", round(math.exp(-lp / (len(test) - 1)), 3))",
            "runtime": COUNTS + "V = len(U)\n#! perplexity of the test sentence\ntest = \"<s> the dog ran </s>\".split()\nlp = sum(math.log((B[(a, b)] + 1) / (U[a] + V)) for a, b in zip(test[:-1], test[1:]))\nprint(\"Perplexity:\", round(math.exp(-lp / (len(test) - 1)), 3))",
            "wrong": COUNTS + "import math\nV = len(U)\n#! normalise by the number of tokens\ntest = \"<s> the dog ran </s>\".split()\nlp = sum(math.log((B[(a, b)] + 1) / (U[a] + V)) for a, b in zip(test[:-1], test[1:]))\nprint(\"Perplexity:\", round(math.exp(-lp / len(test)), 3))",
        },
    },
}
HELPER = {"LAB_SYN1": ("tokenize", "text", "Lowercase a text and split it on whitespace.", "text.lower().split()", r"TOKENIZE\((\w+)\)", r"\1.lower().split()"),
          "LAB_SYN2": ("tokens", "sentence", "Split a sentence into tokens on whitespace.", "sentence.split()", r"TOKENS\((\w+)\)", r"\1.split()")}

# ----------------------------------------------------------------------------- explanation pools
EXPL = {
    ("LAB_SYN1", "T2"): {
        "correct": ["TF-IDF multiplies the term frequency of a word in a document by its inverse document frequency, the log of the number of documents divided by the number of documents containing the word. Words like 'the' that appear in most documents get a low IDF and are down-weighted, while rare words get a higher weight.",
                    "The TF part is how often the term occurs in the document and the IDF part is the logarithm of N over the document frequency. Common words such as 'the' occur in almost every document, so their inverse document frequency is close to zero and they receive less weight than rare, discriminative words."],
        "paraphrase": ["Each word is scored by how many times it shows up in a text, scaled down when that word turns up in lots of other texts. So filler words found everywhere end up almost worthless, while words that are specific to a few texts stand out.",
                       "The score rewards a word for being frequent inside one text but punishes it for being widespread across the collection; a word present in every text contributes essentially nothing, whereas an unusual word that characterises one text gets emphasised."],
        "partial": ["TF-IDF counts how often each word appears in a document (the term frequency) and uses these counts as the features.",
                    "It gives common words like 'the' a smaller weight than other words."],
        "vague": ["TF-IDF is a very popular and important technique in NLP that is used to convert text into numbers.",
                  "This method helps the computer understand which words matter so the model works better."],
        "wrong": ["TF-IDF gives the highest weight to words that appear in most documents, because the inverse document frequency grows with the number of documents containing the word.",
                  "The term frequency is divided by the document length and words like 'the' receive the largest weight because they are the most frequent terms in every document."],
        "offtopic": ["I used the scikit-learn documentation and Stack Overflow to finish this task before the deadline."],
    },
    ("LAB_SYN1", "T3"): {
        "correct": ["Cosine similarity is the dot product of the two vectors divided by the product of their norms, i.e. the cosine of the angle between them. Because it ignores vector magnitude, documents of different length on the same topic can still be similar, which Euclidean distance would penalise.",
                    "We compute the cosine of the angle between the two TF-IDF vectors: their dot product over the product of their lengths. It depends only on direction, not magnitude, so it is insensitive to document length, unlike Euclidean distance."],
        "paraphrase": ["It looks at whether the two document vectors point the same way rather than how far apart their tips are, so a long essay and a short note about the same subject still come out as close.",
                       "Multiply matching entries, add them up and divide by both vectors' sizes; what is left only reflects orientation, so making a text longer by repeating it does not change the result, whereas a straight-line distance would grow."],
        "partial": ["Cosine similarity measures the angle between two vectors.",
                    "It is the dot product of the two vectors, so documents sharing more words get a bigger value."],
        "vague": ["Cosine similarity tells us how similar two documents are. It is better than other measures.",
                  "This is a standard way of comparing documents that works well in practice."],
        "wrong": ["Cosine similarity divides the Euclidean distance by the dot product, so it depends heavily on the length of the documents.",
                  "Cosine similarity is the Euclidean distance between the two vectors; a larger value means the documents are less similar, and longer documents always score higher."],
        "offtopic": ["The cat sat on the mat is my favourite sentence in this corpus because it rhymes."],
    },
    ("LAB_SYN2", "T2"): {
        "correct": ["Maximum likelihood estimation gives zero probability to any bigram that never appeared in the training data, so a sentence containing it gets probability zero. Add-one (Laplace) smoothing adds one to every bigram count and the vocabulary size V to the denominator so that no probability is zero.",
                    "Without smoothing, unseen bigrams get zero probability. Laplace smoothing adds one to each count and adds the vocabulary size to the history count, redistributing some probability mass to unseen events."],
        "paraphrase": ["If a word pair never showed up in the training text, the plain estimate says it is impossible, which wipes out the probability of any sentence using it. We pretend every pair was seen one extra time and enlarge the denominator to match, so nothing is ruled out completely.",
                       "Counts taken straight from the corpus treat anything unobserved as having no chance at all; bumping every count up by one and correcting the total spreads a little belief onto combinations we simply have not met yet."],
        "partial": ["Add-one smoothing adds one to every bigram count.",
                    "Smoothing is needed because some bigrams are unseen in the training data."],
        "vague": ["Smoothing is used to smooth the probabilities of the model.",
                  "Smoothing makes the language model better and more accurate."],
        "wrong": ["Add-one smoothing adds one to the vocabulary size so that frequent bigrams get a higher probability than in maximum likelihood estimation.",
                  "Smoothing removes rare bigrams from the training data so the probability of unseen bigrams stays zero and the model is more confident."],
        "offtopic": ["This lab took me about two hours and I think the instructions were clear."],
    },
    ("LAB_SYN2", "T3"): {
        "correct": ["Perplexity is the inverse probability of the test sentence normalised by the number of words, i.e. the exponent of the average negative log-probability per word. A lower perplexity means the model assigns higher probability to the test data and is a better language model.",
                    "Perplexity is exp of the average negative log probability per token. It measures how surprised the model is by the test text; lower perplexity is better."],
        "paraphrase": ["It is two raised to the per-token cross-entropy, so the less surprised the model is by held-out text, the smaller the value and the better the model.",
                       "Think of it as the effective number of equally likely choices the model is torn between at each word; a model that is rarely caught off guard by new text narrows this down and is preferred."],
        "partial": ["Perplexity is computed from the probability of the test sentence.",
                    "A lower perplexity is better."],
        "vague": ["Perplexity is a metric used for language models.",
                  "Perplexity shows how good the model is."],
        "wrong": ["Perplexity is the inverse probability of the test sentence normalised by the number of words; a higher perplexity means a better language model.",
                  "Perplexity is the number of words in the test sentence multiplied by its probability, so longer sentences always give a better model."],
        "offtopic": ["I would like to try a neural language model in the next lab."],
    },
}
EXPL_POINTS = {"correct": 4, "paraphrase": 4, "partial": 2, "vague": 1, "wrong": 0, "offtopic": 0, "empty": 0}
CODE_POINTS = {"correct_a": 1.0, "correct_b": 1.0, "unexecuted": 0.5, "syntax": 0.0, "runtime": 0.0, "wrong": 0.0}

CODE_DIST = {"high": {"correct_a": .45, "correct_b": .40, "unexecuted": .05, "wrong": .05, "runtime": .05},
             "mid": {"correct_a": .30, "correct_b": .25, "unexecuted": .10, "wrong": .15, "runtime": .10, "syntax": .10},
             "low": {"correct_a": .15, "correct_b": .10, "unexecuted": .10, "wrong": .25, "runtime": .20, "syntax": .20}}
EXPL_DIST = {"high": {"correct": .45, "paraphrase": .35, "partial": .15, "vague": .05},
             "mid": {"correct": .20, "paraphrase": .20, "partial": .25, "vague": .20, "wrong": .15},
             "low": {"partial": .15, "vague": .25, "wrong": .25, "offtopic": .10, "empty": .25}}


def pick(rng, dist):
    r, acc = rng.random(), 0.0
    for k, p in dist.items():
        acc += p
        if r <= acc:
            return k
    return list(dist)[-1]


def apply_traits(lab, code, comment_level, func):
    name, arg, doc, body, pat, inline = HELPER[lab]
    code = re.sub(pat, (name + r"(\1)") if func != "none" else inline, code)
    lines, kept, seen = code.split("\n"), [], 0
    for l in lines:
        if l.lstrip().startswith("#! "):
            seen += 1
            if comment_level == 2 or (comment_level == 1 and seen == 1):
                kept.append(l.replace("#! ", "# ", 1))
        else:
            kept.append(l)
    code = "\n".join(kept)
    return code


def helper_prefix(lab, func):
    name, arg, doc, body, *_ = HELPER[lab]
    if func == "none":
        return ""
    ds = f'    """{doc}"""\n' if func == "doc" else ""
    return f"def {name}({arg}):\n{ds}    return {body}\n\n"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "raw").mkdir(parents=True)
    shutil.copytree(V1 / "references", OUT / "references")
    shutil.copytree(V1 / "week3", OUT / "week3")
    shutil.copytree(V1 / "week4", OUT / "week4")
    rng = random.Random(SEED)
    students = []
    for i in range(1, 31):
        students.append({"sid": f"V2_S{i:02d}",
                         "code_skill": rng.choices(["high", "mid", "low"], [.40, .35, .25])[0],
                         "expl_skill": rng.choices(["high", "mid", "low"], [.40, .35, .25])[0],
                         "comments": rng.choice([0, 1, 2]), "func": rng.choice(["none", "nodoc", "doc"])})
    pii = {"V2_S07": "Fake Student Gamma", "V2_S19": "Fake Student Delta"}
    design, man, ann = [], [], []
    for st in students:
        for lab_id, lab in fx.LABS.items():
            fill, d = {}, {"raw_path": None, "assignment_id": lab_id, "student": st["sid"], **{k: st[k] for k in ("code_skill", "expl_skill", "comments", "func")}}
            pts = 0.0
            for t in lab["tasks"]:
                tid = t["id"]
                v = pick(rng, CODE_DIST[st["code_skill"]])
                src = CODE[lab_id][tid]["correct_a" if v == "unexecuted" else v]
                code = apply_traits(lab_id, src, st["comments"], st["func"])
                fill[tid] = {"code": code, "unexecuted": v == "unexecuted"}
                d[f"code_{tid}"] = v
                pts += CODE_POINTS[v]
                if t["explain"]:
                    k = pick(rng, EXPL_DIST[st["expl_skill"]])
                    txt = rng.choice(["", "Your answer here"]) if k == "empty" else rng.choice(EXPL[(lab_id, tid)][k])
                    fill[tid]["md"] = txt
                    d[f"expl_{tid}"] = k
            expl = sum(EXPL_POINTS[d[f"expl_{t}"]] for t in ("T2", "T3"))
            any_md = any(d[f"expl_{t}"] != "empty" for t in ("T2", "T3"))
            code_mark = int((8 * pts / 3) + 0.5)
            doc = st["comments"] + (1 if st["func"] == "doc" else 0) + (1 if any_md else 0)
            name_cell = (f"**Name:** {pii[st['sid']]}\n\n**Roll No:** J0{90 + int(st['sid'][-2:]) % 9}"
                         if st["sid"] in pii else "**Name:** [synthetic]\n\n**Roll No:** [synthetic]")
            cells = fx.build_cells(lab, fill, name_cell)
            # helper function in its OWN cell inside Task 1 so a broken T1 cell cannot cascade
            # (protocol: tasks are self-contained); fixed before any model was run on v2
            if st["func"] != "none":
                i = next(j for j, c in enumerate(cells) if c.cell_type == "code" and c.source == fill["T1"]["code"])
                cells.insert(i, fx._code(helper_prefix(lab_id, st["func"]).rstrip()))
            nb = fx.new_nb(cells)
            nb.metadata["synthetic_dataset"] = "v2"
            for c in nb.cells:
                if c.cell_type == "code" and any(f["unexecuted"] and c.source == f["code"] for f in fill.values()):
                    c.metadata["_unexecuted"] = True
            raw_dir = OUT / "raw" / lab_id
            raw_dir.mkdir(parents=True, exist_ok=True)
            nb = fx.execute(nb, raw_dir)
            fname = f"{st['sid']}_{lab_id}.ipynb"
            nbformat.write(nb, raw_dir / fname)
            rp = f"raw/{lab_id}/{fname}"
            d.update({"raw_path": rp, "label_code": code_mark, "label_explanation": expl, "label_documentation": doc})
            design.append(d)
            man.append({"raw_path": rp, "assignment_id": lab_id, "group_key": st["sid"], "source": "synthetic_v2",
                        "license": "team-authored synthetic fixture", "synthetic": True,
                        "profile": f"code={st['code_skill']} expl={st['expl_skill']} comments={st['comments']} func={st['func']}"})
            ann.append([rp, "RULE", code_mark, expl, doc, "RULE_DERIVED"])
            print("wrote", rp, code_mark, expl, doc)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader(); w.writerows(man)
    with open(OUT / "design.csv", "w", newline="") as f:
        cols = sorted({k for d in design for k in d}, key=lambda k: (k != "raw_path", k))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(design)
    (OUT / "annotations").mkdir()
    with open(OUT / "annotations" / "raw_annotations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["raw_path", "annotator", "code", "explanation", "documentation", "status"])
        w.writerows(ann)
    (OUT / "class_list.txt").write_text("# SYNTHETIC: fictitious names only\n" + "\n".join(pii.values()) + "\n")
    (OUT / "README_SYNTHETIC.md").write_text(
        "# SYNTHETIC DEVELOPMENT SET v2\n\nTeam-authored notebooks, labels RULE-DERIVED from planted properties "
        "(docs/week5_protocol.md). Not student work. Results are not evidence about real submissions.\n")
    json.dump({"seed": SEED, "n_students": len(students), "n_notebooks": len(man)}, open(OUT / "generation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
