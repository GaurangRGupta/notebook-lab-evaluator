# Week 3 component: semantic similarity of student explanations

**Module:** `src/models/semantic_similarity.py` · **Status:** Version 1, development (synthetic data only)

## What it does
Encodes each task's student explanation and the reference explanation with a pretrained sentence encoder and
returns their cosine similarity, plus a concept matcher that counts a reference concept as covered when it is
matched lexically (Week 2 rule) or when its best phrasing is semantically close to a student sentence.

## Model
| | |
|---|---|
| Name | `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) |
| Size | ~22M parameters, ~90 MB, CPU |
| Output | 384-d, L2-normalised (cosine = dot product) |
| Window | 256 word-piece tokens; longer text split into sentence windows, embeddings averaged |
| Training on our data | **none**: frozen feature extractor (`eval()`, `inference_mode()`, `requires_grad=False`) |
| Revision | resolved commit hash is logged in `metrics.json → semantic_model.revision`; pin it in the config |

## Features produced
| Feature | Definition | Why |
|---|---|---|
| `sem_cos_mean` | mean over tasks of cos(student, reference); 0 if unanswered | overall conceptual closeness |
| `sem_cos_max` | max over tasks | mirrors `tfidf_cos_max` for a like-for-like comparison |
| `sem_concept_coverage` | share of reference concepts matched lexically or semantically | Week 1 concept task (d) without KeyBERT |

## Leakage controls
* Pretrained and frozen: no parameter is fitted on our notebooks, so embedding a validation/test notebook cannot leak.
* Features are per-text and stateless (tested).
* TF-IDF, imputation and Ridge: fitted on train; alpha on validation.
* Concept threshold: chosen on the tune half of `data/synthetic/week3/concept_pairs.csv` (authored; tune and eval use different concepts), never on notebooks.
* Test split: scored once, after variants and settings were frozen.

## Known failure modes (see Week 3 report, section 12)
* Negation and inverted relationships ("higher perplexity is better") stay close to the correct statement.
* Wrong algorithms described with the reference vocabulary (greedy vs Viterbi) can score high.
* Verbatim copies of the reference score maximally: similarity cannot detect copying.
* Topic overlap is not correctness: similarity is a feature for the scorer, never a mark by itself.

## Reuse
```python
from src.models.semantic_similarity import SentenceTransformerEncoder, SemanticSimilarity
sem = SemanticSimilarity(SentenceTransformerEncoder(), concept_threshold=0.5)
sem.similarity(student_text, reference_text)
sem.concept_coverage(student_text, [{"term": "inverse document frequency", "variants": ["idf"]}])
```
`HashingStubEncoder` is a lexical, test-only stand-in. Never report its outputs as semantic results.
