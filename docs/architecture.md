# Where the notebook lab evaluator sits

Group 3. Rolls J013, J015, J016, J017, J018.

The picture in `docs/architecture.png` uses the same boxes as this page. The shared AI University picture in the course handout is a conceptual layout. This component fills the Labs box under Academic Services. It uses semantic similarity and an evaluation service. The support signal is the handoff toward Student Services.

## Vertical flow

```
Input
  one .ipynb file and a lab id
  the answer card stays in the package
        |
        v
Preprocessing
  anonymise, parse, align tasks, drop template cells
        |
        v
NLP model
  TF-IDF plus the frozen MiniLM encoder, then Ridge
  served variant: week3_combined
        |
        v
Post-processing
  marks, missing ideas, feedback sentence,
  confidence, weak-explanation flag
        |
        v
API
  POST /evaluate-notebook on this machine, port 8000
        |
        v
Application
  the demo page at GET /
  and POST /support-signal for a student-support reader
```

## Place in the AI University picture

```
AI University platform
  Academic Services
    Labs  <-- this component
  Shared NLP services used here
    semantic similarity
    evaluation
  Student Services
    Early Warning  <-- reads the support signal
```

Another group calls `POST /evaluate-notebook` with a notebook and a lab id. A student-support reader calls `POST /support-signal` with the JSON that came back. That second call does not score the notebook again. This hand-in does not call another group's server. Their URL is not in this package.
