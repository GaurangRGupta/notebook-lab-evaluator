# Week 6 note

Group 3. Rolls J013, J015, J016, J017, J018.

The usage manual is `docs/manual.md`. The same manual in Word is `results/week6/Group3_Week6_Asset.docx`. It has the function signatures, the arguments, the return fields, and runnable examples. This page is the short cover.

Send one `.ipynb` file and a lab id. Leave the answer card in the package. The student's code is not executed.

`LAB_SYN2` is the language-model lab (add-one smoothing and perplexity). `LAB_SYN1` is the TF-IDF lab. The example file for the first is `examples/language_model_lab.ipynb`.

```bash
python -m src.api.app
curl.exe -s -X POST http://127.0.0.1:8000/evaluate-notebook ^
  -F "assignment_id=LAB_SYN2" ^
  -F "file=@examples/language_model_lab.ipynb"
```

For that file the live marks are code 7.87, explanation 5.13, documentation 2.14, overall 15.14 out of 20. The saved output matches. The write-up for task T3 does not cover "inverse probability" or "surprise", so the flag is true. Explanation confidence is 0.986. Code and documentation confidence are null. Overall confidence is 0.986.

The manual also scores `examples/tfidf_lab.ipynb` (overall 18.17, flag false) and `examples/language_model_weak.ipynb` (overall 7.28, output match 0, with syntax errors in the file).

The two cards in this package were prepared for the project. The marks follow that rubric. A new lab of the same shape is a new card folder. Scoring one notebook does not retrain the model.
