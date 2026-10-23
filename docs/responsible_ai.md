# Responsible AI

Group 3, Jupyter Notebook Lab Evaluator. Rolls J013, J015, J016, J017, J018.

This page states what the current evaluator can be trusted to do, and what a person still has to decide. The numbers below are the ones already recorded for the synthetic set. No new experiment was run for this page.

## Privacy

The scrubber removes roll-number patterns before a notebook is scored. The salt that builds the student hash stays outside the repository. The server listens on this machine only, at `127.0.0.1:8000`. With the optional Grok switch off, notebook text stays on this machine. `data/class/` has the folder layout and no student notebooks. The two answer cards in the package were written for this project. A faculty notebook, when one is provided, replaces the card for that lab. Names and rolls from a real class are not part of this hand-in.

## Bias

Closeness of wording is not the same thing as a correct explanation. On the robustness sentences, the frozen MiniLM cosine is 0.424 for a correct paraphrase and 0.861 for a wrong answer that reuses the reference words. A high cosine here means the words are close to the reference. Seventeen of the twenty test explanations also appear verbatim in the training notebooks, so the test is easier than a set of new wording would be. The code correlation is partly circular: the code label on these notebooks is built from the same syntax, execution, and output signals the features read. The concept threshold stays 0.275. It was set on the authored calibration pairs and was not moved after the robustness numbers were seen.

The marks on the 60-notebook set are rule labels. The annotator id is `RULE`. They are not faculty marks. Agreement with a human grader has not been measured.

## Hallucination and over-confidence

A mark can be printed to two decimal places and still be a poor match to the label. Confidence is the agreement between the Week 2 TF-IDF score and the Week 3 combined score on the explanation features. The method text on every response says this is not a calibrated probability. Code confidence and documentation confidence are null, because those two variants share a feature list and their agreement would always be 1. Overall confidence repeats the explanation value.

The optional Grok grader, when someone turns it on, is told that the stored reference notebook is the only standard. It is off in this hand-in. Its marks are not the served marks. A missing key or a failed call leaves the Ridge marks in place.

## Human oversight

The instructor sets the mark. The component suggests a score and a short feedback sentence. `flag_output_ok_explanation_weak` is true when every saved output matches and the write-up still misses a reference idea. That notebook should be read by a person. `POST /support-signal` turns the same facts into `needs_human_review` for a student-support reader. The reader does not replace the instructor.

## Security

Student code is not executed. The score uses the outputs saved in the file. A notebook with no saved output cannot show a passing output match. The server is not bound to an open address. Do not put it on an open address if the notebooks are student work.

## Limits

The comparison of Week 3 with Week 2 uses 12 synthetic test notebooks. Explanation MAE is 1.346 for Week 3 and 1.512 for Week 2. The 95 percent interval on the difference includes zero. Week 3 is not shown to beat Week 2. Code MAE and documentation MAE are the same for both variants. The Week 4 NLI comparison was not scored. A new lab of the same shape needs a new five-file card. A different notebook shape, or a faculty mark scale, is outside what these fitted models have seen.
