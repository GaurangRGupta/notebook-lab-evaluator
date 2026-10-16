# Group 3: Jupyter Notebook Lab Evaluator

Reusable NLP asset for the AI University.

## 1. Problem

A Jupyter notebook holds code, the output of that code, and writing that explains the work. Automatic graders usually check only that the code runs. This component also checks whether the student understood the method. A new lab is a new description passed in with the notebook.

## 2. Users

A lab instructor calls it, or a grading tool acting for that instructor. It suggests scores and short feedback. The instructor sets the real mark. A second tool, such as one that flags students who need help, can read the same result. Students see feedback only through the instructor's tool.

## 3. Input

The caller sends a notebook file (`.ipynb`) and a lab description. The description is data: the question, a short model explanation, the ideas the writing should mention, and the output the notebook should show.

## 4. Output

One JSON object: a code score from the saved output and errors; an explanation score; a documentation score; whether the saved output matches the lab; an overall score and the maximum; a confidence value; short feedback; and a flag when the output is right but the writing does not describe the code. Other groups call a Python function. Once packaged, that call is POST /evaluate-notebook.

## 5. NLP task

Two language jobs. Grade the markdown and comments against the model explanation and the idea list. Check whether that writing agrees with the code. The notebook is split and the saved output is read first. Code is judged from that saved output.

## 6. Dataset

No public dataset assigns scores to the writing in lab notebooks. The group will label notebooks for at least two short NLP or ML labs. Each item is one notebook and one lab description. Labels are the explanation score, the documentation score, whether the output is right, and whether the explanation fails to match the code. About 40 to 50 notebooks will cover good work, broken code, thin writing, and right output with a wrong explanation. They are split by notebook into train, validation, and test. Anonymized class work can be added only if the course allows it. Names and ids are kept out of the files and are not sent to an outside model.

## 7. Proposed approach

Week 2 baseline: TF-IDF against the model explanation, keyword checks for the expected ideas, and a check for real markdown and comments. Week 3 adds sentence embeddings. They compare the writing with the model explanation and with a plain description of the code. The mismatch flag means the saved output is acceptable and the writing is not similar to the code. Both are scored against human marks with MAE, RMSE, and correlation. Simple features and embeddings work best together (Haller, Aldea, Seifert, and Strisciuglio, 2022, arXiv:2204.03503). nbgrader scores test cells and leaves the writing to a person. A later week may add a large language model beside these two methods.

## 8. Expected institutional value

Instructors get a first pass on the writing, and a clear mark when the output is fine but the explanation is not. One component serves many labs. The result can go to a gradebook or a student-support tool. It will do worse on unusual wording, very short correct answers, and a vague lab description.
