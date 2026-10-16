# Model files

The encoder weights are not in git. From the repository root:

```bash
python scripts/fetch_models.py
```

That script downloads the checked ONNX export of `sentence-transformers/all-MiniLM-L6-v2` into `models/all-MiniLM-L6-v2-onnx/` and checks it against the published similarities 0.666, 0.1046, and 0.1411.

`*.onnx` and the rest of `models/` stay untracked. This note is the only tracked file in the folder.

The Week 4 NLI weights are not fetched by this script. `nli.enabled` stays false until a complete local model is present.
