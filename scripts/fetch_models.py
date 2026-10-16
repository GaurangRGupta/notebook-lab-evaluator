"""Fetch the all-MiniLM-L6-v2 ONNX export into models/ when huggingface.co is unreachable.

Source: npm package @alvix/all-minilm-l6-v2@1.0.1, which contains the Xenova ONNX export of
sentence-transformers/all-MiniLM-L6-v2 (config `_name_or_path` = sentence-transformers/all-MiniLM-L6-v2).
Safeguards: (1) the tarball's sha512 must equal the registry's pinned integrity value;
(2) only data files are extracted (no code is executed); (3) the model must reproduce the
similarity matrix published in the official sentence-transformers README, otherwise it is rejected.
If huggingface.co is reachable, the official repository can be used instead (encoder: sentence-transformers).
"""
import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URL = "https://registry.npmjs.org/@alvix/all-minilm-l6-v2/-/all-minilm-l6-v2-1.0.1.tgz"
SHA512 = "CWer2/eGIia/30MIL7mSGrdk0mMa40BM50HLxNIi+5TQoJDWAC576F20N7hzQjiLlc2FcSx/mn9fXSL4mhulcg=="
PREFIX = "package/dist/Xenova/all-MiniLM-L6-v2/"
WANTED = {"onnx/model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"}
DEST = ROOT / "models" / "all-MiniLM-L6-v2-onnx"


def main():
    if not (DEST / "onnx" / "model.onnx").exists():
        data = urllib.request.urlopen(URL, timeout=300).read()
        got = base64.b64encode(hashlib.sha512(data).digest()).decode()
        if got != SHA512:
            sys.exit("integrity check FAILED; not using this file")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                rel = m.name[len(PREFIX):] if m.name.startswith(PREFIX) else None
                if rel in WANTED and m.isfile():
                    out = DEST / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(tf.extractfile(m).read())
    from src.models.semantic_similarity import OnnxSentenceEncoder, verify_reference
    res = verify_reference(OnnxSentenceEncoder(str(DEST)))
    (DEST / "provenance_check.json").write_text(json.dumps({"source": URL, "sha512": SHA512, **res}, indent=2))
    print(json.dumps(res, indent=2))
    if not res["passed"]:
        sys.exit("model does not reproduce the published reference values; rejected")


if __name__ == "__main__":
    main()
