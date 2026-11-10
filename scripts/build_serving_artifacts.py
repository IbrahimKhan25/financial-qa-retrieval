"""Build the serving artifacts folder: embeddings, BM25 index, texts, ONNX models, manifest.

The embeddings come from the fine-tuned bi-encoder (cached from the offline evaluation); the question encoder
is the fused fp32 ONNX export and the optional reranker the fused int8 export of the off-the-shelf cross-encoder.
Output is git-ignored; scripts/check_serving_parity.py verifies it reproduces the dev numbers before any timing.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from finqa.data import fiqa
from finqa.eval.dense_eval import encode_cached
from finqa.serve.build import save_bm25, write_text_store
from finqa.train.model import BiEncoder


def sha10(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="serving_artifacts")
    ap.add_argument("--bi-checkpoint", default="checkpoints/bi_encoder_D_mined")
    ap.add_argument("--onnx-dir", default="onnx_models")
    ap.add_argument("--alpha", type=float, default=0.8)
    args = ap.parse_args()
    out, onnx_dir = Path(args.out), Path(args.onnx_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    d = fiqa.download()
    corpus = fiqa.load_corpus(d)
    doc_ids = list(corpus)
    texts = [fiqa.doc_text(corpus[i]) for i in doc_ids]

    ckpt = sha10(Path(args.bi_checkpoint) / "model.safetensors")
    emb = encode_cached(BiEncoder(args.bi_checkpoint), texts, f"finetuned-{ckpt}")  # cache hit after the hybrid run
    np.save(out / "doc_embeddings.npy", emb.astype(np.float32))
    (out / "doc_ids.json").write_text(json.dumps(doc_ids))
    write_text_store(texts, out)
    save_bm25(doc_ids, texts, out)

    for sub, src_dir, model_file in (("bi_encoder", "bi_encoder_D", "model_fused_fp32.onnx"),
                                     ("cross_encoder", "cross_encoder_offshelf", "model_fused_int8.onnx")):
        (out / sub).mkdir()
        shutil.copy(onnx_dir / src_dir / model_file, out / sub / "model.onnx")
        shutil.copy(onnx_dir / src_dir / "tokenizer.json", out / sub / "tokenizer.json")

    hy = json.loads(Path("results/hybrid_dev.json").read_text())["systems"]
    ndcg = lambda k: hy[k]["metrics"]["ndcg@10"]["mean"]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    manifest = {"built_at": datetime.now(UTC).isoformat(timespec="seconds"), "git_sha": sha, "alpha": args.alpha,
                "cand_k": 100, "n_docs": len(doc_ids), "dim": int(emb.shape[1]), "bi_encoder_sha256_10": ckpt,
                "onnx_sha256_10": {"bi_encoder": sha10(out / "bi_encoder" / "model.onnx"),
                                   "cross_encoder": sha10(out / "cross_encoder" / "model.onnx")},
                "expected_dev_ndcg@10": {"dense": ndcg("dense_D"), "bm25": ndcg("bm25"), "hybrid": ndcg(f"weighted_D (alpha={args.alpha})")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {out} ({size:.0f} MB)\n{json.dumps(manifest, indent=1)}")


if __name__ == "__main__":
    main()
