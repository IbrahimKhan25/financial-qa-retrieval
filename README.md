# Financial QA Retrieval

Fine-tuned retrieval and reranking on BEIR FiQA-2018, with measured quality/latency tradeoffs
(PyTorch training, ONNX int8 inference, FastAPI serving, CI quality/latency gates).

**Status:** Phase 1 complete (baselines on dev). Test-set numbers are produced once, at the end,
by a single script. Every number below comes from a committed file in `results/`.

## Data and protocol

- BEIR FiQA-2018 (`fiqa.zip`, sha256 `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02`).
- Corpus 57,638 docs, 6,648 queries. Query splits are disjoint:
  train 5,500 (14,166 qrels), **dev 500** (1,238), test 648 (1,706).
- Model selection and tuning use the official **dev** split only. The test split is not evaluated
  until the final run (`run_baselines.py` refuses `--split test`).
- Metrics: NDCG@10, Recall@100, MRR@10 (own implementation, cross-checked against `pytrec_eval` in tests).
- Confidence intervals: percentile bootstrap over queries (10,000 resamples, seed 42). Comparisons
  between systems use a **paired** bootstrap on the same queries. The bootstrap p-value has a
  resolution of 1e-4, so "p=0.0000" means p < 1e-4.

## Results

### Baselines (dev, 500 queries)

| System | NDCG@10 [95% CI] | Recall@100 [95% CI] | MRR@10 [95% CI] |
|---|---|---|---|
| BM25 (bm25s, k1=1.5, b=0.75, en stopwords, no stemming) | 0.2312 [0.2047, 0.2593] | 0.4968 [0.4608, 0.5333] | 0.2873 [0.2536, 0.3232] |
| all-MiniLM-L6-v2, zero-shot (max_seq_length 256) \* | 0.3881 [0.3563, 0.4206] | 0.7204 [0.6878, 0.7529] | 0.4532 [0.4152, 0.4920] |

Paired difference, MiniLM minus BM25: NDCG@10 **+0.157, 95% CI [+0.127, +0.186]**;
Recall@100 +0.224 [+0.187, +0.260]; MRR@10 +0.166 [+0.130, +0.202]; all p < 1e-4.

Test split: *pending (final run)*. Ablation, reranking and latency tables: *pending (later phases)*.

\* **Contamination caveat.** all-MiniLM-L6-v2 was trained on a large mix of public sentence-pair
data that includes StackExchange-style question/answer text. We could not verify that FiQA itself
was excluded, so this row may be optimistic relative to a truly unseen domain. Treat it as an
off-the-shelf reference point, not a clean zero-shot measurement.

### Notes on the baselines

- Our BM25 (bm25s, Lucene-style scoring) is not the Elasticsearch BM25 used for BEIR's published
  numbers (about 0.236 NDCG@10 on test), so exact values differ slightly. Parameters are library
  defaults and were not tuned.
- 19.15% of documents exceed MiniLM's 256-token limit and are truncated. This is relevant to the
  long-document category in the error analysis.
- Hardware for the baseline run: Apple M2, 8 cores, 8 GB RAM, macOS arm64, embeddings on MPS.
  Latency numbers come later, in the optimization phase.

## Reproduce

```bash
uv sync --group dev --group train --group eval
uv run python scripts/download_data.py
uv run python scripts/run_baselines.py --split dev
uv run pytest -q
```
