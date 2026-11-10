# Financial QA Retrieval

Fine-tuned retrieval and reranking on BEIR FiQA-2018, with measured quality/latency tradeoffs
(PyTorch training, ONNX int8 inference, FastAPI serving, CI quality/latency gates).

**Status:** Phases 1 to 3 complete on the dev split (baselines, bi-encoder fine-tuning ablation, hybrid retrieval).
Test-set numbers are produced once, at the end, by a single script. Every number below comes from a
committed file in `results/`.

## Data and protocol

- BEIR FiQA-2018 (`fiqa.zip`, sha256 `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02`).
- Corpus 57,638 docs, 6,648 queries. Query splits are disjoint:
  train 5,500 (14,166 qrels), **dev 500** (1,238), test 648 (1,706).
- Model selection and tuning use the official **dev** split only. The test split is not evaluated
  until the final run (`run_baselines.py` refuses `--split test`).
- Metrics: NDCG@10, Recall@100, MRR@10 (own implementation, cross-checked against `pytrec_eval` in tests).
- Confidence intervals: percentile bootstrap over queries (10,000 resamples, seed 42). Comparisons
  between systems use a **paired** bootstrap on the same queries. The bootstrap p-value has a
  resolution of 1e-4.
- Hard negatives are mined for train queries only, and **every dev and test positive is excluded
  from the pool** (checked on the written files: 0 leaked).

## Results (dev, 500 queries)

### Baselines

| System | NDCG@10 [95% CI] | Recall@100 [95% CI] | MRR@10 [95% CI] |
|---|---|---|---|
| BM25 (bm25s, k1=1.5, b=0.75, en stopwords, no stemming) | 0.231 [0.205, 0.259] | 0.497 [0.461, 0.533] | 0.287 [0.254, 0.323] |
| all-MiniLM-L6-v2, zero-shot (max_seq_length 256) \* | 0.388 [0.356, 0.421] | 0.720 [0.688, 0.753] | 0.453 [0.415, 0.492] |

Paired difference, MiniLM minus BM25: NDCG@10 **+0.157, 95% CI [+0.127, +0.186]**.

\* **Contamination caveat.** all-MiniLM-L6-v2 was trained on a large mix of public sentence-pair
data that includes StackExchange-style question/answer text. We could not verify that FiQA itself
was excluded, so this row may be optimistic relative to a truly unseen domain. Treat it as an
off-the-shelf reference point, not a clean zero-shot measurement.

### Bi-encoder fine-tuning ablation

Own PyTorch loop: mean-pooled MiniLM, InfoNCE loss (temperature 0.05), batch 64, AdamW lr 2e-5,
3 epochs, seed 42, one question per batch, other questions' positives masked out of the negatives.
Rows B, C, C2 and D all start from the same pretrained model; only the negatives differ.

| Row | Training negatives | Best epoch | NDCG@10 [95% CI] | Recall@100 [95% CI] | MRR@10 [95% CI] | NDCG@10 gain vs zero-shot [95% CI] |
|---|---|---|---|---|---|---|
| A. Zero-shot MiniLM | none (no training) | - | 0.388 [0.356, 0.421] | 0.720 [0.688, 0.753] | 0.453 [0.415, 0.492] | - |
| B. Fine-tuned | in-batch only | 1 | 0.415 [0.382, 0.447] | 0.733 [0.702, 0.764] | 0.488 [0.449, 0.527] | +0.027 [+0.015, +0.038] |
| C. Fine-tuned | in-batch + BM25 hard | 1 | 0.410 [0.377, 0.442] | 0.728 [0.696, 0.760] | 0.481 [0.442, 0.520] | +0.021 [+0.008, +0.034] |
| C2. Fine-tuned | in-batch + BM25 hard, likely false negatives removed | 1 | 0.410 [0.378, 0.443] | 0.735 [0.703, 0.766] | 0.484 [0.444, 0.522] | +0.022 [+0.009, +0.036] |
| D. Fine-tuned | in-batch + BM25 + negatives mined from model C | 3 | 0.428 [0.396, 0.460] | 0.744 [0.712, 0.774] | 0.501 [0.462, 0.540] | +0.040 [+0.026, +0.054] |

Differences between rows (paired bootstrap on the same 500 queries):

| Comparison (dev, NDCG@10) | Paired difference [95% CI] | Reading |
|---|---|---|
| C minus B (add BM25 hard negatives) | -0.005 [-0.013, +0.003] | no gain; sign is negative, not significant |
| C2 minus C (remove likely false negatives) | +0.001 [-0.006, +0.008] | no measurable effect |
| D minus B (add model-mined negatives) | +0.014 [+0.004, +0.024] | significant gain |
| D minus C | +0.019 [+0.011, +0.028] | significant gain |

What the ablation shows:

- Fine-tuning helps modestly: in-batch negatives alone move NDCG@10 from 0.388 to 0.415.
- **BM25 hard negatives did not beat in-batch negatives alone** (row C is not better than row B), and
  removing the likely false negatives (row C2) made no measurable difference, even though 51.7% of the
  BM25 negatives are train answers to other questions versus 24.6% of the corpus.
- Negatives mined from the fine-tuned model (row D) gave the best result: +0.040 NDCG@10 over zero-shot
  and +0.014 over row B (significant on NDCG@10; the Recall@100 and MRR@10 differences against B are not).

Caveats on these results:

- **One seed (42).** There is no estimate of run-to-run variance, so differences of a few thousandths
  (B vs C vs C2) are within what another seed could change.
- **The epoch is chosen on dev**, so each row's dev score is slightly optimistic. Rows B, C and C2 did
  not improve after epoch 1; row D kept improving through epoch 3.
- Row D's negatives came from model C, whose checkpoint was selected on dev, so dev has a small indirect
  influence on D's training data (train queries only; no dev or test labels are used as training signal).
- Row D changes two things at once (model-mined negatives and a larger, more diverse pool), so this
  ablation cannot say which one helped. Explanations for why BM25 negatives did not help are hypotheses,
  not tested.
- Dev is not test. No test-set number has been computed.
- Training ran on Colab: Tesla T4, torch 2.14.1+cu130, Python 3.13.16, fp16 autocast. Mean epoch
  time: B 128 s, C 177 s, C2 175 s, D 180 s.
  The zero-shot comparison uses the Phase 1 per-query scores (Apple M2); re-evaluating zero-shot on the T4 gave
  NDCG@10 0.3877 against 0.3881, a negligible device difference.

### Notes on the baselines

- Our BM25 (bm25s, Lucene-style scoring) is not the Elasticsearch BM25 used for BEIR's published
  numbers (about 0.236 NDCG@10 on test), so exact values differ slightly. Parameters are library
  defaults and were not tuned.
- 19.15% of documents exceed MiniLM's 256-token limit and are truncated.

### Hybrid retrieval (BM25 + dense), dev, 500 queries

Two fusions of the BM25 and dense top-100 lists. **RRF** scores each document by the sum of 1 / (60 + rank)
over both lists, using ranks only. **Weighted** min-max scales each list's scores to [0.001, 1] per query, then
takes alpha x dense + (1 - alpha) x BM25 (alpha = 1 is dense alone, alpha = 0 is BM25 alone; tests check both
extremes reproduce the single systems exactly).

| System | NDCG@10 [95% CI] | Recall@100 [95% CI] | MRR@10 [95% CI] |
|---|---|---|---|
| BM25 alone | 0.231 [0.205, 0.259] | 0.497 [0.461, 0.533] | 0.287 [0.254, 0.323] |
| Dense, zero-shot MiniLM | 0.388 [0.356, 0.421] | 0.720 [0.688, 0.753] | 0.453 [0.415, 0.492] |
| Dense, fine-tuned (row D) | 0.428 [0.396, 0.460] | 0.744 [0.712, 0.774] | 0.501 [0.461, 0.539] |
| Hybrid RRF: zero-shot + BM25 | 0.344 [0.313, 0.374] | 0.709 [0.676, 0.742] | 0.414 [0.377, 0.453] |
| Hybrid RRF: row D + BM25 | 0.367 [0.337, 0.398] | 0.729 [0.697, 0.760] | 0.436 [0.398, 0.475] |
| Hybrid weighted: zero-shot + BM25, alpha=0.7 | 0.411 [0.378, 0.443] | 0.720 [0.687, 0.752] | 0.481 [0.443, 0.521] |
| Hybrid weighted: row D + BM25, alpha=0.8 | 0.444 [0.411, 0.477] | 0.742 [0.711, 0.773] | 0.515 [0.476, 0.553] |

Paired differences on the same 500 queries:

| Comparison (first minus second) | NDCG@10 [95% CI] | Recall@100 [95% CI] | Reading |
|---|---|---|---|
| rrf_D minus dense_D | -0.061 [-0.085, -0.036] | -0.015 [-0.034, +0.005] | worse; likely because equal-weight RRF lets the weaker BM25 pull the ranking down (not tested) |
| weighted_D (alpha=0.8) minus dense_D | +0.016 [+0.008, +0.024] | -0.002 [-0.017, +0.013] | better NDCG@10; no Recall@100 gain |
| rrf_D minus bm25 | +0.136 [+0.118, +0.155] | +0.232 [+0.202, +0.263] | better than BM25 alone |
| weighted_D (alpha=0.8) minus bm25 | +0.213 [+0.184, +0.241] | +0.245 [+0.212, +0.279] | better than BM25 alone |
| weighted_D (alpha=0.8) minus rrf_D | +0.076 [+0.055, +0.098] | +0.013 [+0.001, +0.026] | weighted fusion beats RRF here |
| rrf_zs minus dense_zs | -0.045 [-0.067, -0.022] | -0.011 [-0.032, +0.009] | worse, same pattern as row D |
| weighted_zs (alpha=0.7) minus dense_zs | +0.023 [+0.011, +0.034] | -0.000 [-0.018, +0.018] | better NDCG@10; no Recall@100 gain |

Alpha sweep (NDCG@10), showing how sensitive the weighted fusion is:

| alpha | NDCG@10, row D + BM25 | NDCG@10, zero-shot + BM25 |
|---|---|---|
| 0.0 | 0.232 | 0.232 |
| 0.1 | 0.253 | 0.252 |
| 0.2 | 0.279 | 0.274 |
| 0.3 | 0.312 | 0.304 |
| 0.4 | 0.364 | 0.342 |
| 0.5 | 0.416 | 0.386 |
| 0.6 | 0.437 | 0.405 |
| 0.7 | 0.443 | 0.411 |
| 0.8 | 0.444 | 0.405 |
| 0.9 | 0.438 | 0.399 |
| 1.0 | 0.428 | 0.388 |

What the hybrid experiment shows:

- **Weighted fusion helps a little**: row D + BM25 reaches 0.444 NDCG@10,
  +0.016 [+0.008, +0.024] over row D alone. The alpha curve has a plateau from 0.6 to 0.9,
  so the result is not a sharp tuning artifact.
- **Standard equal-weight RRF made things worse** than dense alone (-0.061 [-0.085, -0.036]). A likely reason, not tested: BM25 is far
  weaker than the fine-tuned model (0.231 vs 0.428), and RRF gives both lists equal say. The k=60 default was
  not tuned, and a weighted RRF was not tried.
- **The hybrid did not raise Recall@100** (0.742 vs 0.744 for dense alone;
  the difference is not significant). The gain is in ordering the top of the list, not in finding more answers
  within 100. That matters for reranking, whose ceiling is recall at the rerank depth; Phase 4 will compare
  both candidate generators at depths 10, 20, 50 and 100.
- Fine-tuning shrinks what BM25 adds: the hybrid gain is +0.023 [+0.011, +0.034] on the zero-shot
  model and +0.016 [+0.008, +0.024] on row D.

Caveats: alpha is chosen on dev (one parameter, so the optimism is small, but the selected score is not a clean
estimate), and row D was itself chosen on dev; one training seed; the hybrid run used the checkpoint downloaded
from Colab (checkpoints are git-ignored) and ran on mps (Apple M2); re-evaluating row D there
gave NDCG@10 0.4281, against 0.4284 on the T4, a negligible device difference.
No test-set number has been computed.

Test split, reranking and latency tables: *pending (later phases)*.

## Reproduce

```bash
uv sync --group dev --group train --group eval
uv run python scripts/download_data.py
uv run python scripts/run_baselines.py --split dev      # Phase 1 baselines
uv run pytest -q
```

Phase 2 training runs on a GPU: open `notebooks/colab_train.ipynb` in Colab (T4) and run the cells in order;
it mines negatives and trains rows B, C, C2 and D, writing `results/train_*.json`.
Phase 3 (`uv run python scripts/run_hybrid.py`) needs the row D checkpoint from that run in `checkpoints/bi_encoder_D_mined`.
