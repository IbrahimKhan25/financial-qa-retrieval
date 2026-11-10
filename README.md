# Financial QA Retrieval

Fine-tuned retrieval and reranking on BEIR FiQA-2018, with measured quality/latency tradeoffs
(PyTorch training, ONNX int8 inference, FastAPI serving, CI quality/latency gates).

**Status:** Phases 1 and 2 complete on the dev split (baselines and bi-encoder fine-tuning ablation).
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

Test split, reranking, hybrid and latency tables: *pending (later phases)*.

## Reproduce

```bash
uv sync --group dev --group train --group eval
uv run python scripts/download_data.py
uv run python scripts/run_baselines.py --split dev      # Phase 1 baselines
uv run pytest -q
```

Phase 2 training runs on a GPU: open `notebooks/colab_train.ipynb` in Colab (T4) and run the cells in order;
it mines negatives and trains rows B, C, C2 and D, writing `results/train_*.json`.
