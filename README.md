# Financial QA Retrieval

Fine-tuned retrieval and reranking on BEIR FiQA-2018, with measured quality/latency tradeoffs
(PyTorch training, ONNX int8 inference, FastAPI serving, CI quality/latency gates).

**Status:** Phases 1 to 3 complete on the dev split (baselines, bi-encoder fine-tuning ablation, hybrid retrieval); Phase 4 (reranking) complete: neither the off-the-shelf nor the fine-tuned reranker beats the first stage, so serving is retrieval-only; Phase 5 (ONNX, fusion, int8) measured on an Apple M2 and an Intel i7-9700F (Docker timing pending).
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

### Cross-encoder reranking (dev, 500 queries): off-the-shelf reranker

The off-the-shelf `cross-encoder/ms-marco-MiniLM-L-6-v2` rescored the first stage's top-100 candidates; the
rerank depth is how many of the top candidates it reorders (the rest keep their order, so Recall@100 never
changes). Question and post are encoded together, post trimmed to 256 tokens. A fine-tuned reranker is
**pending** (it needs a GPU run on Colab).

| First stage | Reranking | NDCG@10 [95% CI] | MRR@10 | NDCG@10 change vs no reranking [95% CI] |
|---|---|---|---|---|
| Dense, row D | no reranking | 0.428 [0.396, 0.460] | 0.501 | - |
| Dense, row D | rerank top 10 | 0.423 [0.391, 0.454] | 0.490 | -0.005 [-0.019, +0.010] |
| Dense, row D | rerank top 20 | 0.409 [0.378, 0.440] | 0.475 | -0.019 [-0.039, +0.001] |
| Dense, row D | rerank top 50 | 0.397 [0.365, 0.427] | 0.460 | -0.032 [-0.054, -0.008] |
| Dense, row D | rerank top 100 | 0.379 [0.347, 0.410] | 0.447 | -0.049 [-0.073, -0.025] |
| Hybrid, row D + BM25 (alpha 0.8) | no reranking | 0.444 [0.411, 0.477] | 0.515 | - |
| Hybrid, row D + BM25 (alpha 0.8) | rerank top 10 | 0.423 [0.392, 0.455] | 0.484 | -0.020 [-0.034, -0.007] |
| Hybrid, row D + BM25 (alpha 0.8) | rerank top 20 | 0.412 [0.380, 0.443] | 0.475 | -0.032 [-0.051, -0.012] |
| Hybrid, row D + BM25 (alpha 0.8) | rerank top 50 | 0.390 [0.359, 0.420] | 0.452 | -0.054 [-0.076, -0.032] |
| Hybrid, row D + BM25 (alpha 0.8) | rerank top 100 | 0.365 [0.334, 0.396] | 0.436 | -0.079 [-0.101, -0.055] |

**Result: the off-the-shelf reranker made both strong first stages worse, and the more candidates it reordered,
the worse it got.** Reranking the dense top 100 lost 0.049 NDCG@10
(significant); the hybrid lost more, starting from a higher score. At depth 10 the dense loss is not significant.

**Control experiment.** To rule out a bug, I ran the same reranker on BM25's candidates, where published BEIR
results show rerankers of this family help on FiQA (I did not re-verify the exact published figure here):

| First stage | Reranking | NDCG@10 [95% CI] | MRR@10 | NDCG@10 change vs no reranking [95% CI] |
|---|---|---|---|---|
| BM25 (validation run) | no reranking | 0.231 [0.205, 0.259] | 0.287 | - |
| BM25 (validation run) | rerank top 10 | 0.271 [0.240, 0.302] | 0.359 | +0.040 [+0.027, +0.053] |
| BM25 (validation run) | rerank top 20 | 0.294 [0.263, 0.325] | 0.377 | +0.063 [+0.046, +0.080] |
| BM25 (validation run) | rerank top 50 | 0.319 [0.288, 0.350] | 0.399 | +0.088 [+0.068, +0.108] |
| BM25 (validation run) | rerank top 100 | 0.330 [0.298, 0.360] | 0.411 | +0.098 [+0.078, +0.119] |

It helps strongly and the gain grows with depth, so the scoring pipeline is sound. The likely reason (not tested) is
that this reranker was trained on MS MARCO web-search passages, not financial opinion posts, so it is a weaker
judge of FiQA relevance than a first stage fine-tuned on FiQA: it adds value on top of a weak first stage
(BM25, 0.231) and subtracts from a strong one (0.428 to 0.444).

CPU latency of the reranker alone (per query, one batch of `depth` pairs; includes tokenization and the forward pass):

| Rerank depth | p50 per query | p95 per query |
|---|---|---|
| 10 | 124 ms | 131 ms |
| 20 | 252 ms | 257 ms |
| 50 | 576 ms | 586 ms |
| 100 | 1207 ms | 1322 ms |

Hardware for these latency numbers: **Apple M2 (arm64), 4 threads**, PyTorch 2.15.0 fp32, 100 queries after 3
warm-up queries, outside Docker. Cost grows roughly linearly with depth (12.1 ms per pair at depth 100). The quality
scores above were computed on the Apple GPU (MPS), the latency on CPU. This M2 is not the x86 cloud CPU we would
deploy on, and the Docker benchmark is still to do (Phases 5 and 6); these numbers are the unoptimized baseline
that ONNX and int8 must beat.

Caveats: one off-the-shelf model, one seed for everything upstream; the hybrid's alpha and row D were chosen on dev;
no test-set number has been computed. Tuning a blend of reranker and first-stage scores was not tried.

### Fine-tuned reranker (dev): two runs, neither beats the first stage

The off-the-shelf reranker hurt a strong first stage, so I fine-tuned it on FiQA training questions: listwise group softmax over 1 correct post plus 7
negatives per question (hard negatives mined with BM25 and with the fine-tuned bi-encoder, ranks 0 to 100, dev and test answers excluded), 8 questions
per step, learning rate 2e-5, 2 epochs, seed 42, Tesla T4 with fp16 (about 7.6 minutes per epoch). **Run 1** used hard negatives only. **Run 2** is the
same recipe with 3 of the 7 negatives replaced by random documents. Each epoch reranks the dev candidates; epoch 0 is the untouched model and reproduced
0.365 locally in both runs.

| System (dev, rerank top 100 of the hybrid candidates) | NDCG@10 [95% CI] | MRR@10 | Recall@100 |
|---|---|---|---|
| First stage only (hybrid, no reranking) | 0.444 [0.411, 0.477] | 0.515 | 0.742 |
| Off-the-shelf reranker (epoch 0) | 0.365 [0.334, 0.396] | 0.436 | 0.742 |
| Run 1 (hard negatives only), epoch 1 | 0.289 | 0.351 | 0.742 |
| Run 1 (hard negatives only), epoch 2, best | 0.296 [0.265, 0.327] | 0.356 | 0.742 |
| Run 2 (3 of 7 negatives random), epoch 1 | 0.342 | 0.409 | 0.742 |
| Run 2 (3 of 7 negatives random), epoch 2, best | 0.363 [0.330, 0.395] | 0.431 | 0.742 |

Paired differences (NDCG@10, 500 queries): run 1 minus off-the-shelf -0.069 [-0.091, -0.049]; run 2 minus off-the-shelf
-0.003 [-0.023, +0.018]; run 2 minus the first stage alone -0.081 [-0.105, -0.058].

**Run 1 got worse, and it was not a coding bug.** A diagnostic on training groups (`scripts/diagnose_reranker.py`, results in `results/reranker_diagnostic.json`):

| Training groups of 8 (300 train queries): correct post ranked first | Off-the-shelf | Run 1 | Run 2 |
|---|---|---|---|
| Hard negatives (the training distribution) | 36.7% | 52.7% | 50.0% |
| Random documents (easy) | 93.7% | 16.3% | 88.0% |

Run 1 improved the hard cases it practised but collapsed on easy ones (from 94% to 16%, chance
12.5%): it forgot basic relevance. Run 2, chosen from this training-set diagnostic and not from dev scores, kept the easy-case ability
(88%) and also improved on hard cases, which supports the explanation that hard-only negatives caused the collapse.

**But run 2 did not make the reranker better.** Its dev NDCG@10 (0.363) is statistically the same as the off-the-shelf reranker's (0.365), and both are
significantly below the first stage alone (0.444). The improvement on hard training groups did not transfer to dev. Possible reasons, all untested: label
sparsity (many top-ranked "negatives" are unlabeled good answers: 53% of the BM25 and 60% of the dense negatives are answers to other training questions), too little
training signal, or overfitting. Dev NDCG@10 was still rising from epoch 1 (0.342) to epoch 2 (0.363) and the loss was still falling, so run 2 may be under-trained; I did not
continue, because this was one planned retry and further tuning against dev would make the dev number less honest.

**Decision:** a reranker, off-the-shelf or fine-tuned as above, does not improve on the hybrid first stage on this data, so the service will serve retrieval only
(hybrid BM25 plus fine-tuned bi-encoder) with reranking available as an off-by-default option. Caveats: one seed per run, two runs, best epoch chosen on dev,
no test number computed.

### Inference optimization: ONNX export, graph fusion and int8 quantization (dev)

Both neural pieces were exported to ONNX (opset 17, dynamic batch and length) in two ways. The **default export** uses
PyTorch's built-in attention operator. The **fused export** re-exports with explicit attention and runs ONNX Runtime's
BERT optimizer, which fuses attention, embedding plus LayerNorm, GELU and residual LayerNorm into optimized kernels
(the pipeline raises an error if attention was not fused in every layer). Each is also quantized with dynamic int8
(weights stored as 8-bit integers, activation scales computed at run time, no calibration data). Reference: PyTorch fp32.

**Is the export faithful?** The default-export fp32 bi-encoder matches PyTorch to 2.4e-07 (max absolute
difference in the embedding) and the reranker to 1.2e-05; the fused fp32 graphs match to 2.6e-07
and 9.8e-06. Retrieval quality with ONNX fp32 is identical to PyTorch. Fused int8 reranker scores correlate
0.998 (Pearson) and 0.997 (Spearman) with fp32 on 150 pairs.

**Quality cost of int8 (dev, 500 queries).**

| Default-export int8: bi-encoder setup | NDCG@10 [95% CI] | Recall@100 | MRR@10 | NDCG@10 change vs PyTorch fp32 [95% CI] |
|---|---|---|---|---|
| pytorch fp32 (reference) | 0.428 [0.396, 0.460] | 0.744 | 0.501 | - |
| onnx fp32 queries, fp32 docs | 0.428 [0.396, 0.460] | 0.744 | 0.501 | +0.000 [+0.000, +0.000] |
| int8 queries, fp32 docs | 0.415 [0.383, 0.448] | 0.742 | 0.487 | -0.013 [-0.020, -0.005] |
| int8 queries, int8 docs | 0.405 [0.372, 0.437] | 0.723 | 0.464 | -0.023 [-0.036, -0.011] |

| Default-export int8: reranker (off-the-shelf) | NDCG@10 PyTorch fp32 | NDCG@10 ONNX int8 | Change [95% CI] |
|---|---|---|---|
| rerank top 20 (dense first stage) | 0.4089 | 0.4081 | -0.001 [-0.005, +0.003] |
| rerank top 100 (dense first stage) | 0.3788 | 0.3797 | +0.001 [-0.005, +0.006] |

| Fused int8: bi-encoder setup | NDCG@10 [95% CI] | Recall@100 | MRR@10 | NDCG@10 change vs PyTorch fp32 [95% CI] |
|---|---|---|---|---|
| pytorch fp32 (reference) | 0.428 [0.396, 0.460] | 0.744 | 0.501 | - |
| onnx fp32 queries, fp32 docs | 0.428 [0.396, 0.460] | 0.744 | 0.501 | +0.000 [+0.000, +0.000] |
| int8 queries, fp32 docs | 0.413 [0.381, 0.445] | 0.743 | 0.481 | -0.015 [-0.024, -0.006] |
| int8 queries, int8 docs | 0.400 [0.369, 0.432] | 0.724 | 0.463 | -0.028 [-0.041, -0.015] |

| Fused int8: reranker (off-the-shelf) | NDCG@10 PyTorch fp32 | NDCG@10 ONNX int8 | Change [95% CI] |
|---|---|---|---|
| rerank top 20 (dense first stage) | 0.4089 | 0.4096 | +0.001 [-0.004, +0.005] |
| rerank top 100 (dense first stage) | 0.3788 | 0.3814 | +0.003 [-0.002, +0.008] |

**Speed and size on an Apple M2.** CPU only, one query per call, timings include tokenization, 100 queries after 3 warm-up
queries. Hardware: **Apple M2 (arm64), 4 threads**, PyTorch 2.15.0, ONNX Runtime CPU provider, outside Docker.

| Question encoder (1 query) | p50 | p95 | Documents/s (batch 32) | Size |
|---|---|---|---|---|
| PyTorch fp32 (default) | 5.5 ms | 6.3 ms | 110 | 90.9 MB |
| ONNX fp32, default export | 2.0 ms | 3.0 ms | 64 | 90.4 MB |
| ONNX int8, default export | 1.0 ms | 1.5 ms | 120 | 22.9 MB |
| ONNX fp32, fused | 1.7 ms | 2.6 ms | 73 | 90.3 MB |
| ONNX int8, fused | 0.8 ms | 1.2 ms | 153 | 22.7 MB |

| Reranker | depth 10 p50 / p95 | depth 20 p50 / p95 | depth 50 p50 / p95 | depth 100 p50 / p95 | Pairs/s (batch 32) | Size |
|---|---|---|---|---|---|---|
| PyTorch fp32 (default) | 121 / 130 ms | 264 / 292 ms | 619 / 713 ms | 1281 / 1353 ms | 104 | 90.9 MB |
| ONNX fp32, default export | 245 / 306 ms | 498 / 536 ms | 1256 / 1329 ms | 3199 / 4628 ms | 54 | 91.0 MB |
| ONNX int8, default export | 133 / 154 ms | 266 / 275 ms | 664 / 760 ms | 1709 / 2439 ms | 95 | 23.0 MB |
| ONNX fp32, fused | 216 / 223 ms | 427 / 456 ms | 1028 / 1075 ms | 2120 / 2300 ms | 66 | 90.9 MB |
| ONNX int8, fused | 100 / 104 ms | 198 / 206 ms | 537 / 714 ms | 1018 / 1245 ms | 137 | 22.9 MB |

**Speed and size on x86.** Same benchmark. Hardware: **Intel(R) Core(TM) i7-9700F CPU @ 3.00GHz (x86-64, 8 cores), 4 threads**, 15.9 GB RAM,
Windows-10-10.0.26200-SP0, PyTorch 2.15.0+cpu, ONNX Runtime CPU provider, outside Docker. (An earlier x86 run of only the default export
agreed with these default-export numbers within about 2% for the reranker.)

| Question encoder (1 query) | p50 | p95 | Documents/s (batch 32) | Size |
|---|---|---|---|---|
| PyTorch fp32 (default) | 5.9 ms | 6.7 ms | 72 | 90.9 MB |
| ONNX fp32, default export | 2.6 ms | 3.3 ms | 47 | 90.4 MB |
| ONNX int8, default export | 1.6 ms | 2.3 ms | 56 | 22.9 MB |
| ONNX fp32, fused | 2.1 ms | 2.8 ms | 77 | 90.3 MB |
| ONNX int8, fused | 1.3 ms | 1.9 ms | 93 | 22.7 MB |

| Reranker | depth 10 p50 / p95 | depth 20 p50 / p95 | depth 50 p50 / p95 | depth 100 p50 / p95 | Pairs/s (batch 32) | Size |
|---|---|---|---|---|---|---|
| PyTorch fp32 (default) | 209 / 211 ms | 420 / 426 ms | 1038 / 1049 ms | 2131 / 2191 ms | 65 | 90.9 MB |
| ONNX fp32, default export | 317 / 319 ms | 671 / 674 ms | 1724 / 1753 ms | 3500 / 3594 ms | 42 | 91.0 MB |
| ONNX int8, default export | 281 / 284 ms | 573 / 575 ms | 1463 / 1494 ms | 2939 / 2966 ms | 48 | 23.0 MB |
| ONNX fp32, fused | 186 / 211 ms | 387 / 389 ms | 991 / 996 ms | 2016 / 2121 ms | 69 | 90.9 MB |
| ONNX int8, fused | 158 / 159 ms | 318 / 321 ms | 800 / 804 ms | 1605 / 1609 ms | 85 | 22.9 MB |

What this shows:

- **The default ONNX export was slower than PyTorch for the reranker on both machines**, and I first concluded that ONNX
  does not help. That conclusion was wrong: the exported graph had skipped a standard optimization step.
- **With graph fusion plus int8, the reranker is about 1.3x faster than default PyTorch on both machines.** At depth 100:
  Apple M2 1.26x (1018 ms against 1281 ms p50), Intel i7-9700F 1.33x
  (1605 ms against 2131 ms). Across depths 10 to 100 the speedup ranged 1.15 to 1.33x on the M2
  and 1.30 to 1.33x on the i7. The question encoder is 6.7x faster on the M2 and 4.5x on the i7. Both models shrink to about a
  quarter of the size. Fused fp32 alone is 1.06x PyTorch on the i7 and slower than PyTorch on the M2, so the gain comes from
  fusion and int8 together.
- **Quality cost depends on the model.** The fused int8 reranker lost nothing measurable (+0.001 [-0.004, +0.005] NDCG@10 at depth 20,
  +0.003 [-0.002, +0.008] at depth 100). The fused int8 bi-encoder lost -0.015 [-0.024, -0.006] NDCG@10 with int8
  queries only and -0.028 [-0.041, -0.015] with int8 documents as well (the default-export int8 lost
  -0.013 [-0.020, -0.005] and -0.023 [-0.036, -0.011]). A plausible reason (not tested) is that small embedding
  errors accumulate when a query is compared against 57,000 documents. For serving this suggests int8 for the reranker and fp32 for the stored
  document embeddings and the question encoder, unless the encoder's speed matters more than 0.015 NDCG@10.

Caveats: single runs per machine (the same configuration varies by roughly 10 to 25% between sessions on the M2, and by up to about 20% for the single-query
encoder on the i7); no Docker yet; one x86 CPU without the newest 8-bit instructions, so a server CPU may show a different int8 gain (untested); the quality
numbers are hardware independent but the speed numbers are not; the fine-tuned reranker has not been exported yet (the same script handles it); why the default
export was slow is a hypothesis, not a measured cause.

**Corrections.** (1) An earlier version of the reranker latency table was measured on a PyTorch CPU path that returned NaN scores for this
model: weights loaded straight from the file broke the CPU forward pass on this Mac (copying them into fresh memory fixes it;
my first guess, oneDNN, was tested and was wrong). Quality numbers were unaffected, the corrected latencies were within
about 5 to 15% of the earlier ones, the code now raises on NaN, and a regression test pins correct scores. (2) The first
Phase 5 conclusion that ONNX was not faster was caused by the missing fusion step, described above.

Docker latency, serving and the test split: *pending (later phases)*.

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
