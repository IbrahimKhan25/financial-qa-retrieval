"""Fine-tune the cross-encoder with a listwise group softmax (1 positive + N negatives per query).

Dev only: each epoch reranks the dev candidate lists, epoch 0 is the off-the-shelf model, and the
best fine-tuned epoch is chosen on dev (so its dev score is slightly optimistic). Test is never
touched. --smoke runs a tiny end-to-end check and writes nothing.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.eval.stats import bootstrap_ci, paired_bootstrap
from finqa.retrieval.rerank import rerank_top
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.data import build_pairs, load_negatives, make_batches, positives_of, sample_group
from finqa.train.losses import group_cross_entropy
from finqa.train.model import pick_device
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed
from finqa.utils.tracking import log_metrics, start_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init", help="override the starting weights (used by the smoke test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.init:
        cfg["init"] = args.init
    set_seed(cfg["seed"])
    device = pick_device()
    use_amp = device == "cuda"

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    train_qrels, dev_qrels = fiqa.load_qrels(d, "train"), fiqa.load_qrels(d, "dev")
    all_doc_ids = list(corpus)
    text = {i: fiqa.doc_text(corpus[i]) for i in all_doc_ids}
    pairs, pos_of = build_pairs(train_qrels), positives_of(train_qrels)
    negatives = load_negatives(cfg["negatives"])
    cands = json.loads(Path(cfg["dev_candidates"]).read_text())["candidates"]
    dev_qids = sorted(dev_qrels)
    if args.smoke:
        cfg.update(epochs=1, queries_per_batch=2, group_size=4)
        pairs, dev_qids = pairs[:16], dev_qids[:10]
    dev_qrels = {q: dev_qrels[q] for q in dev_qids}
    depth = len(next(iter(cands.values())))

    model = CrossEncoder(cfg["init"], max_length=cfg["max_len"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps = len(make_batches(pairs, cfg["queries_per_batch"], cfg["seed"])) * cfg["epochs"]
    warmup = max(1, int(steps * cfg["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)))
    rng = random.Random(cfg["seed"])
    ckpt_dir = Path("checkpoints") / cfg["name"]

    def dev_eval():
        qs = [queries[q] for q in dev_qids for _ in cands[q]]
        ds = [text[x] for q in dev_qids for x in cands[q]]
        sc = model.score(qs, ds)
        ranking, off = {}, 0
        for q in dev_qids:
            n = len(cands[q])
            ranking[q] = rerank_top(cands[q], sc[off:off + n].tolist(), depth)
            off += n
        per = metrics.per_query(ranking, dev_qrels)
        return per, metrics.aggregate(per)

    r0 = metrics.per_query({q: cands[q] for q in dev_qids}, dev_qrels)  # first stage, no reranking
    history, best, offshelf = [], None, None
    print(f"{cfg['name']}: device={device} pairs={len(pairs)} dev_queries={len(dev_qids)} "
          f"rerank_depth={depth} negatives={sum(len(v) for v in negatives.values())}", flush=True)
    with start_run("cross_encoder", cfg["name"], {k: str(v) for k, v in cfg.items()}, args.config):
        scores, agg = dev_eval()
        offshelf = scores
        history.append({"epoch": 0, "train_loss": None, "dev": agg})
        log_metrics({f"dev/{m}": v for m, v in agg.items()}, step=0)
        print("epoch 0 (starting weights)", {m: round(v, 4) for m, v in agg.items()}, flush=True)
        g = cfg["group_size"]
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            t0, losses = time.time(), []
            for batch in make_batches(pairs, cfg["queries_per_batch"], cfg["seed"] + epoch):
                qs, ds = [], []
                for i in batch:
                    q, p = pairs[i]
                    group = sample_group(q, p, negatives, pos_of[q], all_doc_ids, rng, g - 1, cfg.get("random_per_group", 0))
                    qs += [queries[q]] * g
                    ds += [text[x] for x in group]
                tok = {k: v.to(device) for k, v in model.tokenize(qs, ds).items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(**tok)
                loss = group_cross_entropy(logits.float().view(len(batch), g))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            scores, agg = dev_eval()
            mean_loss = sum(losses) / len(losses)
            history.append({"epoch": epoch, "train_loss": mean_loss, "dev": agg, "seconds": round(time.time() - t0, 1)})
            log_metrics({"train/loss": mean_loss, **{f"dev/{m}": v for m, v in agg.items()}}, step=epoch)
            print(f"epoch {epoch} loss={mean_loss:.4f}", {m: round(v, 4) for m, v in agg.items()}, flush=True)
            if best is None or agg["ndcg@10"] > best["agg"]["ndcg@10"]:
                best = {"epoch": epoch, "agg": agg, "scores": scores}
                if not args.smoke:
                    model.save(str(ckpt_dir))

    if args.smoke:
        print("smoke run finished OK (nothing written)")
        return

    qids = sorted(best["scores"])
    col = lambda per, m: [per[q][m] for q in qids]
    result = {
        "name": cfg["name"], "config": cfg, "split": "dev", "n_queries": len(qids), "rerank_depth": depth,
        "n_train_pairs": len(pairs), "history": history, "best_finetuned_epoch": best["epoch"],
        "selected_on": "dev ndcg@10 (optimistic by selection)",
        "metrics": {m: bootstrap_ci(col(best["scores"], m)) for m in metrics.METRICS},
        "metrics_starting_weights": {m: bootstrap_ci(col(offshelf, m)) for m in metrics.METRICS},
        "metrics_first_stage_only": {m: bootstrap_ci(col(r0, m)) for m in metrics.METRICS},
        "paired_vs_starting_weights": {m: paired_bootstrap(col(offshelf, m), col(best["scores"], m)) for m in metrics.METRICS},
        "paired_vs_first_stage_only": {m: paired_bootstrap(col(r0, m), col(best["scores"], m)) for m in metrics.METRICS},
        "per_query": {"finetuned": best["scores"], "starting_weights": offshelf, "first_stage_only": r0},
        "hardware": {**describe(), "device": device, "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
    }
    out = Path("results") / f"train_ce_{cfg['name']}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}; checkpoint in {ckpt_dir}")
    for m, v in result["metrics"].items():
        a, b = result["paired_vs_starting_weights"][m], result["paired_vs_first_stage_only"][m]
        print(f"  {m:11s} {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]  vs start {a['diff']:+.4f} "
              f"[{a['ci_low']:+.4f}, {a['ci_high']:+.4f}]  vs first stage {b['diff']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
