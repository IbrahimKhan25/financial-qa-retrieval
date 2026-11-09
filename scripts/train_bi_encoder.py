"""Fine-tune the bi-encoder with in-batch negatives (+ optional mined hard negatives).

Our own PyTorch loop. Dev is the only split evaluated; the epoch is chosen on dev, so the
dev number of the chosen epoch is slightly optimistic. Test is never touched here.
--smoke runs a tiny end-to-end check and writes no results or checkpoints.
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
from finqa.eval.dense_eval import evaluate
from finqa.eval.stats import bootstrap_ci, paired_bootstrap
from finqa.train.data import (
    build_pairs,
    load_negatives,
    make_batches,
    positives_of,
    sample_negative,
)
from finqa.train.losses import false_negative_mask, info_nce
from finqa.train.model import BiEncoder, pick_device
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed
from finqa.utils.tracking import log_metrics, start_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = pick_device()
    use_amp = device == "cuda"

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    train_qrels, dev_qrels = fiqa.load_qrels(d, "train"), fiqa.load_qrels(d, "dev")
    all_doc_ids = list(corpus)
    text = {i: fiqa.doc_text(corpus[i]) for i in all_doc_ids}
    pairs = build_pairs(train_qrels)
    pos_of = positives_of(train_qrels)
    negatives = load_negatives(cfg["negatives"]) if cfg.get("negatives") else {}
    use_hard = bool(negatives)

    eval_doc_ids = all_doc_ids
    if args.smoke:
        cfg.update(epochs=1, batch_size=8)
        pairs = pairs[:64]
        dev_qrels = dict(sorted(dev_qrels.items())[:20])
        keep = {x for rel in dev_qrels.values() for x in rel} | set(all_doc_ids[:2000])
        eval_doc_ids = [i for i in all_doc_ids if i in keep]
    dev_qids = sorted(dev_qrels)
    dev_q_texts = [queries[q] for q in dev_qids]
    eval_texts = [text[i] for i in eval_doc_ids]

    model = BiEncoder(cfg["init"], max_length=cfg["max_len"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = len(make_batches(pairs, cfg["batch_size"], cfg["seed"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(total_steps * cfg["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / (total_steps - warmup)))
    scale = 1.0 / cfg["temperature"]
    rng = random.Random(cfg["seed"])
    ckpt_dir = Path("checkpoints") / cfg["name"]

    def dev_eval():
        return evaluate(model, eval_doc_ids, eval_texts, dev_qids, dev_q_texts, dev_qrels)

    history, best = [], None
    print(f"{cfg['name']}: device={device} pairs={len(pairs)} steps/epoch={steps_per_epoch} "
          f"hard_negatives={use_hard} dev_queries={len(dev_qids)}")
    with start_run("bi_encoder", cfg["name"], {k: str(v) for k, v in cfg.items()}, args.config):
        if cfg.get("eval_at_start", True):
            scores, agg = dev_eval()
            history.append({"epoch": 0, "train_loss": None, "dev": agg})
            log_metrics({f"dev/{m}": v for m, v in agg.items()}, step=0)
            print("epoch 0 (zero-shot)", {m: round(v, 4) for m, v in agg.items()})
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            t0, losses = time.time(), []
            for batch in make_batches(pairs, cfg["batch_size"], cfg["seed"] + epoch):
                qids = [pairs[i][0] for i in batch]
                docs = [pairs[i][1] for i in batch]
                if use_hard:
                    docs += [sample_negative(q, negatives, pos_of[q], all_doc_ids, rng) for q in qids]
                mask = false_negative_mask(qids, docs, pos_of)
                q_tok = {k: v.to(device) for k, v in model.tokenize([queries[q] for q in qids]).items()}
                d_tok = {k: v.to(device) for k, v in model.tokenize([text[x] for x in docs]).items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    q_emb, d_emb = model(**q_tok), model(**d_tok)
                loss = info_nce(q_emb.float(), d_emb.float(), mask, scale)
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
            history.append({"epoch": epoch, "train_loss": mean_loss, "dev": agg,
                            "seconds": round(time.time() - t0, 1)})
            log_metrics({"train/loss": mean_loss, **{f"dev/{m}": v for m, v in agg.items()}}, step=epoch)
            print(f"epoch {epoch} loss={mean_loss:.4f}", {m: round(v, 4) for m, v in agg.items()})
            if best is None or agg["ndcg@10"] > best["agg"]["ndcg@10"]:
                best = {"epoch": epoch, "agg": agg, "scores": scores}
                if not args.smoke:
                    model.save(str(ckpt_dir))

    if args.smoke:
        print("smoke run finished OK (nothing written)")
        return

    qids = sorted(best["scores"])
    base = json.loads(Path("results/baselines_dev.json").read_text())["systems"]["minilm_zeroshot"]["per_query"]
    result = {
        "name": cfg["name"], "config": cfg, "split": "dev", "n_queries": len(qids),
        "n_train_pairs": len(pairs), "history": history,
        "best_finetuned_epoch": best["epoch"], "selected_on": "dev ndcg@10 (optimistic by selection)",
        "metrics": {m: bootstrap_ci([best["scores"][q][m] for q in qids]) for m in metrics.METRICS},
        "paired_vs_zeroshot": {m: paired_bootstrap([base[q][m] for q in qids],
                                                   [best["scores"][q][m] for q in qids])
                               for m in metrics.METRICS},
        "per_query": best["scores"],
        "hardware": {**describe(), "device": device, "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
    }
    out = Path("results") / f"train_{cfg['name']}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}; checkpoint in {ckpt_dir}")
    for m, v in result["metrics"].items():
        p = result["paired_vs_zeroshot"][m]
        print(f"  {m:11s} {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]   "
              f"vs zero-shot {p['diff']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] p={p['p_value']:.4f}")


if __name__ == "__main__":
    main()
