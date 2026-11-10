"""Contrastive loss with in-batch negatives (InfoNCE)."""
import torch
import torch.nn.functional as F


def false_negative_mask(qids: list[str], doc_ids: list[str],
                        positives: dict[str, set[str]]) -> torch.Tensor:
    """mask[i, j] is True when doc j is a known positive for query i but is NOT query i's own
    paired document (column i). Those columns must not be treated as negatives.

    doc_ids = the batch's positives (columns 0..B-1, aligned with qids) followed by any hard
    negatives. A doc that is correct for another query in the batch would otherwise be pushed
    away from this query, which punishes right answers.
    """
    mask = torch.zeros(len(qids), len(doc_ids), dtype=torch.bool)
    for i, q in enumerate(qids):
        pos = positives.get(q, set())
        for j, d in enumerate(doc_ids):
            if j != i and d in pos:
                mask[i, j] = True
    return mask


def group_cross_entropy(scores: torch.Tensor) -> torch.Tensor:
    """scores: [B, G]; column 0 is each query's positive, the rest are its sampled negatives.

    Softmax over the group, so the positive must outscore every negative for the same query.
    """
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, labels)


def info_nce(q: torch.Tensor, d: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    """q: [B, E] queries; d: [N, E] docs with d[i] the positive of q[i] (N >= B).

    Cross-entropy over cosine similarities x scale, where masked columns are removed from the
    softmax. scale = 1 / temperature.
    """
    scores = (q @ d.T) * scale
    scores = scores.masked_fill(mask.to(scores.device), float("-inf"))
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(scores, labels)
