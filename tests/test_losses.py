import math

import torch

from finqa.train.losses import false_negative_mask, info_nce

NO_MASK = lambda b, n: torch.zeros(b, n, dtype=torch.bool)


def test_perfectly_separated_batch_has_near_zero_loss():
    eye = torch.eye(3)
    assert info_nce(eye, eye, NO_MASK(3, 3), scale=20.0).item() < 1e-6


def test_uninformative_scores_give_log_n():
    q = torch.ones(4, 8) / math.sqrt(8)
    d = torch.ones(4, 8) / math.sqrt(8)
    assert abs(info_nce(q, d, NO_MASK(4, 4), scale=20.0).item() - math.log(4)) < 1e-5


def test_masking_a_duplicate_positive_removes_its_penalty():
    # doc 1 is identical to doc 0 and is ALSO correct for query 0: without the mask the
    # query is split between two identical docs (loss log 2); with it the loss vanishes.
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    unmasked = info_nce(q[:1], d, NO_MASK(1, 2), scale=20.0).item()
    mask = torch.tensor([[False, True]])
    masked = info_nce(q[:1], d, mask, scale=20.0).item()
    assert abs(unmasked - math.log(2)) < 1e-5
    assert masked < 1e-6


def test_false_negative_mask_marks_only_foreign_positives():
    positives = {"q0": {"a", "b"}, "q1": {"c"}}
    # columns: 0=a (q0's pair), 1=c (q1's pair), 2=b (hard neg), 3=a again (hard neg)
    mask = false_negative_mask(["q0", "q1"], ["a", "c", "b", "a"], positives)
    assert mask.shape == (2, 4)
    assert not mask[0, 0]          # own pair is never masked
    assert not mask[0, 1]          # c is not a positive of q0
    assert mask[0, 2]              # b is also correct for q0
    assert mask[0, 3]              # a repeated elsewhere is still q0's positive
    assert not mask[1, 1] and not mask[1, 0] and not mask[1, 2] and not mask[1, 3]
