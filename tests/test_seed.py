import numpy as np
import torch

from finqa.utils.seed import set_seed


def test_same_seed_same_draws():
    set_seed(123)
    a = (np.random.rand(5), torch.rand(5))
    set_seed(123)
    b = (np.random.rand(5), torch.rand(5))
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_different_seed_different_draws():
    set_seed(1)
    a = torch.rand(5)
    set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)
