import torch

from clincot.methods.sdpo_loss import compute_dpo_loss, compute_sdpo_loss


def test_dpo_loss_shape():
    b = 4
    x = torch.randn(b)
    y = torch.randn(b)
    losses, _, _ = compute_dpo_loss(x, y, x, y, beta=0.1)
    assert losses.shape[0] == b


def test_sdpo_loss_shape():
    b = 4
    x = torch.randn(b)
    y = torch.randn(b)
    s1 = torch.rand(b)
    s2 = torch.rand(b)
    losses, _, _ = compute_sdpo_loss(x, y, x, y, s1, s2, beta=0.1)
    assert losses.shape[0] == b
