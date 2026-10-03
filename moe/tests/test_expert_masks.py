"""CPU tests for the ToMoE-style expert channel masks (olmoe_masks.py)."""
import pytest
import torch

from olmoe_masks import ExpertMask, masked_expert_forward


def test_topk_keep_fraction_and_binary_values():
    torch.manual_seed(0)
    m = ExpertMask(num_experts=4, dim=128, keep_frac=0.5, seed=0)
    mask = m.forward()
    assert mask.shape == (4, 128)
    assert torch.allclose(mask, mask.round())          # values are hard +/- fp wiggle
    assert torch.equal(m.keep_counts(), torch.full((4,), 64))


def test_keep_fraction_extremes():
    m = ExpertMask(2, 32, keep_frac=1.0)
    assert torch.equal(m.keep_counts(), torch.full((2,), 32))
    m = ExpertMask(2, 32, keep_frac=0.01)
    assert torch.equal(m.keep_counts(), torch.full((2,), 1))
    with pytest.raises(ValueError):
        ExpertMask(2, 32, keep_frac=1.5)


def test_mask_gradient_reaches_logits():
    m = ExpertMask(num_experts=3, dim=64, keep_frac=0.5, seed=1)
    m.forward().sum().backward()
    assert m.logits.grad is not None
    assert m.logits.grad.abs().sum().item() > 0.0


def test_masked_expert_forward():
    h = torch.ones(2, 5)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    out = masked_expert_forward(h, mask)
    assert out.tolist() == [[1.0, 0.0, 1.0, 0.0, 1.0]] * 2
    assert torch.equal(masked_expert_forward(h, torch.zeros(5)), torch.zeros(2, 5))
