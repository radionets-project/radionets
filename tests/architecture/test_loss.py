"""Tests for src/radionets/architecture/loss.py"""

import pytest
import torch

from radionets.architecture.loss import (
    MaskedSplitL1Loss,
    SplitL1Loss,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestSplitL1Loss:
    def test_default_reduction(self):
        loss_fn = SplitL1Loss()
        assert loss_fn.reduction == "mean"

    def test_reduction_sum(self):
        loss_fn = SplitL1Loss(reduction="sum")
        assert loss_fn.reduction == "sum"

    def test_reduction_none(self):
        loss_fn = SplitL1Loss(reduction="none")
        assert loss_fn.reduction == "none"

    def test_forward_mean(self):
        loss_fn = SplitL1Loss(reduction="mean")

        pred = torch.randn(4, 2, 16, 16)
        target = torch.randn(4, 2, 16, 16)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0

    def test_forward_sum(self):
        loss_fn = SplitL1Loss(reduction="sum")

        pred = torch.randn(4, 2, 16, 16)
        target = torch.randn(4, 2, 16, 16)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0

    def test_forward_zero_loss_identical(self):
        loss_fn = SplitL1Loss()

        pred = torch.randn(4, 2, 16, 16)
        loss = loss_fn(pred, pred.clone())

        assert loss == 0.0

    def test_forward_separates_amp_phase(self):
        loss_fn = SplitL1Loss()
        pred = torch.zeros(4, 2, 16, 16)
        target = torch.full((4, 2, 16, 16), 2.0)

        loss = loss_fn(pred, target)

        assert torch.isclose(loss, torch.tensor(4.0), atol=1e-4)

    @cuda
    def test_forward_cuda(self):
        loss_fn = SplitL1Loss().cuda()

        pred = torch.randn(4, 2, 16, 16).cuda()
        target = torch.randn(4, 2, 16, 16).cuda()
        loss = loss_fn(pred, target)

        assert loss >= 0

    def test_forward_large_values(self):
        loss_fn = SplitL1Loss()

        pred = torch.zeros(4, 2, 16, 16)
        target = torch.full((4, 2, 16, 16), 100.0)

        loss = loss_fn(pred, target)

        assert torch.isclose(loss, torch.tensor(200.0), atol=1.0)

    def test_forward_finite(self):
        loss_fn = SplitL1Loss()

        pred = torch.randn(4, 2, 16, 16) * 100.0
        target = torch.randn(4, 2, 16, 16) * 100.0
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)


class TestMaskedSplitL1Loss:
    def test_default_parameters(self):
        loss_fn = MaskedSplitL1Loss()

        assert loss_fn.radius == 30
        assert loss_fn.center is None

    def test_kwargs(self):
        loss_fn = MaskedSplitL1Loss(center=[10, 10], radius=15)

        assert loss_fn.center == [10, 10]
        assert loss_fn.radius == 15

    def test_forward_mask(self):
        loss_fn = MaskedSplitL1Loss(radius=5)

        pred = torch.randn(4, 2, 16, 16)
        target = torch.randn(4, 2, 16, 16)
        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0

    def test_forward_zero_loss_identical(self):
        loss_fn = MaskedSplitL1Loss(radius=5)

        pred = torch.randn(4, 2, 16, 16)
        loss = loss_fn(pred, pred.clone())

        assert loss == 0.0

    def test_masked_vs_unmasked(self):
        loss_fn = SplitL1Loss(reduction="mean")
        masked_loss_fn = MaskedSplitL1Loss(radius=5)

        pred = torch.randn(4, 2, 16, 16)
        target = torch.randn(4, 2, 16, 16)

        unmasked = loss_fn(pred, target)
        masked = masked_loss_fn(pred, target)

        assert masked != unmasked

    def test_forward_different_radii(self):
        loss_fn_small = MaskedSplitL1Loss(radius=2)
        loss_fn_large = MaskedSplitL1Loss(radius=10)

        pred = torch.randn(4, 2, 32, 32)
        target = torch.randn(4, 2, 32, 32)

        loss_small = loss_fn_small(pred, target)
        loss_large = loss_fn_large(pred, target)

        assert loss_small != loss_large

    def test_mask_center_default(self):
        loss_fn = MaskedSplitL1Loss(radius=5)

        pred = torch.randn(4, 2, 32, 32)
        target = torch.randn(4, 2, 32, 32)

        loss = loss_fn(pred, target)

        assert loss >= 0

    @cuda
    def test_forward_cuda(self):
        loss_fn = MaskedSplitL1Loss(radius=5)

        pred = torch.randn(4, 2, 16, 16).cuda()
        target = torch.randn(4, 2, 16, 16).cuda()
        loss = loss_fn(pred, target)

        assert loss >= 0

    def test_forward_finite(self):
        loss_fn = MaskedSplitL1Loss(radius=5)

        pred = torch.randn(4, 2, 16, 16) * 100.0
        target = torch.randn(4, 2, 16, 16) * 100.0
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
