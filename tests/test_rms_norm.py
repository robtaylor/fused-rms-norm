"""Tests for Metal RMS normalization kernels.

Validates rms_norm and fused_add_rms_norm against PyTorch reference
implementations across dtypes and hidden sizes.
"""

import pytest
import torch

from kernels_test_utils import get_available_devices

import fused_rms_norm as ops

DEVICES = get_available_devices()

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
HIDDEN_SIZES = [128, 768, 2048, 4096]
NUM_TOKENS = [1, 7, 32]
EPSILON = 1e-6


def _ref_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Pure-PyTorch reference for RMS normalization."""
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + epsilon)
    return (input.float() * inv_rms * weight.float()).to(input.dtype)


def _ref_fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for fused residual add + RMS norm.

    Returns (normalized_output, updated_residual).
    """
    new_residual = residual.float() + input.float()
    variance = new_residual.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + epsilon)
    normalized = (new_residual * inv_rms * weight.float()).to(input.dtype)
    return normalized, new_residual.to(residual.dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_rms_norm(
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
) -> None:
    torch.manual_seed(42)
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    out = torch.empty_like(input)

    # Run kernel.
    ops.rms_norm(out, input, weight, EPSILON)

    # Run reference on CPU.
    ref = _ref_rms_norm(input.cpu(), weight.cpu(), EPSILON)

    # Compare.
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 2e-2, 2e-2

    torch.testing.assert_close(out.cpu(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_fused_add_rms_norm(
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
) -> None:
    torch.manual_seed(42)
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Compute reference on CPU BEFORE running kernel (kernel modifies in-place).
    ref_normalized, ref_residual = _ref_fused_add_rms_norm(
        input.cpu(), residual.cpu(), weight.cpu(), EPSILON
    )

    # Run kernel (modifies input and residual in-place).
    ops.fused_add_rms_norm(input, residual, weight, EPSILON)

    # Compare.
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 2e-2, 2e-2

    torch.testing.assert_close(
        residual.cpu(), ref_residual, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        input.cpu(), ref_normalized, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_rms_norm_weight_scaling(
    device: str,
    dtype: torch.dtype,
) -> None:
    """Verify that weight=1 gives pure RMS normalization."""
    hidden_size = 256
    num_tokens = 4
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight_ones = torch.ones(hidden_size, dtype=dtype, device=device)
    weight_twos = 2.0 * torch.ones(hidden_size, dtype=dtype, device=device)

    out_ones = torch.empty_like(input)
    out_twos = torch.empty_like(input)

    ops.rms_norm(out_ones, input, weight_ones, EPSILON)
    ops.rms_norm(out_twos, input, weight_twos, EPSILON)

    # weight=2 should produce exactly 2x the weight=1 result.
    torch.testing.assert_close(
        out_twos.cpu(), 2.0 * out_ones.cpu(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_fused_add_rms_norm_residual_accumulation(
    device: str,
    dtype: torch.dtype,
) -> None:
    """Verify residual is correctly accumulated (residual += input)."""
    hidden_size = 128
    num_tokens = 2
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)

    expected_residual = (residual + input).cpu()

    ops.fused_add_rms_norm(input, residual, weight, EPSILON)

    torch.testing.assert_close(
        residual.cpu(), expected_residual, atol=1e-5, rtol=1e-5
    )
