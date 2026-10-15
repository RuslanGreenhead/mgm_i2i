"""Unified validation for direct regression models and conditional Flow Matching.

Pixel FD is computed in raw pixel space from a fixed maximum number of
validation examples. It is a complementary distribution metric, not standard
Inception FID.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.linalg import sqrtm
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


# --------------------------- Pixel-space FD --------------------------- #

def calculate_pixel_frechet_distance(real_images, generated_images, eps=1e-6):
    """Fréchet distance between two image distributions in raw pixel space.

    Args:
        real_images: Tensor [N, C, H, W] (or [N, H, W]), values typically [0, 1].
        generated_images: Tensor with the same shape.
        eps: Diagonal covariance regularization for numerical stability.

    Returns:
        Scalar Pixel FD. This is not standard Inception FID.
    """
    if real_images.shape != generated_images.shape:
        raise ValueError(
            f"Real/generated shapes differ: {tuple(real_images.shape)} vs "
            f"{tuple(generated_images.shape)}."
        )

    real = real_images.detach().cpu().numpy().reshape(len(real_images), -1)
    generated = (
        generated_images.detach().cpu().numpy().reshape(len(generated_images), -1)
    )
    real = real.astype(np.float64, copy=False)
    generated = generated.astype(np.float64, copy=False)

    if len(real) < 2 or len(generated) < 2:
        raise ValueError("At least two images per set are required.")

    mu_real = real.mean(axis=0)
    mu_generated = generated.mean(axis=0)
    cov_real = np.atleast_2d(np.cov(real, rowvar=False))
    cov_generated = np.atleast_2d(np.cov(generated, rowvar=False))

    cov_real += np.eye(cov_real.shape[0]) * eps
    cov_generated += np.eye(cov_generated.shape[0]) * eps

    cov_mean = sqrtm(cov_real @ cov_generated)
    if not np.isfinite(cov_mean).all():
        raise ValueError("Pixel FD covariance square root is non-finite.")

    if np.iscomplexobj(cov_mean):
        max_imag = np.max(np.abs(cov_mean.imag))
        if max_imag > 1e-3:
            raise ValueError(
                f"Pixel FD covariance calculation is numerically unstable "
                f"(max imaginary component {max_imag:.3g})."
            )
        cov_mean = cov_mean.real

    distance = (
        np.sum((mu_real - mu_generated) ** 2)
        + np.trace(cov_real)
        + np.trace(cov_generated)
        - 2.0 * np.trace(cov_mean)
    )
    # Clip tiny negative values caused by floating-point round-off.
    return float(max(distance, 0.0))


# --------------------------- CFM: ODE sampling --------------------------- #

@torch.inference_mode()
def sample_cfm_ode(field, x_src, n_steps=20, method="euler"):
    """Integrate a conditional velocity field from t=0 to t=1.

    Required API: field(t, x_t, x_src) -> velocity.
    Euler uses one field evaluation per step; Heun uses two.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'.")

    batch_size = x_src.shape[0]
    x = x_src.clone()
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_now = torch.full(
            (batch_size,), step / n_steps, device=x.device, dtype=x.dtype
        )
        velocity = field(t_now, x, x_src)

        if method == "euler":
            x = x + dt * velocity
        else:
            x_predict = x + dt * velocity
            t_next = torch.full(
                (batch_size,),
                (step + 1) / n_steps,
                device=x.device,
                dtype=x.dtype,
            )
            next_velocity = field(t_next, x_predict, x_src)
            x = x + 0.5 * dt * (velocity + next_velocity)

    return x


# --------------------------- General validation --------------------------- #

def validate_super_resolution(
    predict_fn,
    dataloader,
    device="cuda",
    clamp=True,
    method="direct",
    ode_steps=20,
    ode_solver="euler",
    fd_max_samples=2000,
):
    """Evaluate paired super-resolution and optionally compute raw-pixel FD.

    MAE/MSE/PSNR/SSIM and out-of-range percentage use the full validation set.
    Pixel FD uses at most fd_max_samples examples, chosen in dataloader order.
    For comparable results, use a deterministic validation loader (shuffle=False)
    and the same subset for every method.

    Args:
        predict_fn: Direct generator if method='direct'; field if method='cfm'.
        dataloader: Yields (x_src, x_hr) batches.
        device: Evaluation device.
        clamp: Clamp predictions to [0, 1] for primary image metrics and Pixel FD.
        method: 'direct' or 'cfm'.
        ode_steps: Number of CFM integration steps.
        ode_solver: 'euler' or 'heun'.
        fd_max_samples: Maximum examples used for Pixel FD. Set to 0 to disable.
            At least 2 are required to calculate Pixel FD.

    Returns:
        Dictionary with MAE, MSE, PSNR, SSIM, Pixel FD, out-of-range percentage,
        number of images used for Pixel FD, and evaluation wall time.
    """
    if method not in {"direct", "cfm"}:
        raise ValueError("method must be 'direct' or 'cfm'.")
    if ode_steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    if fd_max_samples < 0:
        raise ValueError("fd_max_samples must be nonnegative.")

    device = torch.device(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    abs_error_sum = 0.0
    squared_error_sum = 0.0
    out_of_range_count = 0
    pixel_count = 0

    real_image_batches = []
    generated_image_batches = []
    fd_sample_count = 0

    was_training = getattr(predict_fn, "training", None)
    if hasattr(predict_fn, "eval"):
        predict_fn.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            for x_src, x_hr in dataloader:
                x_src = x_src.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                x_hr = x_hr.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                if method == "cfm":
                    raw_prediction = sample_cfm_ode(
                        predict_fn,
                        x_src,
                        n_steps=ode_steps,
                        method=ode_solver,
                    )
                else:
                    raw_prediction = predict_fn(x_src)

                if raw_prediction.shape != x_hr.shape:
                    raise ValueError(
                        f"Prediction shape {tuple(raw_prediction.shape)} != "
                        f"target shape {tuple(x_hr.shape)}."
                    )

                out_of_range_count += (
                    (raw_prediction < 0.0) | (raw_prediction > 1.0)
                ).sum().item()

                prediction = (
                    raw_prediction.clamp(0.0, 1.0) if clamp else raw_prediction
                )
                difference = prediction.double() - x_hr.double()
                abs_error_sum += difference.abs().sum().item()
                squared_error_sum += difference.square().sum().item()
                pixel_count += x_hr.numel()

                psnr_metric.update(prediction, x_hr)
                ssim_metric.update(prediction, x_hr)

                # Only retain the first fd_max_samples examples for Pixel FD.
                if fd_sample_count < fd_max_samples:
                    remaining = fd_max_samples - fd_sample_count
                    n_take = min(x_hr.shape[0], remaining)
                    if n_take > 0:
                        real_image_batches.append(x_hr[:n_take].detach().cpu())
                        generated_image_batches.append(
                            prediction[:n_take].detach().cpu()
                        )
                        fd_sample_count += n_take

        if pixel_count == 0:
            raise ValueError("Evaluation dataloader is empty.")

        if fd_max_samples >= 2 and fd_sample_count >= 2:
            real_images = torch.cat(real_image_batches, dim=0)
            generated_images = torch.cat(generated_image_batches, dim=0)
            pixel_fd = calculate_pixel_frechet_distance(
                real_images, generated_images
            )
        else:
            pixel_fd = float("nan")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - start_time

        return {
            "MAE": abs_error_sum / pixel_count,
            "MSE": squared_error_sum / pixel_count,
            "PSNR (dB)": psnr_metric.compute().item(),
            "SSIM": ssim_metric.compute().item(),
            "Pixel FD": pixel_fd,
            "Pixel FD samples": fd_sample_count,
            "Outside [0,1] (%)": 100.0 * out_of_range_count / pixel_count,
            "evaluation_seconds": elapsed_seconds,
        }

    finally:
        psnr_metric.reset()
        ssim_metric.reset()
        if was_training is True:
            predict_fn.train()
        elif was_training is False:
            predict_fn.eval()


# --------------------------- Compatible evaluation wrapper --------------------------- #

def evaluate_method(
    model,
    val_loader,
    device,
    identity=False,
    predict_fn=None,
    clamp=True,
    method="direct",
    ode_steps=20,
    ode_solver="euler",
    fd_max_samples=2000,
):
    """Return (metrics, timing), supporting direct models, CFM, and identity.

    Set identity=True to evaluate x_src as the prediction (e.g. bicubic input).
    """
    device = torch.device(device)

    if identity:
        def predictor(x_src):
            return x_src
    elif predict_fn is not None:
        predictor = predict_fn
    elif model is not None:
        predictor = model
    else:
        raise ValueError("Provide model, predict_fn, or identity=True.")

    all_metrics = validate_super_resolution(
        predictor,
        val_loader,
        device=device,
        clamp=clamp,
        method=method,
        ode_steps=ode_steps,
        ode_solver=ode_solver,
        fd_max_samples=fd_max_samples,
    )

    wall_seconds = all_metrics["evaluation_seconds"]
    metrics = {
        key: all_metrics[key]
        for key in (
            "MAE", "MSE", "PSNR (dB)", "SSIM", "Pixel FD",
            "Pixel FD samples", "Outside [0,1] (%)",
        )
    }
    timing = {
        "wall_seconds": wall_seconds,
        "wall_hours": wall_seconds / 3600.0,
        # Elapsed evaluation time on the selected device, not measured GPU utilization.
        "gpu_hours": wall_seconds / 3600.0 if device.type == "cuda" else 0.0,
    }
    return metrics, timing
