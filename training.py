"""Unified training utilities for paired super-resolution experiments.

Sections:
    GENERAL: EMA, checkpoints, RNG state, and shared helpers.
    REGRESSION: L1/L2 pixelwise regression via train_pbp_regression.
    CFM: Conditional Flow Matching via train_cgm.

Expected batch format:
    (x_src, x_hr)

Regression model API:
    generator(x_src) -> x_pred

CFM field API:
    field(t, x_t, x_src) -> velocity, same shape as x_hr

The CFM objective here uses the paired straight-line path:
    x_t = (1 - t) * x_src + t * x_hr
    target_velocity = x_hr - x_src

It is a conditional Flow Matching baseline, not the MGM objective.
"""
from __future__ import annotations

import copy
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


#----- GENERAL: EMA, checkpoints, RNG, shared helpers --------#

def update_ema(ema_model, model, decay: float = 0.999) -> None:
    """Update EMA parameters and copy buffers from the live model."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1).")

    with torch.no_grad():
        source_parameters = dict(model.named_parameters())
        for name, ema_parameter in ema_model.named_parameters():
            ema_parameter.lerp_(source_parameters[name], 1.0 - decay)

        source_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_model.named_buffers():
            ema_buffer.copy_(source_buffers[name])


def make_ema(model):
    """Create a frozen EMA copy of a model."""
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    return ema_model


def save_checkpoint_atomic(state: dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint safely, replacing the destination only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)

    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint(checkpoint, map_location="cpu") -> dict[str, Any]:
    """Load a checkpoint path or accept a loaded checkpoint dictionary."""
    if isinstance(checkpoint, (str, Path)):
        return torch.load(checkpoint, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("checkpoint must be a path or dictionary.")


def _rng_state() -> dict[str, Any]:
    """Capture RNG states at an epoch boundary."""
    return {
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    """Restore RNG states from a checkpoint."""
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    if state.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _make_optimizer(model, lr):
    """Shared Adam setup; override betas for a method if needed."""
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))


def _batch_to_device(batch, device):
    """Move a paired batch to device and float32."""
    x_src, x_hr = batch
    x_src = x_src.to(device=device, dtype=torch.float32, non_blocking=True)
    x_hr = x_hr.to(device=device, dtype=torch.float32, non_blocking=True)
    if x_src.shape != x_hr.shape:
        raise ValueError(
            f"Expected equal source/target shapes, got "
            f"{tuple(x_src.shape)} and {tuple(x_hr.shape)}."
        )
    return x_src, x_hr


def _clip_gradients(model, max_grad_norm):
    """Clip gradients if requested and return the pre-clipping norm."""
    if max_grad_norm is not None:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    squared_norm = None
    for parameter in model.parameters():
        if parameter.grad is not None:
            term = parameter.grad.detach().square().sum()
            squared_norm = term if squared_norm is None else squared_norm + term

    if squared_norm is None:
        return torch.tensor(0.0)
    return squared_norm.sqrt()


def _log_batch(method, epoch, total_epochs, batch_index, total_batches,
               loss, extra_metric, grad_norm):
    """Consistent batch-level progress logging for both training methods."""
    total = f"/{total_batches}" if total_batches is not None else ""
    print(
        f"{method} | Epoch [{epoch}/{total_epochs}] "
        f"Batch [{batch_index}{total}] | "
        f"Loss: {loss:.6f} | {extra_metric} | Grad: {grad_norm:.4f}",
        flush=True,
    )


def _checkpoint_state(
    *, epoch, global_step, objective, model, ema_model, optimizer, history, config
):
    """Build a common checkpoint schema, with method-specific model key names."""
    if objective == "regression":
        model_key = "generator_state_dict"
        ema_key = "generator_ema_state_dict"
    elif objective == "conditional_flow_matching":
        model_key = "field_state_dict"
        ema_key = "field_ema_state_dict"
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return {
        "epoch": epoch,
        "global_step": global_step,
        "objective": objective,
        "config": config,
        model_key: model.state_dict(),
        ema_key: ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": list(history),
        **_rng_state(),
    }


def _restore_common(
    *, checkpoint, model, ema_model, optimizer, objective, restore_rng
):
    """Restore the selected method's model, EMA, optimizer, history, and RNG."""
    saved = load_checkpoint(checkpoint, map_location="cpu")
    if saved.get("objective") not in (None, objective):
        raise ValueError(
            f"Checkpoint objective is {saved.get('objective')!r}, expected {objective!r}."
        )

    if objective == "regression":
        model_key = "generator_state_dict"
        ema_key = "generator_ema_state_dict"
    else:
        model_key = "field_state_dict"
        ema_key = "field_ema_state_dict"

    model.load_state_dict(saved[model_key])
    ema_model.load_state_dict(saved[ema_key])

    if "optimizer_state_dict" not in saved:
        raise ValueError("Checkpoint has no optimizer state; exact resume is unavailable.")
    optimizer.load_state_dict(saved["optimizer_state_dict"])

    if restore_rng:
        _restore_rng(saved)

    return int(saved.get("epoch", 0)), int(saved.get("global_step", 0)), list(
        saved.get("history", [])
    )


def _validate(
    model, val_dataloader, device, method, ode_steps, ode_solver
):
    """Call the unified validation module regardless of training objective."""
    try:
        from .validation import validate_super_resolution
    except ImportError:
        from validation import validate_super_resolution

    return validate_super_resolution(
        model,
        val_dataloader,
        device=device,
        clamp=True,
        method=method,
        ode_steps=ode_steps,
        ode_solver=ode_solver,
    )


def _print_epoch_summary(row):
    """Print shared epoch summary and available validation metrics."""
    print(
        f"Train | Loss: {row['train_loss']:.6f} | "
        f"MAE: {row['train_l1']:.6f} | MSE: {row['train_mse']:.6f} | "
        f"Grad: {row['grad_norm']:.4f} | "
        f"Time: {row['epoch_wall_seconds']:.1f}s",
        flush=True,
    )

    if "ema_MAE" in row:
        print(
            f"EMA validation | MAE: {row['ema_MAE']:.6f} | "
            f"MSE: {row['ema_MSE']:.6f} | "
            f"PSNR: {row['ema_PSNR (dB)']:.3f} dB | "
            f"SSIM: {row['ema_SSIM']:.5f} | "
            f"Outside [0,1]: {row['ema_Outside [0,1] (%)']:.2f}% | "
            f"Time: {row['ema_evaluation_seconds']:.2f}s",
            flush=True,
        )


#----- REGRESSION: paired L1/L2 pixel regression --------#

def train_pbp_regression(
    generator,
    dataloader,
    epochs=10,
    pixel_loss="l1",
    lr=1e-5,
    device="cuda",
    max_grad_norm=1.0,
    ema_decay=0.999,
    val_dataloader=None,
    checkpoint=None,
    checkpoint_path=None,
    checkpoint_every=1,
    log_every=None,
    restore_rng=True,
):
    """Train a paired generator with L1 or L2 pixel loss.

    Generator API: generator(x_src) -> x_pred.
    Resume checkpoints are saved at epoch boundaries and include optimizer state.
    """
    pixel_loss = pixel_loss.lower()
    if pixel_loss not in {"l1", "l2", "mse"}:
        raise ValueError("pixel_loss must be 'l1', 'l2', or 'mse'.")
    if epochs < 0 or checkpoint_every < 1:
        raise ValueError("epochs must be >= 0 and checkpoint_every >= 1.")

    device = torch.device(device)
    generator = generator.to(device=device, dtype=torch.float32)
    generator_ema = make_ema(generator)
    optimizer = _make_optimizer(generator, lr)

    history = []
    global_step = 0
    start_epoch = 0

    if checkpoint is not None:
        start_epoch, global_step, history = _restore_common(
            checkpoint=checkpoint,
            model=generator,
            ema_model=generator_ema,
            optimizer=optimizer,
            objective="regression",
            restore_rng=restore_rng,
        )
        print(f"Resuming regression after epoch {start_epoch}.", flush=True)

    state = None
    total_epochs = start_epoch + epochs

    for epoch_offset in range(epochs):
        epoch = start_epoch + epoch_offset + 1
        generator.train()
        epoch_start = time.perf_counter()

        loss_sum = 0.0
        abs_sum = 0.0
        sq_sum = 0.0
        grad_sum = 0.0
        value_count = 0
        batch_count = 0

        try:
            total_batches = len(dataloader)
        except (TypeError, AttributeError):
            total_batches = None

        log_interval = (
            max(1, (total_batches + 9) // 10)
            if log_every is None and total_batches is not None
            else (log_every or 100)
        )

        print(
            f"\nStarting regression epoch [{epoch}/{total_epochs}] "
            f"with {pixel_loss.upper()} loss; logging every {log_interval} batches",
            flush=True,
        )

        for batch_index, batch in enumerate(dataloader, start=1):
            x_src, x_hr = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            prediction = generator(x_src)
            if prediction.shape != x_hr.shape:
                raise ValueError("Generator output shape must match HR target shape.")

            if pixel_loss == "l1":
                loss = F.l1_loss(prediction, x_hr)
            else:
                loss = F.mse_loss(prediction, x_hr)

            loss.backward()
            grad_norm = _clip_gradients(generator, max_grad_norm)
            optimizer.step()
            update_ema(generator_ema, generator, ema_decay)
            global_step += 1

            difference = prediction.detach() - x_hr
            count = x_hr.numel()
            loss_sum += loss.detach().item() * count
            abs_sum += difference.abs().sum().item()
            sq_sum += difference.square().sum().item()
            grad_sum += float(grad_norm.detach().item())
            value_count += count
            batch_count += 1

            should_log = (
                batch_index == 1
                or batch_index % log_interval == 0
                or (total_batches is not None and batch_index == total_batches)
            )
            if should_log:
                _log_batch(
                    "Regression",
                    epoch,
                    total_epochs,
                    batch_index,
                    total_batches,
                    loss.item(),
                    f"MAE: {difference.abs().mean().item():.6f}",
                    float(grad_norm.detach().item()),
                )

        if not batch_count:
            raise ValueError("Training dataloader is empty.")

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "objective": "regression",
            "pixel_loss": pixel_loss,
            "train_loss": loss_sum / value_count,
            "train_l1": abs_sum / value_count,
            "train_mse": sq_sum / value_count,
            "grad_norm": grad_sum / batch_count,
            "epoch_wall_seconds": time.perf_counter() - epoch_start,
        }

        if val_dataloader is not None:
            metrics = _validate(
                generator_ema, val_dataloader, device, "direct", 1, "euler"
            )
            row.update({f"ema_{key}": value for key, value in metrics.items()})

        history.append(row)
        print(f"Regression epoch [{epoch}/{total_epochs}] complete", flush=True)
        _print_epoch_summary(row)

        state = _checkpoint_state(
            epoch=epoch,
            global_step=global_step,
            objective="regression",
            model=generator,
            ema_model=generator_ema,
            optimizer=optimizer,
            history=history,
            config={
                "pixel_loss": pixel_loss,
                "lr": lr,
                "max_grad_norm": max_grad_norm,
                "ema_decay": ema_decay,
                "checkpoint_every": checkpoint_every,
            },
        )

        if checkpoint_path and (
            epoch % checkpoint_every == 0 or epoch_offset == epochs - 1
        ):
            save_checkpoint_atomic(state, checkpoint_path)
            print(f"Regression checkpoint saved: {checkpoint_path}", flush=True)

        print("", flush=True)

    if state is None:
        state = _checkpoint_state(
            epoch=start_epoch,
            global_step=global_step,
            objective="regression",
            model=generator,
            ema_model=generator_ema,
            optimizer=optimizer,
            history=history,
            config={"pixel_loss": pixel_loss, "lr": lr, "ema_decay": ema_decay},
        )

    return generator, generator_ema, history, state


#----- CFM: conditional Flow Matching and ODE-trained field --------#

def train_cfm(
    field,
    dataloader,
    epochs=10,
    lr=1e-4,
    device="cuda",
    max_grad_norm=1.0,
    ema_decay=0.999,
    val_dataloader=None,
    validation_steps=20,
    ode_solver="euler",
    checkpoint=None,
    checkpoint_path=None,
    checkpoint_every=1,
    validate_every=5,
    log_every=None,
    restore_rng=True,
):
    """Train a conditional Flow Matching velocity field.

    Field API: field(t, x_t, x_src) -> velocity.
    Data loader: paired (x_src, x_hr) with matching shapes.

    The paired bridge is x_t=(1-t)*x_src+t*x_hr and target velocity is x_hr-x_src.
    During validation, the field is integrated from t=0 to t=1 using an ODE solver.
    """
    if epochs < 0 or checkpoint_every < 1:
        raise ValueError("epochs must be >= 0 and checkpoint_every >= 1.")
    if validation_steps < 1:
        raise ValueError("validation_steps must be >= 1.")
    if ode_solver not in {"euler", "heun"}:
        raise ValueError("ode_solver must be 'euler' or 'heun'.")

    device = torch.device(device)
    field = field.to(device=device, dtype=torch.float32)
    field_ema = make_ema(field)

    # Keep the MGM optimizer convention for a closer experimental setup.
    optimizer = _make_optimizer(field, lr)

    history = []
    global_step = 0
    start_epoch = 0

    if checkpoint is not None:
        start_epoch, global_step, history = _restore_common(
            checkpoint=checkpoint,
            model=field,
            ema_model=field_ema,
            optimizer=optimizer,
            objective="conditional_flow_matching",
            restore_rng=restore_rng,
        )
        print(f"Resuming CFM after epoch {start_epoch}.", flush=True)

    state = None
    total_epochs = start_epoch + epochs

    for epoch_offset in range(epochs):
        epoch = start_epoch + epoch_offset + 1
        field.train()
        epoch_start = time.perf_counter()

        loss_sum = 0.0
        abs_sum = 0.0
        sq_sum = 0.0
        grad_sum = 0.0
        value_count = 0
        batch_count = 0

        try:
            total_batches = len(dataloader)
        except (TypeError, AttributeError):
            total_batches = None

        log_interval = (
            max(1, (total_batches + 9) // 10)
            if log_every is None and total_batches is not None
            else (log_every or 100)
        )

        print(
            f"\nStarting CFM epoch [{epoch}/{total_epochs}]; "
            f"ODE validation steps={validation_steps}; "
            f"logging every {log_interval} batches",
            flush=True,
        )

        for batch_index, batch in enumerate(dataloader, start=1):
            x_src, x_hr = _batch_to_device(batch, device)
            batch_size = x_hr.shape[0]

            # Sample independent interpolation times for each paired image.
            t = torch.rand(batch_size, device=device)
            t_view = t.view(batch_size, *([1] * (x_hr.ndim - 1)))

            # Construct a point on the straight-line path from LR to HR.
            x_t = (1.0 - t_view) * x_src + t_view * x_hr

            # The derivative of this linear path is constant.
            target_velocity = x_hr - x_src

            optimizer.zero_grad(set_to_none=True)
            predicted_velocity = field(t, x_t, x_src)

            if predicted_velocity.shape != target_velocity.shape:
                raise ValueError(
                    "Field output shape must match the target velocity shape."
                )

            loss = F.mse_loss(predicted_velocity, target_velocity)
            loss.backward()

            grad_norm = _clip_gradients(field, max_grad_norm)
            optimizer.step()
            update_ema(field_ema, field, ema_decay)
            global_step += 1

            # A one-step endpoint proxy is useful for progress monitoring only.
            endpoint_proxy = x_src + predicted_velocity.detach()
            difference = endpoint_proxy - x_hr
            count = x_hr.numel()

            loss_sum += loss.detach().item() * count
            abs_sum += difference.abs().sum().item()
            sq_sum += difference.square().sum().item()
            grad_sum += float(grad_norm.detach().item())
            value_count += count
            batch_count += 1

            should_log = (
                batch_index == 1
                or batch_index % log_interval == 0
                or (total_batches is not None and batch_index == total_batches)
            )
            if should_log:
                _log_batch(
                    "CFM",
                    epoch,
                    total_epochs,
                    batch_index,
                    total_batches,
                    loss.item(),
                    f"Endpoint-proxy MAE: {difference.abs().mean().item():.6f}",
                    float(grad_norm.detach().item()),
                )

        if not batch_count:
            raise ValueError("Training dataloader is empty.")

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "objective": "conditional_flow_matching",
            "train_loss": loss_sum / value_count,
            "train_l1": abs_sum / value_count,
            "train_mse": sq_sum / value_count,
            "grad_norm": grad_sum / batch_count,
            "epoch_wall_seconds": time.perf_counter() - epoch_start,
            "ode_steps": validation_steps,
            "ode_solver": ode_solver,
        }

        if val_dataloader is not None and (epoch % validate_every == 0):
            metrics = _validate(
                field_ema,
                val_dataloader,
                device,
                "cfm",
                validation_steps,
                ode_solver,
            )
            row.update({f"ema_{key}": value for key, value in metrics.items()})

            print(f"CFM epoch [{epoch}/{total_epochs}] complete", flush=True)
            _print_epoch_summary(row)

        history.append(row)

        state = _checkpoint_state(
            epoch=epoch,
            global_step=global_step,
            objective="conditional_flow_matching",
            model=field,
            ema_model=field_ema,
            optimizer=optimizer,
            history=history,
            config={
                "lr": lr,
                "max_grad_norm": max_grad_norm,
                "ema_decay": ema_decay,
                "validation_steps": validation_steps,
                "ode_solver": ode_solver,
                "checkpoint_every": checkpoint_every,
            },
        )

        if checkpoint_path and (
            epoch % checkpoint_every == 0 or epoch_offset == epochs - 1
        ):
            save_checkpoint_atomic(state, checkpoint_path)
            print(f"CFM checkpoint saved: {checkpoint_path}", flush=True)

        print("", flush=True)

    if state is None:
        state = _checkpoint_state(
            epoch=start_epoch,
            global_step=global_step,
            objective="conditional_flow_matching",
            model=field,
            ema_model=field_ema,
            optimizer=optimizer,
            history=history,
            config={
                "lr": lr,
                "ema_decay": ema_decay,
                "validation_steps": validation_steps,
                "ode_solver": ode_solver,
            },
        )

    return field, field_ema, history, state
