from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import (
    barrier,
    close_distributed,
    init_distributed,
    mean_across_ranks,
)
from .models import Critic, Generator


@dataclass(frozen=True)
class Config:
    mode: str
    steps: int
    warmup_steps: int
    batch_size: int
    latent_dim: int
    data_dim: int
    hidden_dim: int
    critic_steps: int
    gp_lambda: float
    lr: float
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("wgan_gp", "spectral_norm"), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--data-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def choose_device(requested: str, local_rank: int) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def sample_real(batch_size: int, data_dim: int, device: torch.device) -> torch.Tensor:
    component = torch.randint(0, 4, (batch_size, 1), device=device)
    centers = torch.tensor((-1.5, -0.5, 0.5, 1.5), device=device)
    mean = centers[component].expand(batch_size, data_dim)
    return mean + 0.35 * torch.randn(batch_size, data_dim, device=device)


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    batch = real.shape[0]
    alpha = torch.rand(batch, 1, device=real.device)
    mixed = alpha * real + (1.0 - alpha) * fake
    mixed.requires_grad_(True)

    score = critic(mixed)
    gradient = torch.autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = gradient.reshape(batch, -1).norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def main() -> None:
    args = parse_args()

    rank_guess = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    provisional_device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    rank, local_rank, world_size = init_distributed(provisional_device)
    device = choose_device(args.device, local_rank)

    config = Config(
        mode=args.mode,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        data_dim=args.data_dim,
        hidden_dim=args.hidden_dim,
        critic_steps=args.critic_steps,
        gp_lambda=args.gp_lambda,
        lr=args.lr,
        seed=args.seed,
        device=device.type,
    )

    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    generator: nn.Module = Generator(args.latent_dim, args.data_dim, args.hidden_dim).to(device)
    critic: nn.Module = Critic(
        args.data_dim,
        args.hidden_dim,
        use_spectral_norm=args.mode == "spectral_norm",
    ).to(device)

    if world_size > 1:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        generator = DDP(generator, **ddp_kwargs)
        critic = DDP(critic, **ddp_kwargs)

    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    measured_step_times: list[float] = []
    generator_losses: list[float] = []
    critic_losses: list[float] = []
    gp_values: list[float] = []

    total_steps = args.warmup_steps + args.steps
    barrier()

    for step in range(total_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        last_c_loss = torch.tensor(0.0, device=device)
        last_gp = torch.tensor(0.0, device=device)

        for _ in range(args.critic_steps):
            real = sample_real(args.batch_size, args.data_dim, device)
            z = torch.randn(args.batch_size, args.latent_dim, device=device)
            fake = generator(z).detach()

            c_opt.zero_grad(set_to_none=True)
            real_score = critic(real)
            fake_score = critic(fake)
            wasserstein_loss = fake_score.mean() - real_score.mean()

            if args.mode == "wgan_gp":
                last_gp = gradient_penalty(critic, real, fake)
                last_c_loss = wasserstein_loss + args.gp_lambda * last_gp
            else:
                last_gp = torch.tensor(0.0, device=device)
                last_c_loss = wasserstein_loss

            last_c_loss.backward()
            c_opt.step()

        z = torch.randn(args.batch_size, args.latent_dim, device=device)
        g_opt.zero_grad(set_to_none=True)
        generated = generator(z)
        g_loss = -critic(generated).mean()
        g_loss.backward()
        g_opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_start

        if step >= args.warmup_steps:
            measured_step_times.append(elapsed)
            generator_losses.append(float(g_loss.detach().item()))
            critic_losses.append(float(last_c_loss.detach().item()))
            gp_values.append(float(last_gp.detach().item()))

    barrier()

    local_elapsed = sum(measured_step_times)
    mean_step = local_elapsed / max(1, len(measured_step_times))
    global_samples = args.steps * args.batch_size * world_size
    samples_per_second = global_samples / local_elapsed if local_elapsed else math.inf

    mean_g = sum(generator_losses) / max(1, len(generator_losses))
    mean_c = sum(critic_losses) / max(1, len(critic_losses))
    mean_gp = sum(gp_values) / max(1, len(gp_values))

    result = {
        "mode": args.mode,
        "world_size": world_size,
        "device": device.type,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size_per_process": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "measured_samples": global_samples,
        "elapsed_seconds": local_elapsed,
        "samples_per_second": samples_per_second,
        "mean_step_seconds": mean_step,
        "mean_generator_loss": mean_g,
        "mean_critic_loss": mean_c,
        "mean_gradient_penalty": mean_gp,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "config": asdict(config),
    }

    if rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print(encoded)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")

    close_distributed()


if __name__ == "__main__":
    main()
