# Wasserstein vs. Spectral Normalization Parallelization Study

A portable reconstruction of an earlier multi-GPU GAN experiment.

The original runs used NERSC infrastructure that is no longer available to me. The original experiment code was not preserved in this repository, so this project does not pretend to be a recovered copy. It rebuilds the comparison from the surviving research context and makes the benchmark runnable on a laptop, a single GPU, or a multi-GPU machine with `torchrun`.

## Question

What changes when the critic is regularized with:

1. **Wasserstein gradient penalty**, or
2. **spectral normalization**

as training scales from one process to multiple GPUs?

The benchmark records throughput, step time, critic/generator loss, gradient-penalty cost where applicable, and CUDA peak memory.

## Current benchmark

The reconstruction deliberately starts with a synthetic vector dataset so infrastructure and scaling behavior can be measured without depending on inaccessible private data or NERSC storage.

Both modes use the same generator, critic width, latent dimension, Wasserstein critic objective, optimizer, batch size, and number of critic updates.

The difference is the critic regularizer:

- `wgan_gp`: unconstrained linear layers + gradient penalty
- `spectral_norm`: spectral normalization on critic linear layers, no gradient penalty

This isolates the regularization path more cleanly than changing several GAN components at once.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Smoke test

```bash
python -m gan_bench.train --mode wgan_gp --steps 20 --device cpu
python -m gan_bench.train --mode spectral_norm --steps 20 --device cpu
```

## Single GPU

```bash
python -m gan_bench.train \
  --mode wgan_gp \
  --device cuda \
  --steps 1000 \
  --batch-size 256 \
  --output results/wgan-gp-1gpu.json
```

Then run the same configuration with:

```bash
python -m gan_bench.train \
  --mode spectral_norm \
  --device cuda \
  --steps 1000 \
  --batch-size 256 \
  --output results/spectral-norm-1gpu.json
```

## Multi-GPU

```bash
torchrun --standalone --nproc-per-node=4 \
  -m gan_bench.train \
  --mode wgan_gp \
  --device cuda \
  --steps 1000 \
  --batch-size 256 \
  --output results/wgan-gp-4gpu.json
```

Repeat with `--mode spectral_norm`.

`batch-size` is per process, so global batch size is:

```text
batch_size_per_process * world_size
```

## Slurm

A generic Slurm launcher is under `scripts/slurm_torchrun.sbatch`.

Cluster account, partition, GPU constraint, and module setup are intentionally left as environment-specific inputs instead of hard-coding old NERSC settings.

## Output

Rank 0 writes one JSON document containing:

- mode
- world size
- device
- training steps
- per-process and global batch size
- total measured samples
- elapsed measured training time
- samples per second
- mean step time
- mean generator loss
- mean critic loss
- mean gradient penalty
- CUDA peak memory when available
- benchmark configuration

## What this does not claim

- It is not a reproduction of the original NERSC result.
- Synthetic data is not a stand-in for the original scientific dataset.
- A faster benchmark does not imply a better generative model.
- Scaling results should be compared only across runs with matched hardware and configuration.

## Next research steps

1. run matched 1-GPU and multi-GPU sweeps;
2. add repeated trials and confidence intervals;
3. plug in a documented public dataset;
4. add distribution-quality metrics appropriate to that dataset;
5. compare DDP communication overhead and regularizer cost separately;
6. only then revisit the original EMG use case if the required data can be used reproducibly.

See `docs/experiment-design.md` for the comparison contract.
