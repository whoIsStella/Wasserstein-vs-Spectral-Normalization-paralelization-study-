# Experiment design

## Reconstruction status

This benchmark is a reconstruction, not recovered NERSC code.

The surviving project context establishes three useful facts:

1. WGAN-GP training code survives in the RoHDE lineage.
2. Separate HPC work survives with Slurm batch scripts and reproducible benchmark output.
3. The original repository name identifies spectral normalization and parallelization as the comparison target.

The exact original architecture, dataset, launch configuration, and reported result are not available here.

## Comparison contract

The first reconstructed benchmark holds constant:

- generator architecture;
- critic architecture width/depth;
- latent dimension;
- data dimension;
- optimizer and learning rate;
- critic updates per generator update;
- local batch size;
- synthetic data distribution;
- random seed policy.

The changed factor is critic regularization:

| Mode | Critic regularization | Gradient penalty |
| --- | --- | --- |
| `wgan_gp` | none | yes |
| `spectral_norm` | spectral normalization on every linear layer | no |

Both use the Wasserstein critic objective.

## Scaling dimensions

Run at minimum:

- CPU smoke test
- 1 GPU
- 2 GPUs
- 4 GPUs when available

Use `torchrun` so multi-GPU runs use DistributedDataParallel rather than single-process DataParallel.

Record hardware and software versions with every result.

## Performance metrics

Primary:

- measured samples / second
- mean training step time
- peak CUDA memory

Secondary:

- critic loss
- generator loss
- gradient-penalty magnitude

Model-quality metrics are intentionally deferred until a public dataset is selected.

## Repetition

Do not treat one timing run as a scaling result.

A proper sweep should use repeated trials after warmup and report median plus dispersion.

## Future dataset adapter

The synthetic sampler exists to make infrastructure testing deterministic and portable.

A real dataset adapter should define:

- source and license;
- exact preprocessing;
- train/evaluation split;
- tensor shape;
- sampling policy;
- quality metric.

Do not silently reuse private EMG data in this public benchmark.
