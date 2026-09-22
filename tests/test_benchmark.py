import torch

from gan_bench.models import Critic, Generator
from gan_bench.train import gradient_penalty, sample_real


def test_models_produce_expected_shapes():
    generator = Generator(latent_dim=8, data_dim=16, hidden_dim=32)
    critic = Critic(data_dim=16, hidden_dim=32, use_spectral_norm=True)

    z = torch.randn(4, 8)
    fake = generator(z)

    assert fake.shape == (4, 16)
    assert critic(fake).shape == (4,)


def test_gradient_penalty_is_finite():
    critic = Critic(data_dim=8, hidden_dim=16, use_spectral_norm=False)
    real = sample_real(4, 8, torch.device("cpu"))
    fake = torch.randn_like(real)

    penalty = gradient_penalty(critic, real, fake)

    assert torch.isfinite(penalty)
    assert penalty.item() >= 0
