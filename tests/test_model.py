from __future__ import annotations

import unittest

import torch

from src.models.memae.model import MemAE, memae_loss


class MemAEModelTests(unittest.TestCase):
    def test_forward_backward_supports_train_batch_size_one(self) -> None:
        model = MemAE(4, latent_dim=2, memory_size=3, hidden_dims=[5], shrink_threshold=0.0)
        model.train()
        batch = torch.randn(1, 4)

        x_hat, _, _, attn = model(batch)
        loss, _, _, _ = memae_loss(batch, x_hat, attn, entropy_weight=0.0)
        loss.backward()

        self.assertEqual(x_hat.shape, batch.shape)
        self.assertTrue(torch.isfinite(x_hat).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_memae_loss_rejects_broadcastable_shape_mismatch(self) -> None:
        x = torch.zeros(2, 3)
        x_hat = torch.zeros(1, 3)
        attn = torch.full((2, 4), 0.25)

        with self.assertRaisesRegex(ValueError, "x_hat shape"):
            memae_loss(x, x_hat, attn)

    def test_single_memory_slot_diversity_loss_is_zero(self) -> None:
        model = MemAE(3, latent_dim=2, memory_size=1, hidden_dims=[4])

        loss = model.memory_diversity_loss()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
