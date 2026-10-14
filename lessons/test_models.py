"""Quick smoke tests: python -m unittest discover -s lessons."""
import unittest
import torch

from gpt_model import MiniGPT, CausalSelfAttention


class TestMiniGPT(unittest.TestCase):
    def test_shapes_and_backprop(self):
        torch.manual_seed(0)
        model = MiniGPT(vocab_size=11, context=8, width=16, heads=4, layers=1)
        x = torch.randint(0, 11, (2, 8))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 8, 11))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.token_emb.weight.grad)

    def test_causal_mask(self):
        torch.manual_seed(0)
        attn = CausalSelfAttention(width=8, heads=2).eval()
        a = torch.randn(1, 5, 8)
        b = a.clone()
        b[:, 4, :] += 100  # Future token must not affect earlier outputs.
        with torch.no_grad():
            output_a, output_b = attn(a), attn(b)
        torch.testing.assert_close(output_a[:, :4], output_b[:, :4])

    def test_generate(self):
        model = MiniGPT(vocab_size=11, context=4, width=16, heads=4, layers=1)
        output = model.generate(torch.zeros((1, 1), dtype=torch.long), 6)
        self.assertEqual(tuple(output.shape), (1, 7))


if __name__ == "__main__":
    unittest.main()
