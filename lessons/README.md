# PyTorch MLP: trace one training step

This is an **independent Stage 1 teaching example**, not a transcription of Karpathy's notebooks.

## Run

```bash
python -m pip install torch
python lessons/pytorch_mlp_trace.py
```

CPU is sufficient. The script uses fixed random seeds and synthetic data.

## What to notice

1. **Forward:** `x (3,4) -> h (3,5) -> logits (3,2) -> scalar loss`. PyTorch's `nn.Linear(4,5)` stores weights with shape `(5,4)`, the transpose of the mathematical row-vector convention.
2. **Backward:** gradients are initially `None`, then `loss.backward()` fills each trainable parameter's `.grad` with the same shape as that parameter.
3. **Update:** `optimizer.step()`, rather than `loss.backward()`, changes the weights.
4. **Inference:** `torch.no_grad()` makes outputs have `requires_grad=False` and `grad_fn=None`, while parameters still have `requires_grad=True`. Previously computed gradients remain until cleared. `model.eval()` controls layer behavior; it is **not** a substitute for `no_grad()`.

## Teach-back (before looking at the output)

- Predict all five printed Forward shapes, including the scalar loss.
- Explain why `0.weight` is `(5,4)`, not `(4,5)`.
- Which statement computes gradients? Which statement changes parameters?
- Why does `model[0].weight.grad` still exist inside the inference section?

## Student exercises (not implemented here)

1. Change the batch size from 3 to 8. Predict which tensor dimensions change.
2. Remove `optimizer.zero_grad()` and execute two forward/backward passes on the same model. Compare accumulated gradients with zeroing before each pass.
3. Remove `torch.no_grad()` from the inference section, and inspect `requires_grad` and `grad_fn`. Restore it afterward.

The demo illustrates *whether* autograd tracks operations; it does **not** measure GPU memory usage or expose the exact set of saved activation tensors.
