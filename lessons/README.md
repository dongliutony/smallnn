# Small neural network lessons

Compact, runnable Python exercises inspired by Andrej Karpathy's **Neural Networks: Zero to Hero** teaching approach. These files are independently written teaching adaptations, not verbatim notebook exports.

## Order

1. Existing `../micrograd/engine.py` + `../micrograd/nn.py`: scalar automatic differentiation and MLP internals.
2. `mlp.py`: PyTorch tensors, MLP, loss, gradients, optimizer on XOR.
3. `gpt_model.py`: causal self-attention, transformer blocks, character-level Mini GPT.
4. `gpt_train.py`: next-character training and autoregressive sampling.
5. `test_models.py`: tensor-shape, gradient, and causal-mask checks.

## Run

From the repo root (Python 3.10+ and PyTorch required):

```bash
python -m pip install torch
python lessons/mlp.py
python lessons/gpt_train.py
python -m unittest discover -s lessons -p 'test_*.py'
```

Both examples run on CPU and use no downloaded dataset. The toy corpus intentionally produces limited-quality samples; the goal is understanding each operation.

## Source lectures

- [Karpathy: The spelled-out intro to neural networks and backpropagation (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Karpathy: Building makemore, part 1](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Karpathy: Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Original makemore repository](https://github.com/karpathy/makemore)
- [Original nanoGPT repository](https://github.com/karpathy/nanoGPT)

## Mentor checkpoints

- Explain every tensor shape in the forward pass: `[B,T,C]`, `[B,H,T,D]`, `[B,T,V]`.
- Explain why `optimizer.zero_grad()` must precede backward each iteration.
- Prove causality by changing a future input token and checking earlier outputs.
- Explain why training processes sequences in parallel but generation appends one token per step.
- Change one hyperparameter; write a prediction before running the experiment.

The separate [build-nanogpt](https://github.com/dongliutony/build-nanogpt) repository remains the advanced GPT-2 reproduction reference.
