# Stage 1: scalar MLP from scratch

This is a small, independently written teaching example inspired by Andrej Karpathy's [micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) and [source code](https://github.com/karpathy/micrograd) (MIT). It does **not** require PyTorch or NumPy.

Read and run in order:

1. `value.py` — scalar computation graph and reverse-mode autodiff.
2. `mlp.py` — neurons, layers, and a small MLP.
3. `train.py` — mean squared error and SGD on XOR.
4. `test_mlp.py` — shared-node gradient, finite-difference check, MLP smoke test.

From the repo root:

```sh
python micrograd_basics/train.py
python -m unittest discover -s micrograd_basics -p 'test_*.py'
```

Student teach-back: Why use `+=` when propagating gradients? Why must parameters be zeroed before each step? What is the number of scalar parameters for `MLP(2, [8, 1])`? Why does a hidden ReLU layer make XOR learnable, while a single linear layer cannot?

This new example lives alongside the older `micrograd/` implementation without modifying it. Compare the designs after understanding each one. The next step is a mentor review, **not** attention or GPT.
