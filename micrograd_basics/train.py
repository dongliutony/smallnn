"""Train a 2 -> 8 -> 1 scalar MLP on four XOR examples.

Run from this directory: python train.py
"""
import random

from mlp import MLP

random.seed(7)
model = MLP(2, [8, 1])
examples = [([0., 0.], 0.), ([0., 1.], 1.),
            ([1., 0.], 1.), ([1., 1.], 0.)]

for step in range(301):
    predictions = [model(x)[0] for x, _ in examples]
    loss = sum((prediction - y) ** 2
               for prediction, (_, y) in zip(predictions, examples)) * (1.0 / len(examples))

    # Zero all parameter gradients before each backpropagation.
    for parameter in model.parameters():
        parameter.grad = 0.0
    loss.backward()
    for parameter in model.parameters():
        parameter.data -= 0.05 * parameter.grad

    if step % 50 == 0:
        print(f"step {step:3d}: mse = {loss.data:.5f}")

for x, y in examples:
    print(f"input={x}, target={y}, prediction={model(x)[0].data:.3f}")
