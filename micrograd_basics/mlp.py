"""A scalar MLP built from Value objects, no tensors or PyTorch."""
import random

from value import Value


class Neuron:
    def __init__(self, inputs, *, activation=True):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.bias = Value(0)
        self.activation = activation

    def __call__(self, x):
        assert len(x) == len(self.weights), "wrong number of inputs"
        z = sum((w * xi for w, xi in zip(self.weights, x)), self.bias)
        return z.relu() if self.activation else z

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, inputs, outputs, *, activation=True):
        self.neurons = [Neuron(inputs, activation=activation) for _ in range(outputs)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, inputs, widths):
        assert widths and all(n > 0 for n in widths)
        sizes = [inputs] + list(widths)
        self.layers = [
            Layer(sizes[i], sizes[i + 1], activation=i < len(widths) - 1)
            for i in range(len(widths))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # always a list, including a single-output model

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
