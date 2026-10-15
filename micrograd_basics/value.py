"""Scalar reverse-mode autodiff: one Value is one node of a computation graph."""


class Value:
    def __init__(self, data, parents=()):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(parents)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = backward
        return result

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        result = Value(self.data ** exponent, (self,))

        def backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * result.grad

        result._backward = backward
        return result

    def relu(self):
        result = Value(max(0.0, self.data), (self,))

        def backward():
            self.grad += (self.data > 0) * result.grad

        result._backward = backward
        return result

    def backward(self):
        order, visited = [], set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        # Each backward call computes a fresh graph pass.
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
