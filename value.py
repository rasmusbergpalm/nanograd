from typing import Union


class Value:
    def __init__(self, val: float, grad_fn=lambda: []):
        assert type(val) == float
        self.v = val
        self.grad_fn = grad_fn
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for input, grad in self.grad_fn():
            input.backprop(grad * bp)

    def backward(self):
        self.backprop(1.0)

    def __add__(self: 'Value', other: 'Value') -> 'Value':
        return Value(self.v + other.v, lambda: [(self, 1.0), (other, 1.0)])

    def __mul__(self: 'Value', other: 'Value') -> 'Value':
        return Value(self.v * other.v, lambda: [(self, other.v), (other, self.v)])

    def __pow__(self, power: Union[float, int]):
        assert type(power) in {float, int}, "power must be float or int"
        return Value(self.v ** power, lambda: [(self, power * self.v ** (power - 1))])

    def __neg__(self: 'Value') -> 'Value':
        return Value(-1.0) * self

    def __sub__(self: 'Value', other: 'Value') -> 'Value':
        return self + (-other)

    def __truediv__(self: 'Value', other: 'Value') -> 'Value':
        return self * other ** -1

    def relu(self):
        return Value(self.v if self.v > 0.0 else 0.0, lambda: [(self, 1.0 if self.v > 0.0 else 0.0)])
