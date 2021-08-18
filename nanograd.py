from typing import Union
from math import tanh


class Var:
    """
    A variable which holds a number and enables gradient computations.
    """

    def __init__(self, val: Union[float, int], parents=None):
        assert type(val) in {float, int}
        if parents is None:
            parents = []
        self.v = val
        self.parents = parents
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for parent, grad in self.parents:
            parent.backprop(grad * bp)

    def backward(self):
        self.backprop(1.0)

    def __add__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v + other.v, [(self, 1.0), (other, 1.0)])

    def __mul__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v * other.v, [(self, other.v), (other, self.v)])

    def __pow__(self, power: Union[float, int]) -> 'Var':
        assert type(power) in {float, int}, "power must be float or int"
        return Var(self.v ** power, [(self, power * self.v ** (power - 1))])

    def __neg__(self: 'Var') -> 'Var':
        return Var(-1.0) * self

    def __sub__(self: 'Var', other: 'Var') -> 'Var':
        return self + (-other)

    def __truediv__(self: 'Var', other: 'Var') -> 'Var':
        return self * other ** -1

    def tanh(self) -> 'Var':
        return Var(tanh(self.v), [(self, 1 - tanh(self.v) ** 2)])

    def relu(self) -> 'Var':
        return Var(self.v if self.v > 0.0 else 0.0, [(self, 1.0 if self.v > 0.0 else 0.0)])

    def __repr__(self):
        return "Var(v=%.4f, grad=%.4f)" % (self.v, self.grad)
