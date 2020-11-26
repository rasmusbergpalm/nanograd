import torch as t
import random
from nanograd import Var


def test_sanity_check():
    x = Var(-4.0)
    z = Var(2.0) * x + Var(2.0) + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = t.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.v == ypt.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()


def test_more_ops():
    a = Var(-4.0)
    b = Var(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + Var(1.0)
    c += Var(1.0) + c + (-a)
    d += d * Var(2.0) + (b + a).relu()
    d += Var(3.0) * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / Var(2.0)
    g += Var(10.0) / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = t.Tensor([-4.0]).double()
    b = t.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.v - gpt.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_mlp():
    n_nodes = 4

    inputs = [Var(random.random() - 0.5) for _ in range(n_nodes)]
    W1 = [[Var(random.random() - 0.5) for _ in range(n_nodes)] for _ in range(n_nodes)]
    W2 = [[Var(random.random() - 0.5) for _ in range(n_nodes)] for _ in range(n_nodes)]

    def matmul(prev, weights):
        layer = []
        for j in range(n_nodes):
            node = Var(0.0)
            for i in range(n_nodes):
                node += weights[i][j] * prev[i]
            layer.append(node)
        return layer

    h1 = matmul(inputs, W1)
    h2 = matmul(h1, W2)
    h2[0].backward()

    vars = [inputs, h1, h2, W1, W2]

    inputs = t.tensor([i.v for i in inputs], requires_grad=True, dtype=t.double)
    W1 = t.tensor([[v.v for v in row] for row in W1], requires_grad=True, dtype=t.double)
    W2 = t.tensor([[v.v for v in row] for row in W2], requires_grad=True, dtype=t.double)

    h1 = t.matmul(inputs, W1)
    h1.retain_grad()
    h2 = t.matmul(h1, W2)
    h2.retain_grad()
    h2[0].backward()

    tol = 1e-6

    def assert_vector(v, tv):
        assert all([abs(_v.v - _tv.item()) < tol for _v, _tv in zip(v, tv)])
        assert all([abs(_v.grad - _tv.item()) < tol for _v, _tv in zip(v, tv.grad)])

    def assert_weights(w, tw):
        for i in range(n_nodes):
            for j in range(n_nodes):
                assert abs(w[i][j].v - tw[i, j].item()) < tol
                assert abs(w[i][j].grad - tw.grad[i, j].item()) < tol

    assert_vector(vars[0], inputs)
    assert_vector(vars[1], h1)
    assert_vector(vars[2], h2)
    assert_weights(vars[3], W1)
    assert_weights(vars[4], W2)


def test_non_leaf_reuse():
    a = Var(3.0)
    b = Var(5.0)
    c = Var(9.0)
    d = b * c
    e = a + d
    f = e * d
    f.backward()

    vars = [a, b, c, d, e, f]

    a = t.scalar_tensor(3.0, requires_grad=True)
    b = t.scalar_tensor(5.0, requires_grad=True)
    c = t.scalar_tensor(9.0, requires_grad=True)
    d = b * c
    e = a + d
    f = e * d
    t_vars = [a, b, c, d, e, f]
    [t_var.retain_grad() for t_var in t_vars]
    f.backward()

    assert [v.v == tv.item() and v.grad == tv.grad.item() for v, tv in zip(vars, t_vars)]
