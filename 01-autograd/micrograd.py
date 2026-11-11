import math


def do_nothing():
    pass


class Value:
    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = do_nothing
        self._prev = _prev
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value({self.label or 'data'}={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # 数字先包成 Value
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "只支持常数次方"
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):  # -a
        return self * -1

    def __sub__(self, other):  # a - b
        return self + (-other)

    def __rsub__(self, other):  # 数字 - a
        return other + (-self)

    __radd__ = __add__  # 数字 + a
    __rmul__ = __mul__  # 数字 * a

    def backward(self):  # 就是 4.8 的 backward，搬进类里
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
