import numpy as np


class Layer:
    """Base class and interface for layers of a NeuralNet.

    All layers need `params` and `grads` for the NeuralNet to work.
    Layers may leave them empty, though.
    """

    def __init__(self):
        self.params = {}
        self.grads = {}
        self.prev = self.next = None

    @staticmethod
    def batch(inputs, targets, batch_size=32, shuffle=True):
        """Go through inputs and targets and return (yield, actually) batches
        of given batch size.
        """
        starts = np.arange(0, len(inputs), batch_size)
        if shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end = start + batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield batch_inputs, batch_targets

    def train(self, loss, optimizer, inputs, targets, n_epochs=5000):
        """Train the neural net with the given inputs/targets, using the
        given loss function and optimizer, over n_epochs.
        """
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in self.batch(inputs, targets):
                predicted = self.forward(batch_inputs)
                batch_loss, grad = loss(predicted, batch_targets)
                epoch_loss += batch_loss
                self.backward(grad)
                optimizer(self)
            print(epoch, epoch_loss)

    def _forward(self, inputs):
        raise NotImplementedError

    def _backward(self, grad):
        raise NotImplementedError

    def forward(self, inputs):
        layer = self
        while layer:
            inputs = layer._forward(inputs)
            layer = layer.next
        return inputs

    def backward(self, grad):
        if self.next:
            return self._backward(self.next.backward(grad))
        return self._backward(grad)

    def params_and_grads(self):
        for name, param in self.params.items():
            grad = self.grads[name]
            yield param, grad
        if self.next:
            yield from self.next.params_and_grads()

    def __rshift__(self, other):
        if not isinstance(other, Layer):
            return NotImplemented
        last = self
        while last.next:
            last = last.next
        last.next = other
        other.prev = last
        return self


class Linear(Layer):
    """Linear layer with output = inputs · w + b"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def _forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def _backward(self, grad):
        # Linear back propagation. Quick mafs by Joel:
        #
        # if y = f(x) and x = a * b + c
        # then dy/da = f'(x) * b
        # and dy/db = f'(x) * a
        # and dy/dc = f'(x)
        #
        # if y = f(x) and x = a @ b + c
        # then dy/da = f'(x) @ b.T
        # and dy/db = a.T @ f'(x)
        # and dy/dc = f'(x)
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


class Activation(Layer):
    """Activation layer with output = f(inputs)"""

    def __init__(self, f, f_prime):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def _forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def _backward(self, grad):
        # chain rule:
        #
        # if y = f(x) and x = g(z)
        # then dy/dz = f'(x) * g'(z)
        return self.f_prime(self.inputs) * grad


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    y = tanh(x)
    return 1 - y ** 2
