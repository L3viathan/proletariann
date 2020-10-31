import numpy as np


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


def make_model(*net):
    net = list(net)

    def train(loss, optimizer, inputs, targets, n_epochs=5000):
        """Train the neural net with the given inputs/targets, using the
        given loss function and optimizer, over n_epochs.
        """

        def predict(data):
            for forward, _, __ in net:
                data = forward(data)
            return data

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in batch(inputs, targets):
                batch_predicted = predict(batch_inputs)
                batch_loss, grad = loss(batch_predicted, batch_targets)
                epoch_loss += batch_loss
                for _, __, backward in reversed(net):
                    grad = backward(grad)
                for _, params_and_grads, __ in net:
                    optimizer(params_and_grads())
            print(epoch, epoch_loss)
        return predict

    return train


def params_and_grads(data):
    def params_and_grads():
        for name, param in data["params"].items():
            grad = data["grads"][name]
            yield param, grad

    return params_and_grads


def linear(input_size, output_size):
    data = {
        "params": {
            "w": np.random.randn(input_size, output_size),
            "b": np.random.randn(output_size),
        },
        "grads": {},
    }

    def forward(inputs):
        data["inputs"] = inputs
        return inputs @ data["params"]["w"] + data["params"]["b"]

    def backward(grad):
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
        data["grads"]["b"] = np.sum(grad, axis=0)
        data["grads"]["w"] = data["inputs"].T @ grad
        return grad @ data["params"]["w"].T

    return forward, params_and_grads(data), backward


def activation(f, f_prime):
    data = {
        "params": {},
        "grads": {},
    }

    def forward(inputs):
        data["inputs"] = inputs
        return f(inputs)

    def backward(grad):
        # chain rule:
        #
        # if y = f(x) and x = g(z)
        # then dy/dz = f'(x) * g'(z)
        return f_prime(data["inputs"]) * grad

    return forward, params_and_grads(data), backward


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    y = tanh(x)
    return 1 - y ** 2


def tse(predicted, actual):
    """Total Squared Error loss function, returns error and gradient."""
    return (
        np.sum((predicted - actual) ** 2),
        2 * (predicted - actual),
    )


# TODO: implement Mean Squared Error mse(predicted, actual)


def sgd(lr=0.01):
    """Stochastic Gradient Descent optimizer, with given learning rate."""

    def _step(params_and_grads):
        for param, grad in params_and_grads:
            param -= lr * grad

    return _step
