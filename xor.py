#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from simplenn import linear, activation, tanh, tanh_prime, sgd, tse, train


if __name__ == "__main__":
    inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    # XOR can't be learned with a simple linear model. Comment out the
    # Tanh and second Linear layer to see for yourself.
    net = [
        linear(input_size=2, output_size=2),
        activation(tanh, tanh_prime),
        linear(input_size=2, output_size=2),
    ]
    predict = train(
        net,
        loss=tse,
        optimizer=sgd(),
        inputs=inputs,
        targets=targets,
        n_epochs=5000,
    )
    print("x, predicted, y")
    for x, y in zip(inputs, targets):
        predicted = predict(x)
        print(x, predicted, y)
