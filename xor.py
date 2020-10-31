#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from proletariann import linear, activation, tanh, tanh_prime, sgd, tse, make_model


if __name__ == "__main__":
    inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    # XOR can't be learned with a simple linear model. Comment out the
    # Tanh and second Linear layer to see for yourself.
    train = make_model(
        linear(input_size=2, output_size=2),
        activation(tanh, tanh_prime),
        linear(input_size=2, output_size=2),
    )
    predict = train(
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
