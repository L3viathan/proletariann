#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from proletariann import linear, activation, tanh, tanh_prime, sgd, tse, train


def fizz_buzz_encode(x):
    """four classes number, "fizz", "buzz", "fizzbuzz"."""
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x):
    """10 digit binary encoding of x."""
    return [x >> i & 1 for i in range(10)]


if __name__ == "__main__":
    inputs = np.array([binary_encode(x) for x in range(101, 1024)])
    targets = np.array([fizz_buzz_encode(x) for x in range(101, 1024)])
    net = [linear(input_size=10, output_size=50), activation(tanh, tanh_prime), linear(input_size=50, output_size=4)]
    predict = train(
        net,
        loss=tse,
        optimizer=sgd(lr=0.001),
        inputs=inputs,
        targets=targets,
        n_epochs=5000,
    )
    print("here's what's wrong: x, predicted, actual:")
    for x in range(1, 101):
        predicted = predict(binary_encode(x))
        predicted_idx = np.argmax(predicted)
        actual_idx = np.argmax(fizz_buzz_encode(x))
        labels = [str(x), "fizz", "buzz", "fizzbuzz"]
        if predicted_idx != actual_idx:
            print(x, labels[predicted_idx], labels[actual_idx])
