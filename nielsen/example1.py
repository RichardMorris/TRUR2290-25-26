"""example1.py
~~~~~~~~~~~~~~~~~~~~~~~

Use network2 to figure out the average starting values of the gradient
error terms \\delta^l_j = \\partial C / \\partial z^l_j = \\partial C /
\\partial b^l_j.

"""

#### Libraries
# Standard library
import json
import math
import random
import shutil
import sys
import functools
sys.path.append("../src/")

# My library
import mnist_loader
import mnist_disp
import network

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import itertools

def main():
    # Load the data
    full_td, _, testdata = mnist_loader.load_data_wrapper()
    trd = list(full_td) 
    tsd = list(testdata)
    epochs = 30 # Number of epochs to train for
    layers = [784, 16, 16, 10]
    print("\n{0} hidden layers: {1}".format(len(layers)-2,layers))
    net = network.Network(layers)

    net.SGD(trd, epochs, 10, 3.0, test_data=tsd)

    training_set, validation_set, test_set = mnist_loader.load_data()
    images = mnist_disp.get_images(test_set)


    correct = 0
    for count in range(0,10):
        index = random.randrange(len(images))
        (img,expected) = tsd[index]
        mnist_disp.plot_mnist_digit(images[index])

        actual = net.feedforward(img)
        ac = [x[0] for x in actual]
        acam = np.argmax(ac)
        exam= int(expected)
        if acam == exam:
            correct+=1
        print(exam,acam,['{:4.2f}'.format(x[0]) for x in actual])
    print(correct,count)
    print(net.evaluate(tsd))
#    plot_training(
#        epochs, "norms_during_training_2_layers.json", 2)



def zip_sum(a, b): 
    return [x+y for (x, y) in zip(a, b)]

def list_sum(l):
    return functools.reduce(zip_sum, l)

def list_norm(l):
    return math.sqrt(sum([x*x for x in l]))

if __name__ == "__main__":
    main()
