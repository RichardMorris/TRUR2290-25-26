"""example2.py
~~~~~~~~~~~~~~~~~~~~~~~

Shows the weight matrices
"""

#### Libraries
# Standard library
import random

# My library
import mnist_loader
import mnist_disp
import network

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the data
    full_td, _, testdata = mnist_loader.load_data_wrapper()
    trd = list(full_td) 
    tsd = list(testdata)

    layers = [784, 16, 16, 10]
    epochs = 10 # Number of epochs to train for
    batch_size = 10
    eta = 3.0

    print("\n{0} hidden layers: {1}".format(len(layers)-2,layers))
    print("epochs",epochs)
    print("batch size", batch_size)
    print("eta",eta)
    net = network.Network(layers)
    weights = [int(np.prod(np.shape(x))) for x in net.weights]
    print("weights",["{0}x{1}".format(np.shape(x)[0],np.shape(x)[1]) for x in net.weights]," sum ",sum(weights))
    bias = [len(x) for x in net.biases]
    print("biases", bias," sum ",sum(bias))
    print("total parameters ", sum(weights) + sum(bias))

    net.SGD(trd, epochs, batch_size, eta, test_data=tsd)

    training_set, validation_set, test_set = mnist_loader.load_data()
    images = mnist_disp.get_images(test_set)

    for i in range(0,layers[1]):
        wt0 = net.weights[0][i]
        d0 = np.reshape(wt0,[28,28])
        mnist_disp.plot_mnist_digit(d0)

    for i in range(1,len(weights)):
        mnist_disp.plot_mnist_digit(net.weights[i])


if __name__ == "__main__":
    main()
