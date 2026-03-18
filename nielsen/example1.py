"""example1.py
~~~~~~~~~~~~~~~~~~~~~~~

Use network2 to figure out the average starting values of the gradient
error terms \\delta^l_j = \\partial C / \\partial z^l_j = \\partial C /
\\partial b^l_j.

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

    # For a reduceed size of trainingset use 
    #  net.SGD(trd[0:10000], epochs, batch_size, eta, test_data=tsd)
    net.SGD(trd, epochs, batch_size, eta, test_data=tsd)

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
    print("correct {0}/{1}".format(correct,count+1))
    print(net.evaluate(tsd))

if __name__ == "__main__":
    main()
