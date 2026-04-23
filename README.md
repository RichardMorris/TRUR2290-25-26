# TRUR2290-25-26
Code for the Artificial Intelligence and Machine Learning unit at Truro and Penwith College University Centre

##  Neilsen example
The nielsen directory is based on code from [unexploredtest](https://github.com/unexploredtest/neural-networks-and-deep-learning) which in turn is bases
on [Nielsen's neural networks and deep learning](https://github.com/mnielsen/neural-networks-and-deep-learning).

The main example there is example1.py which can be run in an interactive windows in a github codespace.
* mnist.pkl.gz is a gzipped pickle file with the training and test data for the handwritten numbers dataset.
* mnist_loader.py functions to loads the data
* mnist_disp.py functions to display images of the digits
* network.py is a class defining a network, allowing it to be trained by stocastic gradient decent.
* example.py can be modified to change the number of layers, number of epochs and other parameter.

## Running linear regression

You can either clone from github, or use a codelab or in a github codespace. 

It runs in python using the ironpython extension for \jupiter labs.

Things which worked for me:
* Using a virtual environment
* Use a bash shell from visual studio to run the code in the first cell under Windows
* pip install seaborn
  
