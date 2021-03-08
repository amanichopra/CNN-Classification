import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import random
from sklearn.datasets import load_iris as iris
from sklearn.model_selection import train_test_split
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Disable some troublesome logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1, num_layers=2, activation='sigmoid'):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.activation = activation
        self.weights = []
        self.num_layers = num_layers
        for layer in range(1, num_layers + 1):
            if layer == 1:
                self.weights.append(np.random.randn(self.inputSize, self.neuronsPerLayer))
            elif layer == num_layers:
                self.weights.append(np.random.randn(self.neuronsPerLayer, self.outputSize))
            else:
                self.weights.append(np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer))
        self.initial_weights = self.weights.copy()

    def __activation(self, x, derivative=False, force_activation=None):
        if force_activation:
            activation = force_activation
        else:
            activation = self.activation
        if activation == 'sigmoid':
            if derivative:
                return np.multiply(x, (1 - x))
            else:
                return 1 / (1 + np.exp(-x))
        elif activation == 'relu':
            if derivative:
                return np.vectorize(lambda x: 1 if x >= 0 else 0)(x)
            else:
                return np.vectorize(lambda x: max(0, x))(x)
        else:
            raise Exception('Invalid Activation!')

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, xVals, yVals, n):
        for i in range(0, len(xVals), n):
            yield xVals[i: i + n], yVals[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=100000, minibatches=True, mbs=100):
        self.xVals = xVals
        self.yVals = yVals
        self.train_errors = np.zeros(epochs)
        self.train_acc = np.zeros(epochs)
        self.epochs = epochs
        if minibatches:
            self.mbs = mbs
        else:
            self.mbs = None

        # TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        for i in range(epochs):
            if minibatches:
                for xVals, yVals in self.__batchGenerator(self.xVals, self.yVals, mbs):
                    layers = self.__forward(xVals)
                    err = np.multiply((layers[-1] - yVals),
                                      self.__activation(layers[-1], derivative=True, force_activation='sigmoid'))
                    delta = np.dot(layers[-2].transpose(), err) / xVals.shape[0]
                    updates = [delta]

                    for j in range(1, len(self.weights)):
                        err = np.multiply(np.dot(err, self.weights[-j].transpose()),
                                          self.__activation(layers[-(j + 1)], derivative=True))
                        try:
                            delta = np.dot(layers[-(j + 2)].transpose(), err)
                        except IndexError:
                            delta = np.dot(xVals.transpose(), err) / xVals.shape[0]
                        updates.append(delta)
                    updates.reverse()
                    for j in range(len(updates)):
                        self.weights[j] -= self.lr * updates[j]

            else:
                layers = self.__forward(self.xVals)
                err = np.multiply((layers[-1] - self.yVals),
                                  self.__activation(layers[-1], derivative=True, force_activation='sigmoid'))
                delta = np.dot(layers[-2].transpose(), err) / self.xVals.shape[0]
                updates = [delta]

                for j in range(1, len(self.weights)):
                    err = np.multiply(np.dot(err, self.weights[-j].transpose()),
                                      self.__activation(layers[-(j + 1)], derivative=True))
                    try:
                        delta = np.dot(layers[-(j + 2)].transpose(), err)
                    except IndexError:
                        delta = np.dot(self.xVals.transpose(), err) / self.xVals.shape[0]
                    updates.append(delta)
                updates.reverse()

                for j in range(len(updates)):
                    self.weights[j] -= self.lr * updates[j]

            self.train_errors[i] = self.__getMSE()
            self.train_acc[i] = self.__getAcc()

    # Forward pass.
    def __forward(self, input):
        outputs = [self.__activation(np.dot(input, self.weights[0]))]
        for i in range(1, len(self.weights)):
            if i == len(self.weights) - 1:
                outputs.append(self.__activation(np.dot(outputs[-1], self.weights[i]), force_activation='sigmoid'))
            else:
                outputs.append(self.__activation(np.dot(outputs[-1], self.weights[i])))
        return outputs

    # Predict.
    def predict(self, xVals):
        return self.__forward(xVals)[-1]

    def __getMSE(self):
        pred = self.predict(self.xVals)
        return np.sum((self.yVals - pred) ** 2) / (2 * pred.size)

    def __getAcc(self):
        out = self.predict(self.xVals)
        preds = np.array([[1 if i == np.argmax(preds) else 0 for i in range(len(preds))] for preds in out])
        acc = 0
        for i in range(preds.shape[0]):
            if list(preds[i]).index(1) == list(self.yVals[i]).index(1):   acc = acc + 1
        acc = acc / float(preds.shape[0])
        return acc

    def getTrainingErrors(self):
        return self.train_errors

    def getTrainingAcc(self):
        return self.train_acc

    def evalMod(self, data):
        xTest, yTest = data
        preds = self.predict(xTest)
        mse = getLoss(yTest, preds)
        preds = np.array([[1 if i == np.argmax(pred) else 0 for i in range(len(pred))] for pred in preds])
        acc = 0
        for i in range(preds.shape[0]):
            if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        accuracy = acc / float(preds.shape[0])
        return accuracy, mse

    def getParams(self, test_data):
        return {'initial_weights': self.initial_weights, 'num_layers': self.num_layers,
                'neurons_per_layer': self.neuronsPerLayer, 'lr': self.lr, 'epochs': self.epochs, 'mbs': self.mbs,
                'train_errors': self.train_errors, "train_acc": self.train_acc,
                'test_error': self.evalMod(test_data)[1], 'test_acc': self.evalMod(test_data)[0]}


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def getLoss(true, preds):
    return np.sum((true - preds) ** 2) / (2 * true.size)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw, alg):
    ((xTrain, yTrain), (xTest, yTest)) = raw

    # flatten 2d array of pixels into 1d array
    if alg == 'guesser' or alg == 'custom_net':
        xTrain = np.array([inp.flatten() for inp in xTrain])
        xTest = np.array([inp.flatten() for inp in xTest])
    else:
        xTrain = np.reshape(xTrain, (xTrain.shape[0], 28, 28, 1))
        xTest = np.reshape(xTest, (xTest.shape[0], 28, 28, 1))

    # scale pixel values between 0-1
    xTrain, xTest = xTrain / 255.0, xTest / 255.0

    # one hot encode labels
    yTrainP = to_categorical(yTrain)
    yTestP = to_categorical(yTest)

    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data, learningRate=0.1, epochs=100000, minibatches=True, mbs=100, inp_size=784,
               out_size=10, layers=[64], activation='sigmoid', ALGORITHM='custom_net'):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        nn = NeuralNetwork_2Layer(inp_size, out_size, layers[0], learningRate=learningRate, num_layers=len(layers) + 1,
                                  activation=activation)
        nn.train(xTrain, yTrain, minibatches=minibatches, epochs=epochs)
        return nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        first_layers = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(0.2), \
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(0.2), \
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(0.2), \
            tf.keras.layers.Flatten()]

        hidden_layers = [tf.keras.layers.Dense(neurons, activation=activation) for neurons in layers] + [
            tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(out_size, activation=tf.nn.softmax)]

        model = tf.keras.models.Sequential(first_layers + hidden_layers)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=epochs)
        print("Model built.")  # TODO: Write code to build and train your keras neural net.
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model, ALGORITHM='custom_net'):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return model.predict(data)  # TODO: Write code to run your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds, ALGORITHM='custom_net', plot=False, out_size=10):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    mse = getLoss(yTest, preds)
    preds = np.array([[1 if i == np.argmax(pred) else 0 for i in range(len(pred))] for pred in preds])
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / float(preds.shape[0])

    conf_mat = confusion_matrix([np.argmax(pred) for pred in yTest], [np.argmax(pred) for pred in preds])
    print("Classifier algorithm: %s" % ALGORITHM)
    if plot:
        conf_mat = pd.DataFrame(conf_mat, index = [i for i in range(out_size)],columns = [i for i in range(out_size)])
        plt.figure(figsize = (10,7))
        sns.heatmap(conf_mat, annot=True)
        plt.title('Confusion Matrix')
    else:
        print('Confusion Matrix:')
        print(conf_mat)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print('Classifier MSE: %f' % mse)
    print()
    return accuracy, mse

def getLabels(preds):
    return np.array([[1 if i == np.argmax(pred) else 0 for i in range(len(pred))] for pred in preds])


def load_iris():
    X = iris().data
    y = to_categorical(iris().target)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    return (x_train, y_train), (x_test, y_test)


#=========================<Main>================================================
def main():
    parser = argparse.ArgumentParser(description='NN')
    # opt for custom_net: epochs=16, mbs=100, lr=5, act='sigmoid', layers=[50]
    # opt for tf_net: epochs=20, act='relu', layers=[256]

    parser.add_argument('--dataset', default='mnist') # dataset to train/eval
    parser.add_argument('--alg', default='custom_net') # algorithm to use
    parser.add_argument('--epochs', type=int, default=16) # number of epochs
    parser.add_argument('--minibatches', type=bool, default=True) # whether to perform mini-batch gradient descent or standard gradient descent
    parser.add_argument('--mbs', type=int, default=100) # minibatch size
    parser.add_argument('--lr', type=float, default=5) # learning rate
    parser.add_argument('--act', default='sigmoid') # activation function to use
    parser.add_argument('--layers', type=list, default=[50]) # list of number of neurons per hidden layer (excluding output)

    args = parser.parse_args()

    if args.dataset == 'mnist':
        raw = getRawData()
        data = preprocessData(raw, args.alg)
    else:
        data = load_iris()

    model = trainModel(data[0], epochs=args.epochs, learningRate=args.lr, layers=args.layers, mbs=args.mbs,
                       inp_size=data[0][0][0].shape[0], out_size=data[0][1][0].shape[0], activation=args.act,
                       ALGORITHM=args.alg)
    preds = runModel(data[1][0], model, ALGORITHM=args.alg)
    evalResults(data[1], preds, ALGORITHM=args.alg)


if __name__ == '__main__':
    main()
