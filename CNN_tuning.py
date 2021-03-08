import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from ANN import trainModel as trainCustomNet
import argparse


random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =========================<Classifier Functions>================================

def guesserClassifier(xTest):
    NUM_CLASSES = xTest.shape[1]
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, DATASET, eps=5, net_type='keras', hidden_layers=[512], activation='sigmoid', load=True):
    if net_type == 'custom':
        x = np.array([rec.flatten() for rec in x])  # ensure input is flattened when using custom net
        mod = trainCustomNet((x, y), epochs=eps, inp_size=x.shape[1], out_size=y.shape[1], activation=activation)
    elif net_type == 'keras':
        if load:
            return keras.models.load_model(DATASET + '_ANN')
        layers = [tf.keras.layers.Dense(neurons, activation=activation) for neurons in hidden_layers] + [
            tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax)]
        mod = tf.keras.models.Sequential(layers)
        mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mod.fit(x, y, epochs=eps)

    else:
        raise ValueError("Invalid net type! Options are 'custom' and 'keras'.")
    return mod


def buildTFConvNet(x, y, DATASET, eps=20, dropRate=0.2, hidden_layers=[256], activation='relu', load=True, rand_crop=False):
    if load:
        return keras.models.load_model(DATASET)
    out_size = y.shape[1]
    if DATASET == 'mnist_d':
        first_layers = []
        if rand_crop:
            first_layers.append(
                keras.layers.experimental.preprocessing.RandomCrop(x[0].shape[0] - 4, x[0].shape[1] - 4))

        first_layers = first_layers + [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=x[0].shape), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Flatten()]
        hid_layers = [tf.keras.layers.Dense(neurons, activation=activation) for neurons in hidden_layers] + [
            tf.keras.layers.Dropout(dropRate), tf.keras.layers.Dense(out_size, activation=tf.nn.softmax)]

        mod = tf.keras.models.Sequential(first_layers + hid_layers)
        mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mod.fit(x, y, epochs=eps)
    elif DATASET == 'mnist_f':
        first_layers = []
        if rand_crop:
            first_layers.append(
                keras.layers.experimental.preprocessing.RandomCrop(x[0].shape[0] - 4, x[0].shape[1] - 4))

        first_layers = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=x[0].shape), \
            tf.keras.layers.BatchNormalization(), \
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=x[0].shape), \
            tf.keras.layers.BatchNormalization(), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.BatchNormalization(), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Flatten()]
        hid_layers = []
        for neurons in hidden_layers:
            hid_layers.extend(
                [tf.keras.layers.Dense(neurons, activation=activation), tf.keras.layers.BatchNormalization(),
                 tf.keras.layers.Dropout(dropRate)])
        hid_layers.append(tf.keras.layers.Dense(out_size, activation=tf.nn.softmax))
        mod = tf.keras.models.Sequential(first_layers + hid_layers)
        mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mod.fit(x, y, epochs=eps)
    elif DATASET == 'cifar_100_f':
        mod = tf.keras.models.Sequential()
        if rand_crop:
            mod.add(keras.layers.experimental.preprocessing.RandomCrop(x[0].shape[0] - 4, x[0].shape[1] - 4))

        mod.add(tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), activation="relu", input_shape=x[0].shape))
        mod.add(tf.keras.layers.BatchNormalization())
        tf.keras.layers.Dropout(0.2)
        for i in range(2):
            mod.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation="relu"))
            mod.add(tf.keras.layers.BatchNormalization())
            mod.add(tf.keras.layers.Dropout(0.2))
        mod.add(tf.keras.layers.Flatten())
        for i in range(2):
            mod.add(tf.keras.layers.Dense(160, activation="relu"))
            mod.add(tf.keras.layers.BatchNormalization())
            mod.add(tf.keras.layers.Dropout(0.05))
        mod.add(tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax))
        mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mod.fit(x, y, epochs=eps)
    else:  # cifar_100_c, cifar_10 datasets
        first_layers = []
        if rand_crop:
            first_layers.append(
                keras.layers.experimental.preprocessing.RandomCrop(x[0].shape[0] - 4, x[0].shape[1] - 4))

        first_layers = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=x[0].shape), \
            tf.keras.layers.BatchNormalization(), \
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=x[0].shape), \
            tf.keras.layers.BatchNormalization(), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"), \
            tf.keras.layers.BatchNormalization(), \
            tf.keras.layers.Dropout(dropRate), \
            tf.keras.layers.Flatten()]
        hid_layers = []
        for neurons in hidden_layers:
            hid_layers.extend(
                [tf.keras.layers.Dense(neurons, activation=activation), tf.keras.layers.BatchNormalization(),
                 tf.keras.layers.Dropout(dropRate)])
        hid_layers.append(tf.keras.layers.Dense(out_size, activation=tf.nn.softmax))
        mod = tf.keras.models.Sequential(first_layers + hid_layers)
        mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mod.fit(x, y, epochs=eps)
    return mod


# =========================<Pipeline Functions>==================================

def getRawData(DATASET):
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar_10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar_10.load_data()
    elif DATASET == "cifar_100_f":
        cifar_100_f = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100_f.load_data()
    elif DATASET == "cifar_100_c":
        cifar_100_c = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100_c.load_data(label_mode='coarse')
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw, DATASET, ALGORITHM):
    if DATASET == "mnist_d":
        NUM_CLASSES = 10
        IH = 28
        IW = 28
        IZ = 1
    elif DATASET == "mnist_f":
        NUM_CLASSES = 10
        IH = 28
        IW = 28
        IZ = 1
    elif DATASET == "cifar_10":
        NUM_CLASSES = 10
        IH = 32
        IW = 32
        IZ = 3
    elif DATASET == "cifar_100_f":
        NUM_CLASSES = 100
        IH = 32
        IW = 32
        IZ = 3
    elif DATASET == "cifar_100_c":
        NUM_CLASSES = 20
        IH = 32
        IW = 32
        IZ = 3
    IS = IH * IW * IZ

    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":  # flatten data
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    xTrainP = xTrainP / 255.0
    xTestP = xTestP / 255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))


def trainModel(data, ALGORITHM, DATASET, eps=20, dropRate=0.2, hidden_layers=[256], activation='relu', net_type='keras', load=True,
               rand_crop=False):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain, DATASET, eps=eps, hidden_layers=hidden_layers, activation=activation,
                                net_type=net_type, load=load)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain, DATASET, eps=eps, dropRate=dropRate, hidden_layers=hidden_layers,
                              activation=activation, load=load, rand_crop=rand_crop)
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model, ALGORITHM):
    NUM_CLASSES = data[1].shape[1]
    data = data[0]

    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds, ALGORITHM):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = float(acc) / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return accuracy

#=========================<Main>================================================

def main():
    parser = argparse.ArgumentParser(description='NN')

    # python CNN_tuning.py --dataset mnist_d --alg tf_conv --epochs=1 --load True, --randCrop True

    parser.add_argument('--dataset', default='mnist_d')  # dataset to train/eval
    parser.add_argument('--alg', default='tf_conv')  # algorithm to use
    parser.add_argument('--netType', default='keras') # if using custom net, net types are keras and custom
    parser.add_argument('--epochs', type=int, default=20)  # number of epochs
    parser.add_argument('--act', default='relu')  # activation function to use
    parser.add_argument('--dropRate', type=float, default=0.2)  # drop rate to use in dropout layers for regularization
    parser.add_argument('--hiddenLayers', type=list,
                        default=[256])  # list of number of neurons per hidden layer (excluding output)
    parser.add_argument('--load', default='True')  # whether or not to load saved weights, or retrain model
    parser.add_argument('--randCrop', default='False') # whether or not to randomly crop images for data augmentation

    args = parser.parse_args()

    raw = getRawData(args.dataset)
    data = preprocessData(raw, args.dataset, args.alg)

    model = trainModel(data[0], args.alg, args.dataset, eps=args.epochs, hidden_layers=args.hiddenLayers, activation=args.act, dropRate=args.dropRate,
                       net_type=args.netType, load=strToBool(args.load), rand_crop=strToBool(args.randCrop))
    preds = runModel(data[1], model, ALGORITHM=args.alg)
    evalResults(data[1], preds, ALGORITHM=args.alg)


def strToBool(str):
    if str == 'True': return True
    else: return False


def buildTuneModel(hp):
    mod = tf.keras.models.Sequential()
    mod.add(tf.keras.layers.Conv2D(filters=hp.Int('input_num_filters', min_value=32, max_value=256, step=32),       kernel_size=(3, 3), activation="relu", input_shape=data[0][0][0].shape))
    mod.add(tf.keras.layers.BatchNormalization())
    tf.keras.layers.Dropout(hp.Float('input_dropout',min_value=0.0,max_value=0.5,default=0.25,step=0.05))
    for i in range(hp.Int('num_conv_layers', 1, 5)):
        mod.add(tf.keras.layers.Conv2D(filters=hp.Int('conv_{i}_num_filters', min_value=32, max_value=256, step=32), kernel_size=(3, 3), activation="relu"))
        mod.add(tf.keras.layers.BatchNormalization())
        mod.add(tf.keras.layers.Dropout(hp.Float('conv_{i}_dropout',min_value=0.0,max_value=0.5,default=0.25,step=0.05)))
    mod.add(tf.keras.layers.Flatten())
    for i in range(hp.Int('num_hid_layers', 1, 5)):
        mod.add(tf.keras.layers.Dense(hp.Int('hid_{i}_num_neurons', min_value=32, max_value=256, step=32), activation="relu"))
        mod.add(tf.keras.layers.BatchNormalization())
        mod.add(tf.keras.layers.Dropout(hp.Float('hid_{i}_dropout',min_value=0.0,max_value=0.5,default=0.25,step=0.05)))
    mod.add(tf.keras.layers.Dense(data[0][1].shape[1], activation=tf.nn.softmax))
    mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return mod


if __name__ == '__main__':
    main()
