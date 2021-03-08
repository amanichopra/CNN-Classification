# CNN Image Classification

Constructed CNN with Keras for MNIST digits (99% accuracy), MNIST fashion (92% accuracy), CIFAR-10 (73% accuracy), CIFAR-100 fine (43% accuracy), and CIFAR-100 coarse (56% accuracy) datasets. Implemented support for saving and loading models to avoid retraining. Incorporated functionality for augmentation and random cropping of input data.

## Usage
Run CNN_tuning.py on CL with the following arguments:

--dataset: 'mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_c', 'cifar_100_f' (default 'mnist_d')<br/>
--alg: 'guesser', 'tf_net', or 'tf_conv' (default 'tf_conv')<br/> 
--netType: 'keras' or 'custom' (default 'keras')<br/>
--epochs: int (default 20)<br/>
--act: 'sigmoid' or 'relu' (default 'relu')<br/>
--dropRate: float (default 0.2)<br/>
--hiddenLayers: list (default [256])
--load: bool (default True)
--randCrop: bool (default False)
