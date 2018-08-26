# TensorFlow introduction
An introduction to artificial neural network with an TensorFlow example.

# Usage
First install Anaconda or miniconda.\
Then install TensorFlow as below:
```cpp
conda create -n tensorflow pip python=3.6
activate tensorflow
conda install -c anaconda tensorflow scipy h5py
```

Run the example:
```cpp
git clone https://github.com/gongfan99/tensorflow_introduction.git
python example.py
```
With epoch=2, we get the accuracy of 98.6% on the test data.

# Introduction
This is a summary of some key concepts of artificial neural network(ANN) which is demonstrated with a TensorFlow example. For how to build an ANN step by step, please check the [reference](https://github.com/gongfan99/tensorflow_introduction#reference).

#### Artificial neural network
An artificial neural network consists of several layers of neurons. Each neuron connects to other neurons with the weight and bias parameters.

#### Neuron
A neuron is the smallest building block of an ANN with three key features:
1. state\
This is the input to a neuron. Generally it is a linear sum of the outputs of other neurons.

2. activation function\
This is the function that converts the input (i.e. state) of the neuron to the output (i.e. activation). ReLU is a popular and effective choice.

3. activation\
This is the output of the neuron.

#### Multilayer perceptron (MLP)
MLP is the earliest and most potent architecture of ANN. The neurons of a layer fully connects all the neurons of the next layer.
```python
x = keras.layers.Dense(units=128, kernal_initializer='he_uniform', kernal_regularizer=keras.regularizers.l2(l=0.0001), activation='relu')(x)
```

#### Convolutional layer
MLP, being very potent due to the full connection between adjacent layers, has large number of weight/bias parameters if a layer has many neurons. For example, if the input to an ANN is a 28 x 28 image, the input layer will have 784 neurons. To decrease the number of the weight/bias parameters and exploit the spatial information in an image, the convolutional layer can be used. Each convolutional layer has several kernels each of which convolutes with the input image to create an output image. Then the output images go through the activation functions to produce the outputs of the layer.
```python
x = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', kernal_initializer='he_uniform', kernal_regularizer=keras.regularizers.l2(l=0.0001), activation='relu')(inpBatchNorm)
```

#### Pooling layer
The pooling layer consumes small and usually disjoint chunks of the image (typically 2×2) and aggregates them into a single value, thus downsampling the input image. Max-pooling, the most popular approach, simply takes the maximum pixel value within each chunk.
```python
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
```

#### Regularization
With large number of weight/bias parameters, an ANN may overfit the potentially noisy training samples, causing decreased accuracy when evaluated with the test samples. Regularization is a technique to avoid overfitting by putting some constraints in the ANN training.\
An easy regularization method is dropout which randomly eliminates the neurons in a layer to prevent any neuron from having overly dominant importance to the entire ANN.
```python
x = keras.layers.Dropout(rate=0.25)(x)
```
Another regularization method is L2 regularization. It introduces a penalization cost if a weight has large magnitude.

#### Batch and normalization
Generally an ANN is trained with many batches of training samples. To speedup the training, the batch normalization can be employed to normalize the activations of a layer to zero mean and unit variance across a batch.
```python
x = keras.layers.BatchNormalization()(x)
```

#### Data augmentation
It is to artificially augment the data with distorted versions during training.
```python
datagen = keras.ppreprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train[:54000])
```

#### Ensembles
The performance of a trained ANN largely depends on the initial conditions. Different initial condition can lead to the different ANN with strength on certain test samples. Therefore, by averaging the outputs from various ANNs trained with different initial conditions, an ANN with the balanced strength can be obtained.
```python
out = keras.layers.average(outs)
```

#### Early stopping
To save time, we can simply stop training once the validation loss has not decreased for a fixed number of epochs (a parameter known as patience).

# Reference
[Deep learning for complete beginners: recognising handwritten digits](https://cambridgespark.com/content/tutorials/deep-learning-for-complete-beginners-recognising-handwritten-digits/index.html)\
[Deep learning for complete beginners: convolutional neural networks with keras](https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html)\
[Deep learning for complete beginners: neural network fine-tuning techniques](https://cambridgespark.com/content/tutorials/neural-networks-tuning-techniques/index.html)

# License
MIT