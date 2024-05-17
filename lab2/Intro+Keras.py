
# coding: utf-8

# # Training Neural Networks with Keras
# 
# ### Goals: 
# - Intro: train a neural network with high level framework `Keras`
# 
# ### Dataset:
# - Digits: 10 class handwritten digits
# - http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

# In[ ]:


# display figures in the notebook
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()

sample_index = 45
plt.figure(figsize=(3, 3))
plt.imshow(digits.images[sample_index], cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.title("image label: %d" % digits.target[sample_index]);


# ### Preprocessing
# 
# - normalization
# - train/test split

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = np.asarray(digits.data, dtype='float32')
target = np.asarray(digits.target, dtype='int32')

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.15, random_state=37)

# mean = 0 ; standard deviation = 1.0
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(scaler.mean_)
# print(scaler.scale_)


# Let's display the one of the transformed sample (after feature standardization):

sample_index = 45
plt.figure(figsize=(3, 3))
plt.imshow(X_train[sample_index].reshape(8, 8),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("transformed sample\n(standardization)")
plt.show()

# The scaler objects makes it possible to recover the original sample:

plt.figure(figsize=(3, 3))
plt.imshow(scaler.inverse_transform(X_train[sample_index]).reshape(8, 8),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("original sample");
plt.show()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # I) Feed Forward NN with Keras
# 
# Objectives of this section:
# 
# - Build and train a first feedforward network using `Keras`
#     - https://keras.io/getting-started/sequential-model-guide/
# - Experiment with different optimizers, activations, size of layers, initializations
# 
# ### a) Keras Workflow

# To build a first neural network we need to turn the target variable into a vector "one-hot-encoding" representation.
# Here are the labels of the first samples in the training set encoded as integers:

print(y_train[:3])


# Keras provides a utility function to convert integer-encoded categorical variables as one-hot encoded values:

import keras
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train)
print(Y_train[:3])


# We can now build an train a our first feed forward neural network using the high level API from keras:
# 
# - first we define the model by stacking layers with the right dimensions
# - then we define a loss function and plug the SGD optimizer
# - then we feed the model the training data for fixed number of epochs

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers

N = X_train.shape[1]
H = 100
K = 10

model = Sequential()
model.add(Dense(H, input_dim=N))
model.add(Activation("tanh"))
model.add(Dense(K))
model.add(Activation("softmax"))

model.compile(optimizer=optimizers.SGD(lr=0.1),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=15, batch_size=32)


# ### b) Exercises: Impact of the Optimizer
# 
# - Try to decrease the learning rate value by 10 or 100. What do you observe?
# 
# - Try to increase the learning rate value to make the optimization diverge.
# 
# - Configure the SGD optimizer to enable a Nesterov momentum of 0.9



# ### c) Exercises: forward pass and generalization
# 
# - Compute predictions on test set using `model.predict_classes(...)`
# - Compute average accuracy of the model on the test set


# ## d) Home assignment: impact of initialization
# 
# Let us now study the impact of a bad initialization when training
# a deep feed forward network.
# 
# By default Keras dense layers use the "Glorot Uniform" initialization
# strategy to initialize the weight matrices:
# 
# - each weight coefficient is randomly sampled from [-scale, scale]
# - scale is proportional to $\frac{1}{\sqrt{n_{in} + n_{out}}}$
# 
# This strategy is known to work well to initialize deep neural networks
# with "tanh" or "relu" activation functions and then trained with
# standard SGD.
# 
# To assess the impact of initialization let us plug an alternative init
# scheme into a 2 hidden layers networks with "tanh" activations.
# For the sake of the example let's use normal distributed weights
# with a manually adjustable scale (standard deviation) and see the
# impact the scale value:


from keras import initializers

normal_init = initializers.RandomNormal(stddev=0.01)

N = X_train.shape[1]
H = 100
K = 10

model = Sequential()
model.add(Dense(H, input_dim=N, kernel_initializer=normal_init))
model.add(Activation("tanh"))
model.add(Dense(K, kernel_initializer=normal_init))
model.add(Activation("tanh"))
model.add(Dense(K, kernel_initializer=normal_init))
model.add(Activation("softmax"))

model.compile(optimizer=optimizers.SGD(lr=0.1),
              loss='categorical_crossentropy', metrics=['accuracy'])


print(model.layers)


# Let's have a look at the parameters of the first layer after initialization but before any training has happened:

print(model.layers[0].weights)

w = model.layers[0].weights[0].eval(keras.backend.get_session())
print(w)

print(w.std())


b = model.layers[0].weights[1].eval(keras.backend.get_session())
print(b)

history = model.fit(X_train, Y_train,
                    epochs=15, batch_size=32)


# #### Questions:
# 
# - Try the following initialization schemes and see whether
#   the SGD algorithm can successfully train the network or
#   not:
#   
#   - a very small e.g. `scale=1e-3`
#   - a larger scale e.g. `scale=1` or `10`
#   - initialize all weights to 0 (constant initialization)
#   
# - What do you observe? Can you find an explanation for those
#   outcomes?
# 
# - Are more advanced solvers such as SGD with momentum or Adam able
#   to deal better with such bad initializations?
