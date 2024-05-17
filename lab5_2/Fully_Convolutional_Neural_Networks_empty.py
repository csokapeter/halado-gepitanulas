
# # Fully Convolutional Neural Networks
#
# Objectives:
# - Load a CNN model pre-trained on ImageNet
# - Transform the network into a Fully Convolutional Network
# - Apply the network perform weak segmentation on images

import warnings

import numpy as np
from skimage.io import imread as scipy_imread
from skimage.transform import resize as scipy_imresize
import matplotlib.pyplot as plt

print(np.random.seed(1))
print(np.random.randint(0,100,10))

# Wrapper functions to disable annoying warnings:
def imread(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scipy_imread(*args, **kwargs)


def imresize(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scipy_imresize(*args, **kwargs)

# Load a pre-trained ResNet50
# We use include_top = False for now
from keras.applications.resnet50 import ResNet50
base_model = ResNet50(include_top=False)

print(base_model.output_shape)

print(base_model.summary())

res5c = base_model.layers[-1]
print(type(res5c))

print(res5c.output_shape)

# ### Fully convolutional ResNet
# #### Regular ResNet layers
#
# The regular ResNet head after the base model is as follows:
# ```py
# x = base_model.output
# x = Flatten()(x)
# x = Dense(1000)(x)
# x = Softmax()(x)
#
# #### Our Version
#
# - We want to retrieve the labels information, which is stored in the Dense layer. We will load these weights afterwards
# - We will change the Dense Layer to a Convolution2D layer to keep spatial information, to output a $W \times H \times 1000$.
# - We can use a kernel size of (1, 1) for that new Convolution2D layer to pass the spatial organization of the previous layer unchanged (it's called a _pointwise convolution_).
# - We want to apply a softmax only on the last dimension so as to preserve the $W \times H$ spatial information.
#
# #### A custom Softmax
#
# We build the following Custom Layer to apply a softmax only to the last dimension of a tensor:
import keras
from keras.engine import Layer
import keras.backend as K

# A custom layer in Keras must implement the four following methods:
class SoftmaxMap(Layer):
    # Init function
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(SoftmaxMap, self).__init__(**kwargs)

    # There's no parameter, so we don't need this one
    def build(self, input_shape):
        pass

    # This is the layer we're interested in:
    # very similar to the regular softmax but note the additional
    # that we accept x.shape == (batch_size, w, h, n_classes)
    # which is not the case in Keras by default.
    # Note that we substract the logits by their maximum to
    # make the softmax more numerically stable.
    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    # The output shape is the same as the input shape
    def get_output_shape_for(self, input_shape):
        return input_shape


# Let's check that we can use this layer to normalize the classes probabilities of some random spatial predictions:
n_samples, w, h, n_classes = 10, 3, 4, 5
random_data = np.random.randn(n_samples, w, h, n_classes)
print(random_data.shape)

# Because those predictions are random, if we some accross the classes dimensions we get random values instead of class probabilities that would need to some to 1:
print(random_data[0].sum(axis=-1))

# Let's wrap the `SoftmaxMap` class into a test model to process our test data:
from keras.models import Sequential

model = Sequential([SoftmaxMap(input_shape=(w, h, n_classes))])
print(model.output_shape)

softmax_mapped_data = model.predict(random_data)
print(softmax_mapped_data.shape)

# All the values are now in the [0, 1] range:
print(softmax_mapped_data[0])

# The last dimension now approximately sum to one, we can therefore be used as class probabilities (or parameters for a multinouli distribution):
print(softmax_mapped_data[0].sum(axis=-1))

# Note that the highest activated channel for each spatial location is still the same before and after the softmax map. The ranking of the activations is preserved as softmax is a monotonic function (when considered element-wise):
print(random_data[0].argmax(axis=-1))
print(softmax_mapped_data[0].argmax(axis=-1))


# #### Exercise
# - What is the shape of the convolution kernel we want to apply to replace the Dense ?
# - Build the fully convolutional model as described above. We want the output to preserve the spatial dimensions but output 1000 channels (one channel per class).
# - You may introspect the last elements of `base_model.layers` to find which layer to remove
# - You may use the Keras Convolution2D(output_channels, filter_w, filter_h) layer and our SotfmaxMap to normalize the result as per-class probabilities.
# - For now, ignore the weights of the new layer(s) (leave them initialized at random): just focus on making the right architecture with the right output shape.

from keras.layers import Convolution2D
from keras.models import Model

input = base_model.layers[0].input

x = base_model.layers[-1].output
x = Convolution2D(1000, (1, 1))(x)

output = SoftmaxMap(axis=-1)(x)
fully_conv_ResNet = Model(inputs=input, outputs=output)



# You can use the following random data to check that it's possible to run a forward pass on a random RGB image:
prediction_maps = fully_conv_ResNet.predict(np.random.randn(1, 224, 224, 3))
print(prediction_maps.shape)

print(prediction_maps.sum(axis=-1))

# ### Loading Dense weights
# - We provide the weights and bias of the last Dense layer of ResNet50 in file `weights_dense.h5`
# - Our last layer is now a 1x1 convolutional layer instead of a fully connected layer
import h5py

with h5py.File('weights_dense.h5', 'r') as h5f:
    w = h5f['w'][:]
    b = h5f['b'][:]

last_layer = fully_conv_ResNet.layers[-2]

print("Loaded weight shape:", w.shape)
print("Last conv layer weights shape:", last_layer.get_weights()[0].shape)

w_reshaped = w.reshape((1, 1, 2048, 1000))

last_layer.set_weights([w_reshaped, b])


# ### A forward pass
# - We define the following function to test our new network.
# - It resizes the input to a given size, then uses `model.predict` to compute the output
from keras.applications.imagenet_utils import preprocess_input

def forward_pass_resize(img_path, img_size):
    img_raw = imread(img_path)
    print("Image shape before resizing: %s" % (img_raw.shape,))
    img = imresize(img_raw, img_size, mode='reflect', preserve_range=True).astype("float32")
    img = preprocess_input(img[np.newaxis])
    print("Image batch size shape before forward pass:", img.shape)
    z = fully_conv_ResNet.predict(img)
    return z,img_raw

output,input = forward_pass_resize("dog.jpg", (600, 800))
print("prediction map shape", output.shape)

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(input)
plt.subplot(1,2,2)
plt.imshow(output[0].max(axis=-1))
plt.show()

# ### Finding dog-related classes
# ImageNet uses an ontology of concepts, from which classes are derived. A synset corresponds to a node in the ontology.
# For example all species of dogs are children of the synset [n02084071](http://image-net.org/synset?wnid=n02084071) (Dog, domestic dog, Canis familiaris):

import imagenet_tool
synset = "n02084071" # synset corresponding to dogs
ids = imagenet_tool.synset_to_dfs_ids(synset)
print("All dog classes ids (%d):" % len(ids))
print(ids)

for dog_id in ids[:10]:
    print(imagenet_tool.id_to_words(dog_id))
print('...')


# ### Unsupervised heatmap of the class "dog"
# The following function builds a heatmap from a forward pass. It sums the representation for all ids corresponding to a synset
def build_heatmap(z, synset):
    class_ids = imagenet_tool.synset_to_dfs_ids(synset)
    class_ids = np.array([id_ for id_ in class_ids if id_ is not None])
    x = z[0, :, :, class_ids].sum(axis=0)
    print("size of heatmap: " + str(x.shape))
    return x

def display_img_and_heatmap(img_path, heatmap):
    dog = imread(img_path)
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 8))
    ax0.imshow(dog)
    ax1.imshow(heatmap, interpolation='nearest', cmap="viridis")
    plt.show()
# **Exercise**
# - What is the size of the heatmap compared to the input image?
# - Build 3 dog heatmaps from `"dog.jpg"`, with the following sizes:
#   - `(400, 640)`
#   - `(800, 1280)`
#   - `(1600, 2560)`
#
# You may plot a heatmap using the above function `display_img_and_heatmap`. You might also want to reuse `forward_pass_resize` to compute the class maps them-selves.

# dog synset
s = "n02084071"

prob1, _ = forward_pass_resize('dog.jpg', (400, 640))
prob2, _ = forward_pass_resize('dog.jpg', (800, 1280))
prob3, _ = forward_pass_resize('dog.jpg', (1600, 2560))

hm1 = build_heatmap(prob1, "n02084071")
hm2 = build_heatmap(prob2, "n02084071")
hm3 = build_heatmap(prob3, "n02084071")

display_img_and_heatmap('/home/csokap/egyetem/halado-gepitanulas/lab5_2/dog.jpg', hm1)
display_img_and_heatmap('/home/csokap/egyetem/halado-gepitanulas/lab5_2/dog.jpg', hm2)
display_img_and_heatmap('/home/csokap/egyetem/halado-gepitanulas/lab5_2/dog.jpg', hm3)


# ### Combining the 3 heatmaps
# By combining the heatmaps at different scales, we obtain a much better information about the location of the dog.
#
# - Combine the three heatmap by resizing them to a similar shape, and averaging them
# - A geometric norm will work better than standard average!

# %load solutions/geom_avg.py

# hm = build_heatmap(output, "n02084071")
# display_img_and_heatmap('dog.jpg', hm)

hm1_res = imresize(hm1, (50,80)).astype('float32')
hm2_res = imresize(hm2, (50,80)).astype('float32')
hm3_res = imresize(hm3, (50,80)).astype('float32')
heatmap_geom_avg = np.power(hm1_res * hm2_res * hm3_res, 0.333)
display_img_and_heatmap("dog.jpg", heatmap_geom_avg)