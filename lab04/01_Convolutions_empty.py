# # Convolutions
# 
# Objectives:
# - Application of convolution on images

# ### Reading and opening images
# 
# The following code enables to read an image, put it in a numpy array and display it in the notebook.



import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.transform import resize


sample_image = imread("bumblebee.png")
sample_image= sample_image.astype(float)

size = sample_image.shape
print("sample image shape: ", sample_image.shape)

plt.imshow(sample_image.astype('uint8'));
plt.show()

# ### A simple convolution filter
# 
# The goal of this section to use Keras to perform individual convolutions on images. This section does not involve training any model yet.
import keras
from keras.models import Sequential
from keras.layers import Conv2D


conv = Sequential([
    Conv2D(filters=3, kernel_size=(5, 5), padding="same",
           input_shape=(None, None, 3))
])
print(conv.output_shape)


print(sample_image.shape)


img_in = np.expand_dims(sample_image, 0)
print(img_in.shape)


img_out = conv.predict(img_in)
print(img_out.shape)


fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.imshow(sample_image.astype('uint8'))
ax1.imshow(img_out[0].astype('uint8'));
plt.show()

print(conv.summary())


print(len(conv.get_weights()))


weights = conv.get_weights()[0]
print(weights.shape)

# One bias per output channel.
biases = conv.get_weights()[1]
print(biases.shape)


# We can instead build a kernel ourselves, by defining a function which will be passed to `Conv2D` Layer.
# We'll create an array with 1/25 for filters, with each channel seperated.
def my_init(shape=(5, 5, 3, 3), dtype=None):
    array = np.zeros(shape=shape, dtype=dtype)
    array[:, :, 0, 0] = 1 / 25
    array[:, :, 1, 1] = 1 / 25
    array[:, :, 2, 2] = 1 / 25
    return array

array = my_init()


print(np.transpose(my_init(), (2, 3, 0, 1)))


conv = Sequential([
    Conv2D(filters=3, kernel_size=(5, 5), padding="same", strides = 2,
           input_shape=(None, None, 3), kernel_initializer=my_init)
])
print(conv.output_shape)


fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.imshow(img_in[0].astype('uint8'))
ax1.imshow(conv.predict(img_in)[0].astype('uint8'));
plt.show()



# convert image to greyscale
grey_sample_image = sample_image.sum(axis=2) / 3.

# add the channel dimension even if it's only one channel so
# as to be consistent with Keras expectations.
grey_sample_image = grey_sample_image[:, :, np.newaxis]


# matplotlib does not like the extra dim for the color channel
# when plotting gray-level images. Let's use squeeze:
plt.imshow(np.squeeze(grey_sample_image.astype(np.uint8)),
           cmap=plt.cm.gray);

plt.show()

# **Exercise**
# - Build an edge detector using `Conv2D` on greyscale image
# - You may experiment with several kernels to find a way to detect edges
# - https://en.wikipedia.org/wiki/Kernel_(image_processing)

# TODO


#img_in = np.expand_dims(grey_sample_image, 0)
#img_out = conv_edge.predict(img_in)

#fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
#ax0.imshow(np.squeeze(img_in[0]).astype(np.uint8),
#           cmap=plt.cm.gray);
#ax1.imshow(np.squeeze(img_out[0]).astype(np.uint8),
#           cmap=plt.cm.gray);
#
#plt.show()
