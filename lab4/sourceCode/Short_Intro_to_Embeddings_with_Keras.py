
# coding: utf-8

# # Categorical Embeddings
#
# 
# Let us assume that we are given a pre-trained embedding matrix for an vocabulary of size 10. Each embedding vector in that matrix has dimension 4. Those dimensions are too small to be realistic and are only used for demonstration purposes:


import numpy as np

embedding_size = 4
vocab_size = 10

embedding_matrix = np.arange(embedding_size * vocab_size, dtype='float32')
embedding_matrix = embedding_matrix.reshape(vocab_size, embedding_size)
print(embedding_matrix)

i = 3
print(embedding_matrix[i])


def onehot_encode(dim, label):
    return np.eye(dim)[label]


onehot_i = onehot_encode(vocab_size, i)
print(onehot_i)


embedding_vector = np.dot(onehot_i, embedding_matrix)
print(embedding_vector)


# ### The Embedding layer in Keras
# 
# In Keras, embeddings have an extra parameter, `input_length` which is typically used when having a sequence of symbols
# as input (think sequence of words). In our case, the length will always be 1.
# Furthermore, we load the fixed weights from the previous matrix instead of using a random initialization:

from keras.layers import Embedding

embedding_layer = Embedding(
    output_dim=embedding_size, input_dim=vocab_size,
    weights=[embedding_matrix],
    input_length=1, name='my_embedding')


# Let's use it as part of a Keras model:

from keras.layers import Input
from keras.models import Model

x = Input(shape=[1], name='input')
embedding = embedding_layer(x)
model = Model(inputs=x, outputs=embedding)


# The output of an embedding layer is then a 3-d tensor of shape `(batch_size, sequence_length, embedding_size)`.
print(model.output_shape)

# The embedding weights can be retrieved as model parameters:
print(model.get_weights())


# The `model.summary()` method gives the list of trainable parameters per layer in the model:
print(model.summary())


# We can use the `predict` method of the Keras embedding model to project a single integer label into the matching embedding vector:
labels_to_encode = np.array([[3]])
print(model.predict(labels_to_encode))


labels_to_encode = np.array([[3], [3], [0], [9]])
print(model.predict(labels_to_encode))


# To remove the sequence dimension, useless in our case, we use the `Flatten()` layer
from keras.layers import Flatten

x = Input(shape=[1], name='input')
y = Flatten()(embedding_layer(x))
model2 = Model(inputs=x, outputs=y)


print(model2.output_shape)


print(model2.predict(np.array([3])))