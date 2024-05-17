# ## Text classification using Neural Networks
# 
# The goal of this notebook is to learn to use Neural Networks for text classification.
# 
# In this notebook, we will:
# - Train a shallow model with learning embeddings
# - Download pre-trained embeddings from Glove
# - Use these pre-trained embeddings
# 
# ### 20 Newsgroups Dataset
# The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups http://qwone.com/~jason/20Newsgroups/
import numpy as np
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

print(newsgroups_train['data'][1000])
print(newsgroups_train['target'][1000])

print(len(newsgroups_train['data']))
print(len(newsgroups_test['data']))

target_names = newsgroups_train["target_names"]

target_id = newsgroups_train["target"][1000]
print("Class of previous message:", target_names[target_id])

print(target_names)

# ### Preprocessing text for the (supervised) CBOW model
# - using a tokenizer. You may use different tokenizers (from scikit-learn, NLTK, custom Python function etc.). 
# This converts the texts into sequences of indices representing the `20000` most frequent words
# - sequences have different lengths, so we pad them (add 0s at the end until the sequence is of length `1000`)
# - we convert the output classes as 1-hot encodings

from keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 20000

# get the raw text data
texts_train = newsgroups_train["data"]
texts_test = newsgroups_test["data"]

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)


print(sequences[0])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print(len(sequences))

# The tokenizer object stores a mapping (vocabulary) from word strings to token ids that can be inverted to reconstruct the original message (without formatting):
type(tokenizer.word_index), len(tokenizer.word_index)

index_word = tokenizer.index_word

print(" ".join([index_word[i] for i in sequences[0]]))

# Let's have a closer look at the tokenized sequences:
seq_lens = [len(s) for s in sequences]
print("average length: %0.1f" % np.mean(seq_lens))
print("max length: %d" % max(seq_lens))

import matplotlib.pyplot as plt

plt.hist(seq_lens, bins=100);

# Let's zoom on the distribution of regular sized posts. The vast majority of the posts have less than 1000 symbols:
plt.hist([l for l in seq_lens if l < 1000], bins=50);
print(np.sum(np.array(seq_lens)>1000))


# Let's truncate and pad all the sequences to 1000 symbols to build the training set:
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 1000

# pad sequences with 0s
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train.shape)
print('Shape of data test tensor:', x_test.shape)


from keras.utils.np_utils import to_categorical
y_train = newsgroups_train["target"]
y_test = newsgroups_test["target"]

print(y_train[100])
y_train = to_categorical(np.asarray(y_train))
print(y_train[100])

print('Shape of label tensor:', y_train.shape)


# ### A simple supervised CBOW model in Keras
# 
# The following computes a very simple model, as described in [fastText](https://github.com/facebookresearch/fastText):
# - Build an embedding layer mapping each word to a vector representation
# - Compute the vector representation of all words in each sequence and average them
# - Add a dense layer to output 20 classes (+ softmax)
from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model

EMBEDDING_DIM = 50
N_CLASSES = len(target_names)

# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embedded_sequences = embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)
predictions = Dense(N_CLASSES, activation='softmax')(average)

model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

print(model.summary())

history = model.fit(x_train, y_train, validation_split=0.1,
          epochs=60, batch_size=128, verbose=1)

# **Exercice**
#  - compute model accuracy on test set
# TODO

pred = model.predict(x_test)
pred_class = np.argmax(pred, axis=-1)
accuracy = np.mean(pred_class==y_test)
print(f'Test accuracy: {accuracy}')

plt.plot(history.history['val_acc'], label='val')
plt.plot(history.history['acc'], label='training')
plt.legend(loc='best')
plt.show()




# ### Building more complex models
# 
# **Exercise**
# - From the previous template, build more complex models using:
#   - 1d convolution and 1d maxpooling. Note that you will still need a GloabalAveragePooling or Flatten after the convolutions
#   - Recurrent neural networks through LSTM (you will need to reduce sequence length before)
# 
# Note: The goal is to build working models rather than getting better test accuracy. 
# To achieve much better results, we'd need more computation time and data quantity. 
# Build your model, and verify that they converge to OK results.
from keras.layers import Conv1D, MaxPooling1D, Flatten

# TODO
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(64, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)

predictions = Dense(20, activation='softmax')(x)

model = Model(inputs=sequence_input, outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

print(model.summary())


history = model.fit(x_train, y_train, validation_split=0.1,
          epochs=10, batch_size=128, verbose=1)


output_test = model.predict(x_test)
test_classes = np.argmax(output_test, axis=-1)
print("test accuracy:", np.mean(test_classes == y_test))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('Loss');
plt.show()


from keras.layers import LSTM, Conv1D, MaxPooling1D

# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# 1D convolution with 64 output channels
x = Conv1D(64, 5)(embedded_sequences)
# MaxPool divides the length of the sequence by 5
x = MaxPooling1D(5)(x)
x = Conv1D(64, 5)(x)
x = MaxPooling1D(5)(x)
# LSTM layer with a hidden size of 64
x = LSTM(64)(x)
predictions = Dense(20, activation='softmax')(x)

model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())

history = model.fit(x_train, y_train, validation_split=0.1,
          epochs=10, batch_size=128, verbose=1)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('Loss');
plt.show()


embeddings = model.layers[1].get_weights()[0]
embeddings = np.vstack(embeddings)
norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
embeddings_normed = embeddings / norms

from sklearn.manifold import TSNE

word_emb_tsne = TSNE(perplexity=30).fit_transform(embeddings_normed[:1000])

import matplotlib.pyplot as plt

plt.figure(figsize=(40, 40))
axis = plt.gca()
np.set_printoptions(suppress=True)
plt.scatter(word_emb_tsne[:, 0], word_emb_tsne[:, 1], marker=".", s=1)

for idx in range(1000):
    plt.annotate(index_word[idx+1],
                 xy=(word_emb_tsne[idx, 0], word_emb_tsne[idx, 1]),
                 xytext=(0, 0), textcoords='offset points')
plt.savefig("tsne.png")
plt.show()

