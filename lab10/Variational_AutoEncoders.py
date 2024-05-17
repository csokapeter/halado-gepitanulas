import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import fashion_mnist


# ## Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


plt.figure(figsize=(16, 8))
for i in range(0, 18):
    plt.subplot(3, 6, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.show()

print(y_train[0:10])

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# ## Standard full-connected VAE model
# 
# Let's define a VAE model with fully connected MLPs for the encoder and decoder networks.
x_train_standard = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_standard = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train_standard.shape, x_test_standard.shape)


# ## Encoder
original_dim = 784
intermediate_dim = 256
latent_dim = 2


def make_encoder(original_dim, intermediate_dim, latent_dim):
    x = Input(shape=(original_dim,))
    hidden = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)
    return Model(inputs=x, outputs=[z_mean, z_log_var],
                name="mlp_encoder")

    
encoder = make_encoder(original_dim, intermediate_dim, latent_dim)
print(encoder.summary())


# ### The VAE stochastic latent variable
def sampling_func(inputs):
    z_mean, z_log_var = inputs
    batch_size = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


sampling_layer = Lambda(sampling_func, output_shape=(latent_dim,),
                        name="latent_sampler")


# ### Decoder
def make_decoder(latent_dim, intermediate_dim, original_dim):
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(intermediate_dim, activation='relu')(decoder_input)
    x = Dense(original_dim, activation='sigmoid')(x)
    return Model(decoder_input, x, name="mlp_decoder")


decoder = make_decoder(latent_dim, intermediate_dim, original_dim)
print(decoder.summary())


# By default the decoder outputs has random weights and output noise:
random_z_from_prior = np.random.normal(loc=0, scale=1, size=(1, latent_dim))
generated = decoder.predict(random_z_from_prior)
plt.imshow(generated.reshape(28, 28), cmap=plt.cm.gray)
plt.axis('off');
plt.show()


def make_vae(input_shape, encoder, decoder, sampling_layer):
    # Build de model architecture by assembling the encoder,
    # stochastic latent variable and decoder:
    x = Input(shape=input_shape, name="input")
    z_mean, z_log_var = encoder(x)
    z = sampling_layer([z_mean, z_log_var])
    x_decoded_mean = decoder(z)
    vae = Model(x, x_decoded_mean)

    # Define the VAE loss
    xent_loss = original_dim * metrics.binary_crossentropy(
        K.flatten(x), K.flatten(x_decoded_mean))
    # Kullback–Leibler divergence
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae

vae = make_vae((original_dim,), encoder, decoder,
               sampling_layer=sampling_layer)
print(vae.summary())

# vae.fit(x_train_standard, epochs=50, batch_size=100,
#         validation_data=(x_test_standard, None))

vae.load_weights("standard_weights.h5")

plt.figure(figsize=(16, 8))
for i in range(0, 18):
    plt.subplot(3, 6, i + 1)
    random_z_from_prior = np.random.normal(size=(1, latent_dim))
    generated = decoder.predict(random_z_from_prior)
    plt.imshow(generated.reshape(28, 28), cmap=plt.cm.gray)
    plt.axis('off');
plt.show()


id_to_labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 
                5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}


x_test_encoded, x_test_encoded_log_var = encoder.predict(x_test_standard, batch_size=100)
plt.figure(figsize=(7, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test,
            cmap=plt.cm.tab10)
cb = plt.colorbar()
cb.set_ticks(list(id_to_labels.keys()))
cb.set_ticklabels(list(id_to_labels.values()))
cb.update_ticks()
plt.show()



# ### 2D panel view of samples from the VAE manifold
n = 15  # figure with 15x15 panels
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()



# ## Convolutional Variational Auto Encoder
x_train_conv = np.expand_dims(x_train, -1)
x_test_conv = np.expand_dims(x_test, -1)
print(x_train_conv.shape, x_test_conv.shape)


from keras.layers import BatchNormalization

img_rows, img_cols, img_chns = 28, 28, 1
filters = 32
kernel_size = 3
intermediate_dim = 128
latent_dim = 2


def make_conv_encoder(img_rows, img_cols, img_chns,
                      latent_dim, intermediate_dim):
    inp = x = Input(shape=(img_rows, img_cols, img_chns))
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu')(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu',
                    strides=(2, 2))(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu')(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu',
                    strides=(2, 2))(x_conv)
    flat = Flatten()(x_conv)
    hidden = Dense(intermediate_dim, activation='relu')(flat)
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)
    return Model(inputs=inp, outputs=[z_mean, z_log_var],
                 name='convolutional_encoder')


conv_encoder = make_conv_encoder(img_rows, img_cols, img_chns,
                                 latent_dim, intermediate_dim)
print(conv_encoder.summary())


sampling_layer = Lambda(sampling_func, output_shape=(latent_dim,),
                        name="latent_sampler")


# ## Decoder
def make_conv_decoder(latent_dim, intermediate_dim, original_dim,
                      spatial_size=7, filters=16):
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(intermediate_dim, activation='relu')(decoder_input)
    x = Dense(filters * spatial_size * spatial_size, activation='relu')(x)
    x = Reshape((spatial_size, spatial_size, filters))(x)
    # First up-sampling:
    x = Conv2DTranspose(filters,
                        kernel_size=3,
                        padding='same',
                        strides=(2, 2),
                        activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters,
                        kernel_size=3,
                        padding='same',
                        strides=1,
                        activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Second up-sampling:
    x = Conv2DTranspose(filters,
                        kernel_size=3,
                        strides=(2, 2),
                        padding='valid',
                        activation='relu')(x)
    x = BatchNormalization()(x)
    # Ouput 1 channel of gray pixels values between 0 and 1:
    x = Conv2D(1, kernel_size=2, padding='valid',
               activation='sigmoid')(x)
    return Model(decoder_input, x, name='convolutional_decoder')


conv_decoder = make_conv_decoder(latent_dim, intermediate_dim, original_dim,
                                 spatial_size=7, filters=filters)
print(conv_decoder.summary())


generated = conv_decoder.predict(np.random.normal(size=(1, latent_dim)))
plt.imshow(generated.reshape(28, 28), cmap=plt.cm.gray)
plt.axis('off');
plt.show()


input_shape = (img_rows, img_cols, img_chns)
vae = make_vae(input_shape, conv_encoder, conv_decoder,
               sampling_layer)
print(vae.summary())


# vae.fit(x_train_conv, epochs=15, batch_size=100,
#         validation_data=(x_test_conv, None))

vae.load_weights("convolutional_weights.h5")


generated = conv_decoder.predict(np.random.normal(size=(1, latent_dim)))
plt.imshow(generated.reshape(28, 28), cmap=plt.cm.gray)
plt.axis('off');
plt.show()


# ### 2D plot of the image classes in the latent space
x_test_encoded, _ = conv_encoder.predict(x_test_conv, batch_size=100)
plt.figure(figsize=(7, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test,
            cmap=plt.cm.tab10)
cb = plt.colorbar()
cb.set_ticks(list(id_to_labels.keys()))
cb.set_ticklabels(list(id_to_labels.values()))
cb.update_ticks()
plt.show()


# ### 2D panel view of samples from the VAE manifold
n = 15  # figure with 15x15 panels
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = conv_decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()