
# # Explicit Feedback Neural Recommender Systems
# 
# Goals:
# - Understand recommender data
# - Build different models architectures using Keras
# - Retrieve Embeddings and visualize them
# - Add metadata information as input to the model

import matplotlib.pyplot as plt
import numpy as np
import os.path as op

from zipfile import ZipFile

try:
    from urllib.request import urlretrieve
except ImportError:  # Python 2 compat
    from urllib import urlretrieve


ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_100K_FILENAME = ML_100K_URL.rsplit('/', 1)[1]
ML_100K_FOLDER = 'ml-100k'

if not op.exists(ML_100K_FILENAME):
    print('Downloading %s to %s...' % (ML_100K_URL, ML_100K_FILENAME))
    urlretrieve(ML_100K_URL, ML_100K_FILENAME)

if not op.exists(ML_100K_FOLDER):
    print('Extracting %s to %s...' % (ML_100K_FILENAME, ML_100K_FOLDER))
    ZipFile(ML_100K_FILENAME).extractall('.')


# ### Ratings file
# 
# Each line contains a rated movie: 
# - a user
# - an item
# - a rating from 1 to 5 stars


import pandas as pd
pd.set_option('display.max_columns',100)

raw_ratings = pd.read_csv(op.join(ML_100K_FOLDER, 'u.data'), sep='\t',
                      names=["user_id", "item_id", "rating", "timestamp"])
print(raw_ratings.head())


# ### Item metadata file
# 
# The item metadata file contains metadata like the name of the movie or the date it was released. The movies file contains columns indicating the movie's genres. Let's only load the first five columns of the file with `usecols`.

m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
items = pd.read_csv(op.join(ML_100K_FOLDER, 'u.item'), sep='|',
                    names=m_cols, usecols=range(5), encoding='latin-1')
print(items.head())


# Preprocessing code to extract the release year as an integer value:
def extract_year(release_date):
    if hasattr(release_date, 'split'):
        components = release_date.split('-')
        if len(components) == 3:
            return int(components[2])
    return 1920


items['release_year'] = items['release_date'].map(extract_year)
items.hist('release_year', bins=50);
plt.show()


all_ratings = pd.merge(items, raw_ratings)
print(all_ratings.head())


# ### Data preprocessing

# - the number of users
max_user_id = all_ratings['user_id'].max()
print(max_user_id)

# - the number of items
max_item_id = all_ratings['item_id'].max()
print(max_item_id)

# - the rating distribution
print(all_ratings['rating'].describe())


# - the popularity of each movie
popularity = all_ratings.groupby('item_id').size().reset_index(name='popularity')
items = pd.merge(popularity, items)
print(items.nlargest(10, 'popularity'))


# Enrich the ratings data with the popularity as an additional metadata.
all_ratings = pd.merge(popularity, all_ratings)
print(all_ratings.head())


# Let's split the enriched data in a train / test split to make it possible to do predictive modeling:
from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(
    all_ratings, test_size=0.2, random_state=0)

user_id_train = ratings_train['user_id']
item_id_train = ratings_train['item_id']
rating_train = ratings_train['rating']

user_id_test = ratings_test['user_id']
item_id_test = ratings_test['item_id']
rating_test = ratings_test['rating']


# # Explicit feedback: supervised ratings prediction
# 
# For each pair of (user, item) try to predict the rating the user would give to the item.
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers import Dot
from keras.models import Model


user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 30
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

y = Dot(axes=1)([user_vecs, item_vecs])

model = Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(optimizer='adam', loss='mae')


initial_train_preds = model.predict([user_id_train, item_id_train])
print(initial_train_preds.shape)


# ### Model error
# Using `initial_train_preds`, compute the model errors:
# - mean absolute error
# - mean squared error

squared_differences = np.square(initial_train_preds[:,0] - rating_train.values)
absolute_differences = np.abs(initial_train_preds[:,0] - rating_train.values)

print("Random init MSE: %0.3f" % np.mean(squared_differences))
print("Random init MAE: %0.3f" % np.mean(absolute_differences))


from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Random init MSE: %0.3f" % mean_squared_error(initial_train_preds, rating_train))
print("Random init MAE: %0.3f" % mean_absolute_error(initial_train_preds, rating_train))

# ### Monitoring runs
# 
# Keras enables to monitor various variables during training. 
history = model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, epochs=6, validation_split=0.1,
                    shuffle=True)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('Loss');
plt.show()


test_preds = model.predict([user_id_test, item_id_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))


train_preds = model.predict([user_id_train, item_id_train])
print("Final train MSE: %0.3f" % mean_squared_error(train_preds, rating_train))
print("Final train MAE: %0.3f" % mean_absolute_error(train_preds, rating_train))


# ## A Deep recommender model
# 
# Using a similar framework as previously, the following deep model described in the course was built (with only two fully connected)

from keras.layers import Concatenate

# 
# ### Exercise
# 
# - The following code has **4 errors** that prevent it from working correctly. **Correct them and explain** why they are critical.


# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 30
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = Concatenate()([user_vecs, item_vecs])
input_vecs = Dropout(0.4)(input_vecs)

x = Dense(64, activation='relu')(input_vecs)
y = Dense(1, activation='linear')(x)

model = Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train])


history = model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, epochs=6, validation_split=0.1,
                    shuffle=True)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('Loss');
plt.show()


train_preds = model.predict([user_id_train, item_id_train])
print("Final train MSE: %0.3f" % mean_squared_error(train_preds, rating_train))
print("Final train MAE: %0.3f" % mean_absolute_error(train_preds, rating_train))

test_preds = model.predict([user_id_test, item_id_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))


# ### Model Embeddings
# 
# - It is possible to retrieve the embeddings by simply using the Keras function `model.get_weights` which returns all the model learnable parameters.
# - The weights are returned the same order as they were build in the model

weights = model.get_weights()
print([w.shape for w in weights])


# Solution: 
print(model.summary())


user_embeddings = weights[0]
item_embeddings = weights[1]
print("First item name from metadata:", items["title"][1])
print("Embedding vector for the first item:")
print(item_embeddings[1])
print("shape:", item_embeddings[1].shape)


# ### Finding most similar items
# Finding k most similar items to a point in embedding space
# 
# - Write in numpy a function to compute the cosine similarity between two points in embedding space
# - Write a function which computes the euclidean distance between a point in embedding space and all other points
# - Write a most similar function, which returns the k item names with lowest euclidean distance
# - Try with a movie index, such as 181 (Return of the Jedi). What do you observe? Don't expect miracles on such a small training set.


EPSILON = 1e-07

def cosine(x, y):
    # TODO:
    pass

# Computes cosine similarities between x and all item embeddings
def cosine_similarities(x):
    # TODO:
    pass

# Computes euclidean distances between x and all item embeddings
def euclidean_distances(x):
    # TODO:
    pass

# Computes top_n most similar items to an idx
def most_similar(idx, top_n=10, mode='euclidean'):
    # TODO:
    pass


print(most_similar(180))

# sanity checks:
print("cosine of item 1 and item 1: %0.3f"
      % cosine(item_embeddings[1], item_embeddings[1]))
euc_dists = euclidean_distances(item_embeddings[1])
print(euc_dists.shape)
print(euc_dists[1:5])
print()

# Test on movie 180: Return of the Jedi
print("Items closest to 'Return of the Jedi':")
for title, dist in most_similar(180, mode="euclidean"):
    print(title, dist)


# ### Visualizing embeddings using TSNE
# 
# - we use scikit learn to visualize items embeddings
# - Try different perplexities, and visualize user embeddings as well

from sklearn.manifold import TSNE

item_tsne = TSNE(perplexity=30).fit_transform(item_embeddings)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(item_tsne[:, 0], item_tsne[:, 1])
plt.xticks(()); plt.yticks(())
plt.show()


def recommend(user_id, top_n=10):
    item_ids = range(1, max_item_id)
    seen_mask = all_ratings["user_id"] == user_id
    seen_movies = set(all_ratings[seen_mask]["item_id"])
    item_ids = list(filter(lambda x: x not in seen_movies, item_ids))

    print("User %d has seen %d movies, including:" % (user_id, len(seen_movies)))
    for title in all_ratings[seen_mask].nlargest(20, 'popularity')['title']:
        print("   ", title)
    print("Computing ratings for %d other movies:" % len(item_ids))
    
    item_ids = np.array(item_ids)
    user_ids = np.zeros_like(item_ids)
    user_ids[:] = user_id
    
    rating_preds = model.predict([user_ids, item_ids])
    
    item_ids = np.argsort(rating_preds[:, 0])[::-1].tolist()
    rec_items = item_ids[:top_n]
    return [(items["title"][movie], rating_preds[movie][0])
            for movie in rec_items]
			
for title, pred_rating in recommend(5):
    print("    %0.1f: %s" % (pred_rating, title))