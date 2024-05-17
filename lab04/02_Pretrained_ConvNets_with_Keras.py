
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from zipfile import ZipFile

if not op.exists("images_resize"):
    print('Extracting image files...')
    zf = ZipFile('images_pascalVOC.zip')
    zf.extractall('.')


# # Using a pretrained model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image

model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
print(model.summary())

# ### Classification of an image
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

path = "images_resize/000007.jpg"
img = cv2.imread(path)
plt.imshow(img)

img = cv2.resize(img, (224,224)).astype("float32")
# add a dimension for a "batch" of 1 image
img_batch = preprocess_input(img[np.newaxis])

predictions = model.predict(img_batch)
decoded_predictions= decode_predictions(predictions)

for s, name, score in decoded_predictions[0]:
    print(name, score)

#Computing the representation of an image
input = model.layers[0].input
output = model.layers[-2].output
base_model = Model(input, output)
print(base_model.output_shape)

representation = base_model.predict(img_batch)
print("Shape of representation:", representation.shape)

print("Proportion of zeros in the feature vector: %0.3f"
      % np.mean(representation[0] == 0))

import os
paths = ["images_resize/" + path
         for path in sorted(os.listdir("images_resize/"))]

import h5py

with h5py.File('img_emb.h5', 'r') as h5f:
    out_tensors = h5f['img_emb'][:]

from sklearn.manifold import TSNE

img_emb_tsne = TSNE(perplexity=30).fit_transform(out_tensors)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(img_emb_tsne[:, 0], img_emb_tsne[:, 1]);
plt.xticks(()); plt.yticks(());
plt.show()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.io import imread
from skimage.transform import resize

def imscatter(x, y, paths, ax=None, zoom=1, linewidth=0):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, p in zip(x, y, paths):
        try:
            im = imread(p)
        except:
            print(p)
            continue
        im = resize(im, (224, 224), preserve_range=False, mode='reflect')
        im = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data',
                            frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor='red',
                                           linewidth=linewidth))
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

fig, ax = plt.subplots(figsize=(50, 50))
imscatter(img_emb_tsne[:, 0], img_emb_tsne[:, 1], paths, zoom=0.5, ax=ax)
plt.show()