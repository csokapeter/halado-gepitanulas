
import matplotlib.pyplot as plt
import numpy as np
import os.path as op

# It will download PascalVOC dataset (400Mo) and
# pre-computed representations of images (450Mo)

import tarfile
try:
    from urllib.request import urlretrieve
except ImportError:  # Python 2 compat
    from urllib import urlretrieve


URL_VOC = ("http://host.robots.ox.ac.uk/pascal/VOC/"
           "voc2007/VOCtrainval_06-Nov-2007.tar")
FILE_VOC = "VOCtrainval_06-Nov-2007.tar"
FOLDER_VOC = "VOCdevkit"

if not op.exists(FILE_VOC):
    print('Downloading from %s to %s...' % (URL_VOC, FILE_VOC))
    urlretrieve(URL_VOC, './' + FILE_VOC)

if not op.exists(FOLDER_VOC):
    print('Extracting %s...' % FILE_VOC)
    tar = tarfile.open(FILE_VOC)
    tar.extractall()
    tar.close()

URL_REPRESENTATIONS = ("https://github.com/m2dsupsdlclass/lectures-labs/"
                       "releases/download/0.2/voc_representations.h5")
FILE_REPRESENTATIONS = "voc_representations.h5"

if not op.exists(FILE_REPRESENTATIONS):
    print('Downloading from %s to %s...'
          % (URL_REPRESENTATIONS, FILE_REPRESENTATIONS))
    urlretrieve(URL_REPRESENTATIONS, './' + FILE_REPRESENTATIONS)


# # Classification and Localisation model
# 
# The objective is to build and train a classification and localisation network. This exercise will showcase the
# flexibility of Deep Learning with several, heterogenous outputs (bounding boxes and classes)

# ## Loading images and annotations
# We will be using Pascal VOC 2007, a dataset and we'll only use 5 classes: "dog", "cat", "bus", "car", "aeroplane".
# 

import numpy as np
import xml.etree.ElementTree as etree
import os
import os.path as op


def extract_xml_annotation(filename):
    z = etree.parse(filename)
    objects = z.findall("./object")
    size = (int(z.find(".//width").text), int(z.find(".//height").text))
    fname = z.find("./filename").text
    dicts = [{obj.find("name").text:[int(obj.find("bndbox/xmin").text), 
                                     int(obj.find("bndbox/ymin").text), 
                                     int(obj.find("bndbox/xmax").text), 
                                     int(obj.find("bndbox/ymax").text)]} 
             for obj in objects]
    return {"size": size, "filename": fname, "objects": dicts}



annotations = []

filters = ["dog", "cat", "bus", "car", "aeroplane"]
idx2labels = {k: v for k, v in enumerate(filters)}
labels2idx = {v: k for k, v in idx2labels.items()}

annotation_folder = "VOCdevkit/VOC2007/Annotations/"
for filename in sorted(os.listdir(annotation_folder)):
    annotation = extract_xml_annotation(op.join(annotation_folder, filename))

    new_objects = []
    for obj in annotation["objects"]:
        # keep only labels we're interested in
        if list(obj.keys())[0] in filters:
            new_objects.append(obj)

    # Keep only if there's a single object in the image
    if len(new_objects) == 1:
        annotation["class"] = list(new_objects[0].keys())[0]
        annotation["bbox"] = list(new_objects[0].values())[0]
        annotation.pop("objects")
        annotations.append(annotation)


print("Number of images with annotations:", len(annotations))

print("Contents of annotation[0]:\n", annotations[0])


print("Correspondence between indices and labels:\n", idx2labels)
print("Correspondence between labels and indices:\n", labels2idx)


# ## Pre-computing representations
# 
# Before designing the object detection model itself, we will pre-process all the dataset to project the images as
# spatial maps in a `(7, 7, 2048)` dimensional space once and for all. The goal is to avoid repeateadly processing the
# data from the original images when training the top layers of the detection network.

# **Exercise**: Load a headless pre-trained `ResNet50` model from Keras and all the layers after the `AveragePooling2D` layer (included):


# TODO

headless_conv = None


# ### Predicting on a batch of images
# 
# The `predict_batch` function is defined as follows:
# - open each image, and resize them to `img_size`
# - stack them as a batch tensor of shape `(batch, img_size_x, img_size_y, 3)`
# - preprocess the batch and make a forward pass with the model


from imageio import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input

def predict_batch(model, img_batch_path, img_size=None):
    img_list = []

    for im_path in img_batch_path:
        img = imread(im_path)
        if img_size:
            img = resize(img,img_size)

        img = img.astype('float32')
        img_list.append(img)
    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError(
            'when both img_size and crop_size are None, all images '
            'in image_paths must have the same shapes.')

    return model.predict(preprocess_input(img_batch))


# Let's test our model:
#TODO



# ### Compute representations on all images in our annotations
# Otherwise, load the pre-trained representations in h5 format using the following:

import h5py
h5f = h5py.File('voc_representations.h5','r')
reprs = h5f['reprs'][:]
h5f.close()


print(reprs.shape)


# ## Building ground truth from annotation
# 
# We cannot use directly the annotation dictionnary as ground truth in our model. #
# #### Boxes coordinates
# - The image is resized to a fixed 224x224 to be fed to the usual ResNet50 input,
# the boxes coordinates of the annotations need to be resized accordingly.
# - We have to convert the top-left and bottom-right coordinates `(x1, y1, x2, y2)` to center, height, width `(xc, yc, w, h)`
# 
# #### Classes labels
# - The class labels are mapped to corresponding indexes

print(annotations)


img_resize = 224
num_classes = len(labels2idx.keys())

def tensorize_ground_truth(annotations):
    all_boxes = []
    all_cls = []
    for idx, annotation in enumerate(annotations):
        # Build a one-hot encoding of the class
        cls = np.zeros((num_classes))
        cls_idx = labels2idx[annotation["class"]]
        cls[cls_idx] = 1.0
        
        coords = annotation["bbox"]
        size = annotation["size"]
        # resize the image
        x1, y1, x2, y2 = (coords[0] * img_resize / size[0],
                          coords[1] * img_resize / size[1], 
                          coords[2] * img_resize / size[0],
                          coords[3] * img_resize / size[1])
        
        # compute center of the box and its height and width
        cx, cy = ((x2 + x1) / 2, (y2 + y1) / 2)
        w = x2 - x1
        h = y2 - y1
        boxes = np.array([cx, cy, w, h])
        all_boxes.append(boxes)
        all_cls.append(cls)

    # stack everything into two big np tensors
    return np.vstack(all_cls), np.vstack(all_boxes)


classes, boxes = tensorize_ground_truth(annotations)
print("Classes and boxes shapes:", classes.shape, boxes.shape)

print("First 2 classes labels:\n")
print(classes[0:2])

print("First 2 boxes coordinates:\n")
print(boxes[0:2])


# ### Interpreting output of model
# Interpreting the output of the model is going from the output tensors to a set of classes
# (with confidence) and boxes coordinates. It corresponds to reverting the previous process.

def interpret_output(cls, boxes, img_size=(500, 333)):
    cls_idx = np.argmax(cls)
    confidence = cls[cls_idx]
    classname = idx2labels[cls_idx]
    cx, cy = boxes[0], boxes[1]
    w, h = boxes[2], boxes[3]
    
    small_box = [max(0, cx - w / 2), max(0, cy - h / 2), 
                 min(img_resize, cx + w / 2), min(img_resize, cy + h / 2)]
    
    fullsize_box = [int(small_box[0] * img_size[0] / img_resize), 
                    int(small_box[1] * img_size[1] / img_resize),
                    int(small_box[2] * img_size[0] / img_resize), 
                    int(small_box[3] * img_size[1] / img_resize)]
    output = {"class": classname, "confidence":confidence, "bbox": fullsize_box}
    return output


# **Sanity check**: interpret the classes and boxes tensors of some known annotations:

img_idx = 1

print("Original annotation:\n")
print(annotations[img_idx])

print("Interpreted output:\n")
print(interpret_output(classes[img_idx], boxes[img_idx],img_size=annotations[img_idx]["size"]))


# ### Intersection over Union
# In order to assess the quality of our model, we will monitor the IoU between ground truth box and predicted box. 
# The following function computes the IoU:

def iou(boxA, boxB):
    # find the intersecting box coordinates
    x0 = max(boxA[0], boxB[0])
    y0 = max(boxA[1], boxB[1])
    x1 = min(boxA[2], boxB[2])
    y1 = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    inter_area = max(x1 - x0, 0) * max(y1 - y0, 0) + 1

    # compute the area of each box
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    return inter_area / float(boxA_area + boxB_area - inter_area)


print(iou([47, 35, 147, 101], [1, 124, 496, 235]))


# **Sanity check** the IoU of the bounding box of the original annotation with
# the bounding box of the interpretation of the resized version of the same annotation be close to 1.0:

img_idx = 1
original = annotations[img_idx]
interpreted = interpret_output(classes[img_idx], boxes[img_idx],
                               img_size=annotations[img_idx]["size"])

print("iou:", iou(original["bbox"], interpreted["bbox"]))


# ### Classification and Localisation model
# 
# A two headed model for classification and localisation

from keras.objectives import mean_squared_error, categorical_crossentropy
from keras.layers import Input, Convolution2D, Dropout, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, GlobalMaxPooling2D
from keras.models import Model


def classif_and_loc(num_classes):
    model_input = Input(shape=(7,7,2048))
    x = GlobalAveragePooling2D()(model_input)
    
    x = Dropout(0.2)(x)
    head_classes = Dense(num_classes, activation="softmax", name="head_classes")(x)
    
    y = Convolution2D(4, (1,1), strides=1, activation='relu', name='hidden_conv')(model_input)
    y = Flatten()(y)
    y = Dropout(0.2)(y)
    head_boxes = Dense(4, name="head_boxes")(y)
    
    model = Model(model_input, outputs = [head_classes, head_boxes], name="resnet_loc")
    model.compile(optimizer="adam", loss=['categorical_crossentropy', "mse"], 
                  loss_weights=[1., 1/(224*224)]) 
    return model


model = classif_and_loc(num_classes)
model.summary()


inputs = reprs[0:10]
out = model.predict(inputs)
print("model output shapes:", out[0].shape, out[1].shape)



# ### Displaying images and bounding box
# The `display` function:
# - takes a single index and computes the result of the model
# - interpret the output of the model as a bounding box
# - calls the `plot_annotations` function



import matplotlib.pyplot as plt

def patch(axis, bbox, display_txt, color):
    coords = (bbox[0], bbox[1]), bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
    axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    axis.text(bbox[0], bbox[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
def plot_annotations(img_path, annotation=None, ground_truth=None):
    img = imread(img_path)
    plt.imshow(img)
    current_axis = plt.gca()
    if ground_truth:
        text = "gt " + ground_truth["class"]
        patch(current_axis, ground_truth["bbox"], text, "red")
    if annotation:
        conf = '{:0.2f} '.format(annotation['confidence'])
        text = conf + annotation["class"]
        patch(current_axis, annotation["bbox"], text, "blue")
    plt.axis('off')
    plt.show()

def display(index, ground_truth=True):
    res = model.predict(reprs[index][np.newaxis,])
    output = interpret_output(res[0][0], res[1][0], img_size=annotations[index]["size"])
    plot_annotations("VOCdevkit/VOC2007/JPEGImages/" + annotations[index]["filename"], 
                     output, annotations[index] if ground_truth else None)


display(2)


display(194)


# ### Computing Accuracy
# For each example `(class_true, bbox_true)`, we consider it positive if and only if:
# - the argmax of `output_class` of the model is `class_true`
# - the IoU between the `output_bbox` and the `bbox_true` is above a threshold (usually `0.5`)


# Compute class accuracy, iou average and global accuracy
def accuracy_and_iou(preds, trues, threshold=0.5):
    sum_valid, sum_accurate, sum_iou = 0, 0, 0
    num = len(preds)
    for pred, true in zip(preds, trues):
        iou_value = iou(pred["bbox"], true["bbox"])
        if pred["class"] == true["class"] and iou_value > threshold:
            sum_valid = sum_valid + 1
        sum_iou = sum_iou + iou_value
        if pred["class"] == true["class"]:
            sum_accurate = sum_accurate + 1
    return sum_accurate / num, sum_iou / num, sum_valid / num


# Compute the previous function on the whole train / test set
def compute_acc(train=True):
    if train:
        beg, end = 0, (9 * len(annotations) // 10)
        split_name = "train"
    else:
        beg, end = (9 * len(annotations)) // 10, len(annotations) 
        split_name = "test"
    res = model.predict(reprs[beg:end])
    outputs = []
    for index, (classes, boxes) in enumerate(zip(res[0], res[1])):
        output = interpret_output(classes, boxes,
                                  img_size=annotations[index]["size"])
        outputs.append(output)
    
    acc, iou, valid = accuracy_and_iou(outputs, annotations[beg:end],
                                       threshold=0.5)
    
    print('{} acc: {:0.3f}, mean iou: {:0.3f}, acc_valid: {:0.3f}'.format(
            split_name, acc, iou, valid) )


compute_acc(train=True)
compute_acc(train=False)


# ### Training on the whole dataset
# We split our dataset into a train and a test dataset

# Keep last examples for test
test_num = reprs.shape[0] // 10
train_num = reprs.shape[0] - test_num
test_inputs = reprs[train_num:]
test_cls, test_boxes = classes[train_num:], boxes[train_num:]
print(train_num)


model = classif_and_loc(num_classes)


batch_size = 32
inputs = reprs[0:train_num]
out_cls, out_boxes = classes[0:train_num], boxes[0:train_num]

history = model.fit(inputs, y=[out_cls, out_boxes], 
                    validation_data=(test_inputs, [test_cls, test_boxes]), 
                    batch_size=batch_size, epochs=10, verbose=2)


import matplotlib.pyplot as plt
plt.plot(np.log(history.history["head_boxes_loss"]), label="boxes_loss")
plt.plot(np.log(history.history["head_classes_loss"]), label="classes_loss")
plt.plot(np.log(history.history["loss"]), label="loss")
plt.legend(loc="upper left")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


compute_acc(train=True)
compute_acc(train=False)


display(194)

