from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

import random
import os
import pathlib
import zipfile
import datetime


###-----------data visualization-----------###

def view_random_img(target_dir, target_class):
    """
    Picks a random image from a selected class and displays it together with the class and shape.
    Expects images annotated in folder structure
    dataset / train_data    / label1    / img1
                                        / img2
                                        / img3
                                        ...
                            / label2...
                            / label3...
                            ...
            / test_data     / label1...
                            / label2...
                            / label3...
                            ...
    :param target_dir: (str) path, typically train or test data
    :param target_class: (str) of the desired class
    :return: plt image
    """
    target_folder = target_dir + "/" + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")


def get_random_image_and_class(target_dir, train_data):
    """
    :param target_dir: (str) path to train or test directory
    :param train_data: train_data returned by image_dataset_from_directory
    :return: image object and target class string
    """
    #TODO test with ImageDataGenerator
    target_class = random.choice(train_data.class_names)
    target_folder = target_dir + "/" + target_class
    random_image = random.choice(os.listdir(target_folder))
    img = mpimg.imread(target_folder + "/" + random_image)

    return img, target_class


def view_multiple_images(train_data, labels):
    """
    plots multiple (4) images together with their label to get familiar with the data. Used with imported test data
    sets, e.g. fashion mnist
    :param train_data: (str) path to training data
    :param labels: (array) of labels
    """
    plt.figure(figsize=(10,7))
    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        rand_index = random.choice(range(len(train_data)))
        plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
        plt.title(labels[labels[rand_index]])
        plt.axis(False)


###-----------data manipulation-----------###

def unzip_file(file):
    zip_ref = zipfile.ZipFile(file)
    zip_ref.extractall()
    zip_ref.close()

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3) using tensorflow library
    :param filename: (str) filename of target image
    :param img_shape: (int) size to resize target image to, default 224
    :param scale: (bool) whether to scale pixel values to range(0, 1), default True
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img/255.
    else:
        return img


def generate_image_data_idg(data_dir, target_size=(224,224), class_mode="binary"):
    """
    loads images for testing purposes using ImageDataGenerator
    :param data_dir: (str) path to the test data
    :param target_size: (tuple) default is (224,224)
    :param class_mode: (str) default is binary, change to categorical if needed
    :return: data and labels
    """
    datagen = ImageDataGenerator(rescale=1/255.)
    dataset = datagen.flow_from_directory(data_dir,
                                       target_size=target_size,
                                       class_mode=class_mode)

    return dataset


def generate_augmented_data_idg(data_dir, amount=0.2, target_size=(224,224), class_mode="binary", shuffle=True):
    """
    adds augmentation to images to enhance training using ImageDataGenerator. Rescales images to 0-1
    :param data_dir: (str) path to the train dataset
    :param amount: (float) the degree of augmentation, default is 0.2
    :param target_size: (tuple) defaults to (224,224)
    :param class_mode: (str) defaults to binary, can be changed to categorical
    :param shuffle: (bool) defaults to true, can be turned off
    :return: augmented images and augmented labels
    """
    datagen = ImageDataGenerator(rescale=1/255.,
                                 rotation_range=amount,
                                 shear_range=amount,
                                 zoom_range=amount,
                                 width_shift_range=amount,
                                 height_shift_range=amount,
                                 horizontal_flip=True)
    augmented_dataset = datagen.flow_from_directory(data_dir,
                                                 target_size=target_size,
                                                 batch_size=32,
                                                 class_mode=class_mode,
                                                 shuffle=shuffle)
    return augmented_dataset


def generate_image_data_idfd(train_dir, test_dir, label_mode="categorical"):
    """
    generates data using image_dataset_from_directory
    :param train_dir:
    :param test_dir:
    :param label_mode:
    :return:
    """
    train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                     image_size=(224,224),
                                                                     label_mode=label_mode)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                    image_size=(224,224),
                                                                    label_mode=label_mode)
    return train_data, test_data


def get_data_and_labels_from_gen_dataset(dataset):
    """
    returns images and labels from ImageDataGenerator generated dataset
    :param dataset: data returned by generate_augmented_data()
    :return: images and labels in the form of tensors
    """
    data, labels = next(dataset)
    return data, labels


def display_augmented_image(images, aug_images):
    """
    plots an augmented image next to the original
    :param images: (array) output from ImageDataGenerator
    :param aug_images: (array) output from ImageDataGenerator
    :return: plot
    """
    random_number = random.randint(0, len(images))
    print(f"Showing image number {random_number}")
    plt.imshow(images[random_number])
    plt.title("Original image")
    plt.axis=False
    plt.figure()
    plt.imshow(aug_images[random_number])
    plt.title("Augmented image")
    plt.axis=False





###-----------data visualization with predictions-----------###

def plot_and_predict_random_image(model, images, true_labels, classes):
    """
    picks a random image, plots it and labels it with the prediction and truth classes
    :param model: trained model to be used for prediction, input shape is (28,28)
    :param images: (array) list of images, an image will be reshaped before prediction, X_test
    :param true_labels: (array) y_test
    :param classes: (array) of classes
    """
    i = random.randint(0, len(images))

    # create predictions and targets
    target_image = images[i]
    predicted_probabilities = model.predict(target_image.reshape(1,28,28))
    predicted_label = classes[predicted_probabilities.argmax()] # returns the highest probability index
    true_label = classes[true_labels[i]]

    # plot the image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # change colour of titles based on result
    if predicted_label == true_label:
        color = "green"
    else:
        color = "red"

    # add xlabel info
    plt.xlabel(f"Pred: {predicted_label} {100 * tf.reduce_max(predicted_probabilities):2.0f}% True: {true_label}", color=color)


def plot_and_predict_image(model, filename, class_names):
    """
    plotting image with predictions for binary classification
    :param model: trained model
    :param filename: (str) filename of target image
    :param class_names: (array) of two classnames
    :return: plot
    """
    img =  load_and_prep_image(filename)
    prediction = model.predict(tf.expand_dims(img, axis=0))
    prediction_class = class_names[int(tf.round(prediction))]
    plt.imshow(img)
    plt.title(f"Prediction: {prediction_class}")
    plt.axis=False


def plot_decision_boundary(model, X, y):
    """
    Used with the circles dataset
    :param model:
    :param X: training data
    :param y: training labels
    :return:
    """
    x_min = X[:, 0].min() - 0.1
    x_max = X[:, 0].max() + 0.1
    y_min = X[:, 1].min() - 0.1
    y_max = X[:, 1].max() + 0.1

    # use meshgrid to return a tuple of coordinate matrices from coordinate vectors.
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), # generate an array of evenly spaced values between two specified numbers
                        np.linspace(y_min, y_max, 100))

    # stack 2D arrays together
    x_in = np.c_[xx.ravel(), yy.ravel()] # ravel to remove the dimension and create an array

    y_pred = model.predict(x_in)

    # check for multiclass classification
    if len(y_pred[0]) > 1:
        print('Doing multi class classification')
        prediction = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        prediction = np.round(y_pred).reshape(xx.shape)

    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


###-----------model training and results-----------###

def get_classes_from_folder_structure(folder):
    """
    get class names from folder structure for classification
    :param folder: (str) path
    :return: (np array) of classes
    """
    data_dir = pathlib.Path(folder)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    return class_names


def get_image_classification_predictions(model, test_data):
    """
    :param model: model instance
    :param test_data: (str) path to generated image data using generate_image_data() or load_and_prep_image()
    :return: predictions
    """
    dataset = generate_image_data_idg(test_data)
    images, labels = get_data_and_labels_from_gen_dataset(dataset)
    probabilities = model.predict(images)

    # check for multiclass vs. binary classification
    if len(probabilities[0]) > 1:
        predictions = np.argmax(probabilities, axis=1)
    else:
        predictions = np.round(probabilities)
    return predictions

def create_confusion_matrix(y_true, y_preds, classes=None, figsize=(10, 10), text_size=20):
    """
    creates and displays confusion matrix
    :param y_true: (array) true labels, must be same shape as y_preds
    :param y_preds: (array) predicted labels
    :param classes: (array) (str) human-readable class names. If none is provided integers are used
    :param figsize: (tuple) defaults to (10,10)
    :param text_size: (int) defaults to 20
    :return: confusion matrix
    """

    # create confusion matrix
    cm = confusion_matrix(y_true, y_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # create the plots
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # set labels to be classes
    if classes:
      labels = classes
    else:
      labels = np.arange(cm.shape[0])

    # label the axes
    ax.set(title="Confusion matrix",
          xlabel="Predicted label",
          ylabel = "True label",
          xticks=np.arange(n_classes),
          yticks=np.arange(n_classes),
          xticklabels=labels,
          yticklabels=labels)

    # set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # set threshold for different colours
    threshold = (cm.max() + cm.min()) / 2

    # plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i,j] > threshold else "black",
              size=7)


def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # plot loss
    plt.figure()
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend()

    # plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.legend()


###-----------creating models-----------###

def create_feature_extraction_model(model_url, image_shape, num_classes=10):
    """
    create an unfit feature extraction model from tensorflow hub
    :param model_url: (str)
    :param image_shape: (tuple) shape of the input image
    :param num_classes: (int) defaults to 10
    :return: unfit model instance
    """
    model = hub.resolve(model_url)
    inputs = tf.keras.Input(
        shape=image_shape + (3,),
        name="input_layer"
    )
    feature_extractor_layer = tf.keras.layers.TFSMLayer(
        model,
        trainable=False,
        name='feature_extraction_layer'
    )
    x = feature_extractor_layer(inputs)
    tensor_name = list(x.keys())[0]
    x = x[tensor_name]
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='output_layer'
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def create_tensorboard_callback(dir_name, experiment_name):
    """
    :param dir_name: (str) path to directory
    :param experiment_name: (str) experiment name
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard  log files to {log_dir}")
    return tensorboard_callback