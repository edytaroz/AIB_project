import os
from pathlib import Path

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


def data_loading():
    image_dir = Path('shapes')
    filepaths = list(image_dir.glob(r'**/*.png'))
    labels = list(map(lambda x: os.path.split((os.path.split(x)[0]))[1], filepaths))
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images_df = pd.concat([filepaths, labels], axis=1)
    train_df, test_df = train_test_split(images_df, train_size=0.7, shuffle=True, random_state=0)
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation'
    )

    test_images = train_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32
    )
    return train_images, val_images


train_images, val_images = data_loading()


def neuralnetwork():
    # train_images, val_images = data_loading()

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights('my_model_weights.h5')
    for layer in model.layers[:-1]:
        layer.trainable = False

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Fit model on training data")
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=50,
    )

    # print(history.history)
    # model.save_weights('my_model_weights2.h5')
    print("Evaluate on test data")
    return model.get_weights()


def test():
    networks = []
    for _ in range(15):
        weights = neuralnetwork()
        networks.append(weights)

    new_weights = list()
    for weights_list_tuple in zip(*networks):
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))

    train_images, val_images = data_loading()

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.set_weights(new_weights)
    for layer in model.layers[:-1]:
        layer.trainable = False

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Fit model on training data")
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=50,
    )

    print(history.history)
    model.save_weights('my_model_weights2.h5')
    print("Evaluate on test data")
