import glob
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from tensorflow.keras.utils import plot_model

train_data_dir = "C:/Users/lockd/Downloads/archive/train/"
val_data_dir = "C:/Users/lockd/Downloads/archive/valid/"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.15,
    brightness_range=(0.3, 0.8),
    shear_range=0.2
)
val_datagen = ImageDataGenerator(
    rescale=1. / 255,
)
train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    batch_size=16,
    # class_mode default value is clasification
)
val_generator = train_datagen.flow_from_directory(
    directory=val_data_dir,
    target_size=(224, 224),
    batch_size=16,
)


def vgg_block(filters, x, repeats=2):
    for i in range(repeats):
        x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    x = MaxPool2D(strides=2, padding='same')(x)
    return x


inputs = keras.Input(shape=(224, 224, 3))

x = vgg_block(64, inputs, 2)
x = vgg_block(128, x, 2)
x = vgg_block(256, x, 3)
x = vgg_block(512, x, 3)
x = vgg_block(512, x, 3)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(53, activation='softmax')(x)
vgg = keras.Model(inputs, x)
vgg.summary()

#모델 시각화
#plot_model(vgg, show_shapes=True, to_file='model.png')

# vgg.compile(
#     loss='categorical_crossentropy',
#     optimizer=keras.optimizers.Adam(learning_rate=0.0003),
#     metrics=['accuracy'])
# with tf.device("/device:GPU:0"):
#     history = vgg.fit(
#         train_generator,
#         epochs=50,
#         shuffle=True,
#         verbose=1,
#         validation_data=val_generator)

# plt.plot(history.history['loss'])
# plt.plot(np.clip(history.history['val_loss'], 0, 10))
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
