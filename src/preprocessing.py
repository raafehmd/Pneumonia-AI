import os
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 150    # resize image height
IMG_WIDTH = 150     # resize image width
BATCH_SIZE = 32     # set how many images will be processed at a time

def load_data(base_path):
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    test_path = os.path.join(base_path, 'test')

    train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
    val_test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(train_path, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary', color_mode='grayscale')
    val_data = val_test_gen.flow_from_directory(val_path, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary', color_mode='grayscale')
    test_data = val_test_gen.flow_from_directory(test_path, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary', color_mode='grayscale', shuffle=False)

    return train_data, val_data, test_data