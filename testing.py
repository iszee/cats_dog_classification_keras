from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
import os, cv2, re, random
import pandas as pd
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing import image

img_width = 400
img_height = 400
train_data_dir = 'data/train'
valid_data_dir = 'data/validation'
test_data_dir = 'data/ntest/'
test_images_dogs_cats = [test_data_dir+i for i in os.listdir(test_data_dir)]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")



for i in test_images_dogs_cats:
    
    test_image = image.load_img(i,target_size = (img_width, img_height))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict_classes(test_image)

    if result[0][0] == 0:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(i," is a ",prediction)
