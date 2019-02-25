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

# step 1: load data

img_width = 400
img_height = 400
train_data_dir = 'train'
valid_data_dir = 'validation'
test_data_dir = 'data/test'

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(directory=train_data_dir,target_size=(img_width,img_height),classes=['dogs','cats'],class_mode='binary',batch_size=16)
validation_generator = datagen.flow_from_directory(directory=valid_data_dir,target_size=(img_width,img_height),classes=['dogs','cats'],class_mode='binary',batch_size=16)
#test_generator = datagen.flow_from_directory(directory=test_data_dir,target_size=(img_width,img_height),classes=['dogs','cats'],class_mode='binary',batch_size=32)


# step-2 : build

model =Sequential()

model.add(Conv2D(64,(3,3),strides=(2, 2),padding='same',activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',input_shape=(img_width,img_height,3)))
#model.add(Conv2D(32,(3,3),strides=(2, 2),padding='same',activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',input_shape=(img_width,img_height,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))

model.add(Conv2D(128,(3,3),strides=(2, 2),padding='same',activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',input_shape=(img_width,img_height,3)))
#model.add(Conv2D(64,(3,3),strides=(2, 2),padding='same',activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',input_shape=(img_width,img_height,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))

model.add(Conv2D(256,(3,3),strides=(2, 2),padding='same',activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',input_shape=(img_width,img_height,3)))
#model.add(Conv2D(128,(3,3),strides=(2, 2),padding='same',activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',input_shape=(img_width,img_height,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))



model.add(Flatten())



model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# step-3 : training

print('model complied!!')
print('starting training....')
training = model.fit_generator(generator=train_generator,steps_per_epoch=12000//16,epochs=50,shuffle=True,validation_data=validation_generator,validation_steps=4002//16)
print('training finished!!')


# step-4 : model save

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

test_image = image.load_img('data/test/1.jpg', target_size = (img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
'''

# step-5 : plot graphs

print(training.history.keys())
plt.plot(training.history['acc'])
plt.plot(training.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

