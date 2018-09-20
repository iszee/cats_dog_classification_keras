from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


# step 1: load data

img_width = 1023
img_height = 768
train_data_dir = 'data/train'
valid_data_dir = 'data/validation'
#test_data_dir = 'data/test'

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,target_size=(img_width,img_height),classes=['dogs','cats'],class_mode='binary',batch_size=16)

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,target_size=(img_width,img_height),classes=['dogs','cats'],class_mode='binary',batch_size=32)

#test_generator = datagen.flow_from_directory(directory=test_data_dir,target_size=(img_width,img_height),classes=['dogs','cats'],class_mode='binary',batch_size=32)


# step-2 : build model

model =Sequential()

model.add(Conv2D(512,(3,3),strides=(2, 2),padding='valid', activation='relu',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),strides=(2, 2),padding='valid', activation='relu',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),strides=(2, 2),padding='valid', activation='relu',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print('model complied!!')
print('starting training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=2048 // 16,epochs=12,validation_data=validation_generator,validation_steps=832//16)
print('training finished!!')

model.save('my_model.h5')

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(training.history['loss'],'r',linewidth=3.0)
plt.plot(training.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(training.history['acc'],'r',linewidth=3.0)
plt.plot(training.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


#print('starting testing....')
#training = model.fit_generator(generator=test_generator, steps_per_epoch=2048 // 16,epochs=20,validation_data=validation_generator,validation_steps=832//16)
#print('test finished!!')

