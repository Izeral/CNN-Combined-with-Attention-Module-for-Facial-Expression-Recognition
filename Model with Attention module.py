import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

train_dir = './train'
test_dir = './test'
img_size = 48 

train_datagen = ImageDataGenerator(width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   horizontal_flip=True,
                                   rescale=1./255,
                                   validation_split=0.2)
validation_datagen = ImageDataGenerator(rescale = 1./255,
                                         validation_split = 0.2)

traingen = train_datagen.flow_from_directory(train_dir,
                                       shuffle = True,
                                       target_size = (img_size, img_size),
                                       color_mode = "grayscale",
                                       class_mode = 'categorical',
                                       subset='training')

valgen = validation_datagen.flow_from_directory(test_dir, 
                                     shuffle = False, 
                                     target_size = (img_size, img_size),
                                     color_mode = "grayscale",
                                     class_mode = 'categorical',
                                     subset='validation')


plt.figure(figsize=(14,22))
i = 1
for expression in os.listdir(train_dir):
    if expression == '.DS_Store':
        continue
    img = load_img(  train_dir + '/' + expression+'/'+ (os.listdir(train_dir + '/'+expression)[1]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()

inputs = tf.keras.Input(shape=(48, 48, 1))
model1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
model1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(model1)
model1 = BatchNormalization()(model1)
model1 = MaxPooling2D(pool_size=(2, 2))(model1)
model1 = Dropout(0.3)(model1)

model1 = Conv2D(256, (3, 3), padding='same', activation='relu')(model1)
model1 = BatchNormalization()(model1)
model1 = MaxPooling2D(pool_size=(2, 2))(model1)
model1 = Dropout(0.25)(model1)

model1 = Conv2D(512, (3, 3), padding='same', activation='relu')(model1)
model1 = BatchNormalization()(model1)
model1 = MaxPooling2D(pool_size=(2, 2))(model1)
model1 = Dropout(0.25)(model1)
b = model1

model1 = keras.layers.GlobalAveragePooling2D()(model1)
model1 = keras.layers.Reshape((1, 1, 512))(model1)
model1 = keras.layers.Dense(units=512 // 4, activation='relu', use_bias=False)(model1)
model1 = keras.layers.Dense(units=512, activation='sigmoid', use_bias=False)(model1)

model1 = keras.layers.Multiply()([b, model1])
model1 = keras.layers.Activation("relu")(model1)

model1 = Flatten()(model1)
model1 = Dense(256,activation = 'relu')(model1)
model1 = BatchNormalization()(model1)
model1 = Dropout(0.25)(model1)

model1 = Dense(512,activation = 'relu')(model1)
model1 = BatchNormalization()(model1)
model1 = Dropout(0.25)(model1)

outputs = Dense(7, activation='softmax')(model1)

model1 = tf.keras.Model(inputs=inputs, outputs=outputs)

model1.compile(
    optimizer = 'SGD', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
  )

model1.summary()

history1 = model1.fit(traingen, epochs = 45, validation_data = valgen)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(history1.history['accuracy'], color='blue')
ax1.plot(history1.history['val_accuracy'], color='blue', linestyle='dashed')
ax2.plot(history1.history['loss'], color='orange')
ax2.plot(history1.history['val_loss'], color='orange', linestyle='dashed')
ax1.set_xlabel('epochs')
ax2.set_ylabel('loss', color='orange')
ax1.set_ylabel('accuracy', color='b')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

model1.save('emo_class.h5')

train_loss, train_accu = model1.evaluate(traingen)
test_loss, test_accu = model1.evaluate(valgen)

#randomly pick an angry img to validate our model
angry_path = './test/angry/PrivateTest_10131363.jpg'
angry_img = load_img(angry_path,target_size = (48,48),color_mode = "grayscale")
plt.imshow(angry_img)
img = np.expand_dims(angry_img,axis = 0) #makes image shape (1,48,48)
img = img.reshape(1,48,48,1)
result = model1.predict(img)
result = list(result[0])
label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
print(result)
img_index = result.index(max(result))
print(label_dict[img_index])
