import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
from tf_keras_vis.saliency import Saliency
from keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.utils import load_img

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
    img = load_img(train_dir + '/' + expression + '/' + (os.listdir(train_dir + '/'+expression)[1]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()

plt.figure(figsize=(5, 5))
a = traingen[0][0][1]
plt.subplot(1, 3, 1)
plt.title('normal')
plt.imshow(a)
plt.subplot(1, 3, 2)
plt.title('flip_up_down')
plt.imshow(tf.image.flip_up_down(a))
plt.subplot(1, 3, 3)
plt.title('flip_left_right')
plt.imshow(tf.image.flip_left_right(a))

inputs = tf.keras.Input(shape=(48, 48, 1))
model2 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
model2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(model2)
model2 = BatchNormalization()(model2)
model2 = MaxPooling2D(pool_size=(2, 2))(model2)
model2 = Dropout(0.3)(model2)

model2 = Conv2D(256, (3, 3), padding='same', activation='relu')(model2)
model2 = BatchNormalization()(model2)
model2 = MaxPooling2D(pool_size=(2, 2))(model2)
model2 = Dropout(0.25)(model2)

model2 = Conv2D(512, (3, 3), padding='same', activation='relu')(model2)
model2 = BatchNormalization()(model2)
model2 = MaxPooling2D(pool_size=(2, 2))(model2)
model2 = Dropout(0.25)(model2)

model2 = Flatten()(model2)
model2 = Dense(256,activation = 'relu')(model2)
model2 = BatchNormalization()(model2)
model2 = Dropout(0.25)(model2)

model2 = Dense(512,activation = 'relu')(model2)
model2 = BatchNormalization()(model2)
model2 = Dropout(0.25)(model2)

outputs = Dense(7, activation='softmax')(model2)

model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

model2.compile(
    optimizer = 'SGD', # SGD
    loss='categorical_crossentropy', 
    metrics=['accuracy']
  )

print(model.summary())

history = model.fit(traingen, epochs = 40, validation_data = valgen)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(history.history['accuracy'], color='blue')
ax1.plot(history.history['val_accuracy'], color='blue', linestyle='dashed')
ax2.plot(history.history['loss'], color='orange')
ax2.plot(history.history['val_loss'], color='orange', linestyle='dashed')
ax1.set_xlabel('epochs')
ax2.set_ylabel('loss', color='orange')
ax1.set_ylabel('accuracy', color='b')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

model.save('emo_class.h5')

train_loss, train_accu = model.evaluate(traingen)
test_loss, test_accu = model.evaluate(valgen)

angry_path = './test/angry/PrivateTest_10131363.jpg'
angry_img = load_img(angry_path,target_size = (48,48),color_mode = "grayscale")
plt.imshow(angry_img)
img = np.expand_dims(angry_img,axis = 0) # makes image shape (1,48,48)
img = img.reshape(1,48,48,1)
result = model.predict(img)
result = list(result[0])
label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
print(result)
img_index = result.index(max(result))
print(label_dict[img_index])

validation_steps_per_epoch = np.math.ceil(valgen.samples / valgen.batch_size)
predictions = model.predict_generator(valgen, steps=validation_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1) 
true_classes = valgen.classes
class_labels = list(valgen.class_indices.keys())  
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report) 
