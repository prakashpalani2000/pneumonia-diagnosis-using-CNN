!pip install -q visualkeras
!pip install -q ann_visualizer
!pip install -q dtreeviz


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras import regularizers

from tensorflow.keras.preprocessing import image
import visualkeras
from ann_visualizer.visualize import ann_viz
from dtreeviz.trees import *
from tensorflow.keras.utils import plot_model

import warnings 
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/Colab Notebooks/pneumonia/Training'
data = tf.keras.preprocessing.image_dataset_from_directory(data_dir)


datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest',
        validation_split = 0.2)

height = 224
width = 224
channels = 3
batch_size = 32
img_shape = (height, width, channels)
img_size = (height, width)


train_data = datagen.flow_from_directory(
    data_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training')

val_data = datagen.flow_from_directory(
    data_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode='categorical',
    subset = 'validation')

def plotImages(image_arr):
    fig,axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(image_arr,axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

img_array = [train_data[0][0][0] for i in range(6)]
plotImages(img_array)

num_classes = len(data.class_names)
print('.... Number of Classes : {0} ....'.format(num_classes))
model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=(224,224,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="softmax"))

model.compile(
       optimizer = "adam",
          loss = "categorical_crossentropy",
    metrics = ['accuracy']
)

model.summary()


STEP_SIZE_TRAIN = train_data.n // train_data.batch_size
STEP_SIZE_VALID = val_data.n // val_data.batch_size

history = model.fit_generator(train_data,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = val_data,
                    validation_steps = STEP_SIZE_VALID,
                    epochs = 2,
                    verbose = 1)

model.save('model.h5')

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

plt.figure(figsize=(6, 3), dpi=150)
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='training set')
plt.plot(history.history['val_loss'], label= 'test set')
plt.legend()
plt.figure(figsize=(6, 3), dpi=150)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label= 'training set')
plt.plot(history.history['val_accuracy'], label='test set')
plt.legend()


ann_viz(model, view=True, title='Custom model')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

import warnings 
warnings.filterwarnings('ignore')

classes = ['COVID-19','Normal','Pneumonia-Bacterial','Pneumonia-Viral']

from tensorflow.keras.models import load_model
model=load_model('model.h5')

def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    prediction = model.predict(img_processed)
    print(prediction)
    index = np.argmax(prediction)
    plt.title("Prediction - {}".format(str(classes[index]).title()), size=18, color='red')
    plt.imshow(img_array)


predict_image('/content/drive/MyDrive/Colab Notebooks/pneumonia/Training/COVID-19/COVID-19 (1).jpg', model)


predict_image('/content/drive/MyDrive/Colab Notebooks/pneumonia/Training/Normal/Normal (1).jpg', model)
predict_image('/content/drive/MyDrive/Colab Notebooks/pneumonia/Training/Pneumonia-Bacterial/Pneumonia-Bacterial (1).jpg', model)
