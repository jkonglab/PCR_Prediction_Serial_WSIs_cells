import tensorflow as tf
print(tf.__version__)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cv2
import numpy as np
from glob import glob
from models import Yolov4
from tensorflow.keras import layers, models, optimizers
import keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

#### replace with your own data loading
X1_train=np.random.rand(8,416,416,3) ## HE images
X2_train=np.random.rand(8,416,416,9) ## stacked HE, Ki-67, and PHH3 images
Y_train=np.zeros((8,2))
Y_train[:4,0]=1
Y_train[4:,1]=1
X1_test=np.random.rand(1,416,416,3)
X2_test=np.random.rand(1,416,416,9)
Y_test=np.zeros((1,2))
###

batch_size = 8
nb_epoch = 200



model = Yolov4(weight_path='./TumorDetector.weights', ## your own trained tumor detection model
               class_name_path='./class_names/coco_classes.txt')


detect_model = models.Model(inputs=model.yolo_model.input,
                            outputs=model.yolo_model.get_layer('conv2d_93').input)

x = detect_model.output

### spatial attention module
x = layers.Conv2D(128,3,padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2D(128,3,padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2DTranspose(128,3,strides=(2,2),padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)

x = layers.Conv2D(64,3,padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2D(64,3,padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2DTranspose(64,3,strides=(2,2),padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)

x = layers.Conv2D(32,3,padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2D(32,3,padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2DTranspose(1,3,strides=(2,2),padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('sigmoid')(x)


### Prediction module

input2 = layers.Input((416,416,9))
x = K.repeat_elements(x,9,3)
x = layers.Multiply()([x,input2]) 

x = layers.Conv2D(64, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(64, (3, 3),activation='relu',padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(128, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(128, (3, 3),activation='relu',padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(256, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(256, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(256, (3, 3),activation='relu',padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(512, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(512, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(512, (3, 3),activation='relu',padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(512, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(512, (3, 3),activation='relu',padding='same')(x)
x = layers.Conv2D(512, (3, 3),activation='relu',padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.GlobalAveragePooling2D()(x)
#x = layers.Dense(4096, activation='relu')(x)
x = layers.Dense(2, activation='softmax')(x)

model = models.Model([detect_model.input,input2], x)
###
print(model.summary())

model.compile(loss='categorical_crossentropy',
              #optimizer='SGD',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])

checkpoint_filepath = './PredictionCheckpoints/'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath+'model.{epoch:03d}-{val_loss:.3f}-{val_acc:.3f}.h5',monitor='val_loss',mode='min',save_weights_only=True)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
model.fit([X1_train,X2_train], Y_train, batch_size=batch_size,
                        steps_per_epoch=X1_train.shape[0] // batch_size,
                        validation_data=([X1_test,X2_test], Y_test),
                        epochs=nb_epoch, verbose=1, shuffle=True,
                        #callbacks=[early_stopper, model_checkpoint_callback])
                        callbacks=[model_checkpoint_callback])


