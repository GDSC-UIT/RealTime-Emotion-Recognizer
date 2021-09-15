import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
import json, os, sys

from data_preprocess import X_train, X_valid, y_train, y_valid, num_classes, img_height, img_depth, img_width

batch_size = 32
epochs = 500
def build_model(model):
    ''' First model '''
    model.add(Conv2D(filters=64,kernel_size=(5,5),input_shape=(img_width, img_height, img_depth),activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64,kernel_size=(5,5),activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    ''' Second layer '''
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    ''' Extra layer '''
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    ''' Third layer '''
    model.add(Flatten())
    model.add(Dense(128,activation='relu',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )
    return model

''' Callbacks '''
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    ModelCheckpoint('model/vgg-face.h5',save_best_only=False,verbose=0),
    early_stopping,
    lr_scheduler,
]

''' Initialize model '''
model = Sequential(name='DCNN')
model = build_model(model)

model.summary()

''' Train model '''
model.fit(X_train, y_train,
          callbacks=callbacks,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_valid, y_valid),
          shuffle=True)

''' Saving the  model to  use it later on'''
fer_json = model.to_json()
with open("model/vgg-face-model.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("model/vgg-face.h5")
