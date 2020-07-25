import tensorflow as tf
import os
import zipfile

DESIRED_ACCURACY = 0.999

# Processing the data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import binary_crossentropy

train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory('/Users/vipul/PycharmProjects/TFExams/TFinPractice/data/happy-or'
                                                    '-sad', target_size=(256, 256), batch_size=10, class_mode='binary')


# Defining the callback

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] > DESIRED_ACCURACY:
            print("\nAchieved accuracy greater than .999 so stopping")
            self.model.stop_training = True


cb = MyCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    # check input shape and make sure con2d is less then next Conv Layers
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss=binary_crossentropy, metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=8, epochs=5, callbacks=[cb])
