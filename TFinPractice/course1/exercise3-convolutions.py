import tensorflow as tf


class MYCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > .998:
            print("\nReached 99.8 % accuracy, stopping training")
            self.model.stop_training = True


mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# print(training_images.shape)
# print(test_images.shape)

# (60000, 28, 28)
# (10000, 28, 28)

cb = MYCallBack()

training_images = training_images.reshape(60000, 28, 28, 1) / 255
test_images = test_images.reshape(10000, 28, 28, 1) / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, callbacks=[cb])
test_loss = model.evaluate(test_images, test_labels)
print(f"The loss for test data is {test_loss}")
