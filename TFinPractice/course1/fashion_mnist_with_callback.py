import tensorflow as tf


class MnistCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.6:
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


cb = MnistCallback()

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Dividing by 255 to scale to 1
x_train, x_test = x_train / 255, x_test / 255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[cb])

pred = model.evaluate(x_test, y_test)

print(f"the test for model is {pred}")
