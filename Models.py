import tensorflow as tf

def get_base_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    return model

