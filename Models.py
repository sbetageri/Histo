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


def get_resnet_model():
    resnet_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    resnet_base.trainable = False
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.ZeroPadding2D(padding=(128, 128)))
    model.add(resnet_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model
