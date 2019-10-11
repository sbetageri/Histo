import tensorflow as tf

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, filters=32, pool=False):
        super(ResNetBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.relu = tf.keras.layers.ReLU()
        self.max_pool = None
        if pool:
            self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2)

    def call(self, x):
        if self.max_pool is not None:
            orig = self.max_pool(x)
            x = self.conv1(orig)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn(x)
            op = tf.concat([x, orig], axis=3)
            op = self.relu(op)
            return op
        else:
            orig = x
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            op = tf.concat([x, orig], axis=3)
            op = self.relu(op)
            return op
