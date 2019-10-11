import tensorflow as tf
from models.ResNetBlock import ResNetBlock as Block

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=32, kernel_size=5)
        self.r_block1 = Block(64)
        self.r_block2 = Block(64)

        self.pool_block1 = Block(128, pool=True)

        self.r_block3 = Block(128)
        self.r_block4 = Block(128)

        self.pool_block2 = Block(128, pool=True)

        self.r_block5 = Block(128)
        self.r_block6 = Block(128)

        self.pool_block3 = Block(256, pool=True)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv(x)
        x = self.r_block1(x)
        x = self.r_block2(x)

        x = self.pool_block1(x)
        x = self.r_block3(x)
        x = self.r_block4(x)

        x = self.pool_block2(x)
        x = self.r_block5(x)
        x = self.r_block6(x)
        x = self.pool_block3(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x
