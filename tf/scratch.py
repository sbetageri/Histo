from models.ResNet import ResNet
import numpy as np
import tensorflow as tf

z = np.ones((4, 96, 96, 3))

l = ResNet()

op = l(z)

print(op.shape)
