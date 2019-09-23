import tensorflow as tf
import pandas as pd
import numpy as np

from PIL import Image

class HistDataset(tf.keras.utils.Sequence):
    TRAIN_SET = 9
    VAL_SET = 18
    TEST_SET = 27
    def __init__(self, dataframe, img_dir, dataset_flag=TRAIN_SET, img_dim=(96, 96)):
        super(HistDataset, self).__init__()
        self.df = dataframe
        self.img_dir = img_dir
        self.dataset_flag = dataset_flag
        self.batch_size = 4
        self.img_dim = img_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_id = data['id']
        label = data['label']

        img = Image.open(self.img_dir + img_id + '.tif')
        img = img.resize(self.img_dim)
        imgs = [
            np.array(img.transpose(Image.FLIP_LEFT_RIGHT)),
            np.array(img.transpose(Image.FLIP_TOP_BOTTOM)),
            np.array(img.transpose(Image.ROTATE_90)),
            np.array(img)
        ]

        ## ToDo
        ## Do we normalise all the images?

        labels = [label] * self.batch_size

        if self.dataset_flag == HistDataset.TEST_SET:
            t_imgs = tf.convert_to_tensor(imgs)
            return t_imgs

        t_imgs = tf.convert_to_tensor(imgs)
        t_labels = tf.convert_to_tensor(labels)

        return t_imgs, t_labels



