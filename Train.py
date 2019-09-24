import pathlib
import Data
import Dataset
import datetime
import Models
import pandas as pd
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split

def get_stats(img_dir):
    p = pathlib.Path(img_dir)
    avg_dim = [-1, -1, -1]
    max_dim = [-1, -1, -1]
    min_dim = [-1, -1, -1]
    for idx, path in enumerate(p.glob('*.tif')):
        img = Image.open(path)
        img = np.array(img)
        dim = img.shape
        idx += 1
        avg_dim[0] = (avg_dim[0] * (idx - 1) + dim[0]) / idx
        avg_dim[1] = (avg_dim[1] * (idx - 1) + dim[1]) / idx
        avg_dim[2] = (avg_dim[2] * (idx - 1) + dim[2]) / idx

        max_dim[0] = dim[0] if dim[0] > max_dim[0] else max_dim[0]
        max_dim[1] = dim[1] if dim[1] > max_dim[1] else max_dim[1]
        max_dim[2] = dim[2] if dim[2] > max_dim[2] else max_dim[2]

        min_dim[0] = dim[0] if dim[0] > min_dim[0] else min_dim[0]
        min_dim[1] = dim[1] if dim[1] > min_dim[1] else min_dim[1]
        min_dim[2] = dim[2] if dim[2] > min_dim[2] else min_dim[2]

    print(avg_dim)
    print(max_dim)
    print(min_dim)

if __name__ == '__main__':
    df = pd.read_csv(Data.train_csv)
    train_df, val_df = train_test_split(df, test_size=0.18)
    train_dataset = Dataset.HistDataset(train_df, Data.train_dir, Dataset.HistDataset.TRAIN_SET)
    val_dataset = Dataset.HistDataset(train_df, Data.train_dir, Dataset.HistDataset.VAL_SET)
    model = Models.get_base_model((96, 96, 3))

    loss_obj = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    accuracy = tf.keras.metrics.Accuracy()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                  patience=3)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=[accuracy]
    )

    model.fit_generator(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=[early_stopping, reduce_lr, tensorboard_callback]
    )

    model.save('base_model.h5')

