import pathlib
import Data
import Dataset
import Models
import pandas as pd
import numpy as np
from PIL import Image

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
    dataset = Dataset.HistDataset(df, Data.train_dir, Dataset.HistDataset.TRAIN_SET)
    model = Models.get_base_model()


