import pathlib
import tensorflow as tf
import pandas as pd

from tqdm import tqdm

def preprocess_img(image):
    '''Resize and normalize a given image.
    
    :param image: Image
    :type image: Raw Tensor Data
    :return: Resized immage
    :rtype: Tensor
    '''
    image = tf.image.decode_jpeg(image)    
    image = tf.image.resize(image, [32, 32])
    image /= 255.0
    return image

def get_file_path_labels(id_label_df, train_dir):
    '''Get list of file paths and corresponding labels
    
    :param id_label_df: Dataframe with id to label mapping
    :type id_label_df: Pandas Dataframe
    :param train_dir: Training directory
    :type train_dir: String
    :return: Paths to images, labels
    :rtype: 2 lists
    '''
    train_dir = pathlib.Path(train_dir)
    img_label = []
    img_path = []
    for i in tqmd(train_dir.glob('*.tif')):
        img_id = i.stem
        label = id_label_df[id_label_df[id] == img_id]['label'].values[0]
        img_label.append(label)
        img_path.append(str(i.resolve()))
    return img_path, img_label

def build_dataset(img_paths, labels):
    '''Build dataset given paths and labels
    
    :param img_paths: Path to all images
    :type img_paths: List of string paths
    :param labels: Labels of all images
    :type labels: List of integer labels
    :return: Dataset
    :rtype: tf.data.Dataset
    '''
    image_ds = tf.data.Dataset.from_tensor_slices(all_img_path)
    image_ds = image_ds.map(preprocess_img)
    image_labels = tf.data.Dataset.from_tensor_slices(all_img_path)
    dataset = tf.data.Dataset.zip((image_ds, image_labels))
    return dataset

def get_dataset(csv_path, train_dir):
    '''Build dataset from given csv labels and path to image dir
    
    :param csv_path: Path to labels csv file
    :type csv_path: String
    :param train_dir: Path to training directory
    :type train_dir: String
    :return: Dataset
    :rtype: tf.data.Dataset
    '''
    img_label_df = pd.read_csv(csv_path)
    img_path, img_label = get_file_path_labels(img_label_df, train_dir)
    dataset = build_dataset(img_path, img_label)
    return dataset

def build_multiple_tf_records(csv_path, train_dir):
    '''Build multiple tf record files
    
    :param csv_path: Path to labels csv
    :type csv_path: String
    :param train_dir: Training directory
    :type train_dir: String
    '''
    img_label_df = pd.read_csv(csv_path)
    img_path, img_label = get_file_path_labels(img_label_df, train_dir)
    idx = 0
    base_file = 'hist_rec_'
    suffix = '.tfrec'
    count = 1
    while True:
        tfrec_file_name = base_file + str(count) + suffix
        if idx >= len(all_img_path):
            break
        with tf.compat.v1.python_io.TFRecordWriter(tfrec_file_name) as writer:
            while idx < len(all_img_path):
                path = all_img_path[idx]
                label = all_img_labels[idx]
                example = get_img_example(path, label)
                writer.write(example.SerializeToString())
                idx += 1
                if idx > 0 and idx % 10000 == 0:
                    count += 1
                    break
    