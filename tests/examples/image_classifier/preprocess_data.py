import numpy as np
import os
import random
import shutil
import struct

from tests.examples.image_classifier.constants import (
    MNIST_IMAGES_PATH,
    DATA_PATH,
    SPLIT_DATA_PATH,
    VALID_DATA_SPLIT_PARTITIONS,
)
from tests.examples.image_classifier.user_script import create_data_loader


def read_idx3_file(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number
        magic_number = struct.unpack('>I', f.read(4))[0]

        # Read the number of images
        num_images = struct.unpack('>I', f.read(4))[0]

        # Read the number of rows and columns in each image
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)

    # Reshape the image data to a 3D array (num_images, num_rows, num_cols)
    image_data = image_data.reshape(num_images, num_rows, num_cols)

    return image_data


def read_idx1_file(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number
        magic_number = struct.unpack('>I', f.read(4))[0]

        # Read the number of items (labels)
        num_items = struct.unpack('>I', f.read(4))[0]

        # Read label data
        label_data = np.frombuffer(f.read(), dtype=np.uint8)

    return label_data


def save_to_idx3(data, file_path):
    with open(file_path, 'wb') as f:
        # Write magic number, number of images, rows, and columns
        f.write(struct.pack('>I', 2051))  # Magic number for IDX3
        f.write(struct.pack('>I', len(data)))
        f.write(struct.pack('>I', data.shape[1]))
        f.write(struct.pack('>I', data.shape[2]))

        # Write image data
        f.write(data.tobytes())


def save_to_idx1(data, file_path):
    with open(file_path, 'wb') as f:
        # Write magic number and number of items (labels)
        f.write(struct.pack('>I', 2049))  # Magic number for IDX1
        f.write(struct.pack('>I', len(data)))

        # Write label data
        f.write(data.tobytes())


def get_output_data_path(partition_num):
    return SPLIT_DATA_PATH / f"{partition_num}/MNIST/raw"


def split_and_save_data(split_into=2, random_seed=42):
    if split_into not in VALID_DATA_SPLIT_PARTITIONS:
        raise Exception(f"Can only split MNIST dataset into {VALID_DATA_SPLIT_PARTITIONS} partitions.")

    shutil.rmtree(SPLIT_DATA_PATH, ignore_errors=True)

    for i in range(split_into):
        os.makedirs(get_output_data_path(i + 1), exist_ok=True)

    # Set random seed for reproducibility
    random.seed(random_seed)

    if not os.path.exists(MNIST_IMAGES_PATH):
        create_data_loader(DATA_PATH)

    for prefix in ("train", "t10k"):
        images = read_idx3_file(MNIST_IMAGES_PATH / f'{prefix}-images-idx3-ubyte',)
        labels = read_idx1_file(MNIST_IMAGES_PATH / f'{prefix}-labels-idx1-ubyte',)

        data_pairs = list(zip(images, labels))
        random.shuffle(data_pairs)
        partition_length = len(data_pairs) // split_into

        for i in range(split_into):
            data_pairs_partition = data_pairs[i*partition_length:(i+1)*partition_length]
            images_partition, labels_partition = zip(*data_pairs_partition)
            images_partition = np.array(images_partition)
            labels_partition = np.array(labels_partition)
            save_to_idx3(images_partition, get_output_data_path(i + 1) / f'{prefix}-images-idx3-ubyte')
            save_to_idx1(labels_partition, get_output_data_path(i + 1) / f'{prefix}-labels-idx1-ubyte')


if __name__ == "__main__":
    split_and_save_data(split_into=4, random_seed=42)
