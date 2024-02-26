import struct
import numpy as np
import random


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


def split_and_save_data(
        idx3_path, idx1_path, output_part1_idx3, output_part1_idx1, output_part2_idx3, output_part2_idx1,
        split_ratio=0.5, random_seed=42
):
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Read IDX3 and IDX1 files
    images = read_idx3_file(idx3_path)
    labels = read_idx1_file(idx1_path)

    # Combine images and labels into a list of tuples
    data_pairs = list(zip(images, labels))

    # Shuffle the data
    random.shuffle(data_pairs)

    # Calculate the split index
    split_index = int(len(data_pairs) * split_ratio)

    # Split data and labels
    train_data, val_data = data_pairs[:split_index], data_pairs[split_index:]
    train_images, train_labels = zip(*train_data)
    val_images, val_labels = zip(*val_data)

    # Convert back to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    # Save training set to new IDX files
    save_to_idx3(train_images, output_part1_idx3)
    save_to_idx1(train_labels, output_part1_idx1)

    # Save validation set to new IDX files
    save_to_idx3(val_images, output_part2_idx3)
    save_to_idx1(val_labels, output_part2_idx1)


if __name__ == "__main__":
    from simulate_vulkan.data_manipulation.constants import MNIST_IMAGES_PATH, SPLIT_DATA_PATH

    INPUT_DATA_PATH = MNIST_IMAGES_PATH
    OUTPUT_DATA_PATH = SPLIT_DATA_PATH

    idx3_path = INPUT_DATA_PATH + '/train-images-idx3-ubyte'
    idx1_path = INPUT_DATA_PATH + '/train-labels-idx1-ubyte'
    output_part1_idx3 = OUTPUT_DATA_PATH + '/part1/train-images-idx3-ubyte'
    output_part1_idx1 = OUTPUT_DATA_PATH + '/part1/train-labels-idx1-ubyte'
    output_part2_idx3 = OUTPUT_DATA_PATH + '/part2/train-images-idx3-ubyte'
    output_part2_idx1 = OUTPUT_DATA_PATH + '/part2/train-labels-idx1-ubyte'

    split_and_save_data(
        idx3_path,
        idx1_path,
        output_part1_idx3,
        output_part1_idx1,
        output_part2_idx3,
        output_part2_idx1,
        split_ratio=0.5,
        random_seed=42
    )
