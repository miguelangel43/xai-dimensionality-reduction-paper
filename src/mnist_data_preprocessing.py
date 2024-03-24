import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import struct
from array import array


class MnistDataloader(object):
    """MNIST Data Loader Class"""

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        """
        Initialize the data loader with file paths.

        Args:
            training_images_filepath (str): File path for training images.
            training_labels_filepath (str): File path for training labels.
            test_images_filepath (str): File path for test images.
            test_labels_filepath (str): File path for test labels.
        """
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Read images and labels from binary files.

        Args:
            images_filepath (str): File path for images.
            labels_filepath (str): File path for labels.

        Returns:
            tuple: Tuple containing images and labels.
        """
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    'Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    'Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        """
        Load training and test data.

        Returns:
            tuple: Tuple containing training and test data.
        """
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


def sample_data(X_train, y_train, X_test, y_test, n_samples):
    """
    Randomly sample data from the dataset.

    Args:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Testing data.
        y_test (array-like): Testing labels.
        n_samples (tuple): Number of samples to select for training and testing data.

    Returns:
        tuple: Tuple containing sampled training and testing data.
    """
    # Randomly select 600 instances for training data
    train_indices = np.random.choice(
        len(X_train), size=n_samples[0], replace=False)
    X_train = np.array(X_train)[train_indices]
    y_train = np.array(y_train)[train_indices]

    # Randomly select 100 instances for testing data
    test_indices = np.random.choice(
        len(X_test), size=n_samples[1], replace=False)
    X_test = np.array(X_test)[test_indices]
    y_test = np.array(y_test)[test_indices]

    return X_train, y_train, X_test, y_test


def show_images(images, title_texts):
    """
    Helper function to display images.

    Args:
        images (list): List of images.
        title_texts (list): List of corresponding title texts.
    """
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15)
        index += 1


if __name__ == "__main__":

    # File paths
    training_images_filepath = os.getcwd() + '/data/mnist/raw/train-images-idx3-ubyte'
    training_labels_filepath = os.getcwd() + '/data/mnist/raw/train-labels-idx1-ubyte'
    test_images_filepath = os.getcwd() + '/data/mnist/raw/t10k-images-idx3-ubyte'
    test_labels_filepath = os.getcwd() + '/data/mnist/raw/t10k-labels-idx1-ubyte'

    # Initialize data loader
    mnist_dataloader = MnistDataloader(
        training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    # Load and split data
    (X_train, y_train), (X_test, y_test) = mnist_dataloader.load_data()

    # Sample data
    X_train, y_train, X_test, y_test = sample_data(
        X_train, y_train, X_test, y_test, n_samples=(600, 100))

    # Save split data
    pickle.dump(X_train, open(os.getcwd() +
                              '/data/mnist/split/X_train.pkl', 'wb'))
    pickle.dump(X_test, open(os.getcwd() +
                             '/data/mnist/split/X_test.pkl', 'wb'))
    pickle.dump(y_train, open(os.getcwd() +
                              '/data/mnist/split/y_train.pkl', 'wb'))
    pickle.dump(y_test, open(os.getcwd() +
                             '/data/mnist/split/y_test.pkl', 'wb'))

    # Normalize the pixel values to be between 0 and 1
    X_train = np.array(X_train) / 255.0
    X_test = np.array(X_test) / 255.0

    # Subtract the mean from the data
    mean = np.mean(X_train, axis=0)
    X_train = X_train - mean
    X_test = X_test - mean

    # Flatten the images
    X_train = np.array(X_train).reshape((len(X_train), -1))
    X_test = np.array(X_test).reshape((len(X_test), -1))

    os.path.dirname(os.path.dirname(os.path.abspath(
                    __file__)))
    # Save preprocessed data
    pickle.dump(X_train, open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
                              '/data/mnist/processed/X_train.pkl', 'wb'))
    pickle.dump(X_test, open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
                             '/data/mnist/processed/X_test.pkl', 'wb'))
    pickle.dump(y_train, open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
                              '/data/mnist/processed/y_train.pkl', 'wb'))
    pickle.dump(y_test, open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
                             '/data/mnist/processed/y_test.pkl', 'wb'))

    print('Data saved in folder: ', '/data/mnist/processed/')
