import numpy as np
import pickle
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load raw data from file."""
    datasetFaces = []

    for person in range(1, 41):
        temp = []

        for pose in range(1, 11):
            data = plt.imread(file_path + f's{str(person)}/{str(pose)}.pgm')
            temp.append(data)

        datasetFaces.append(np.array(temp))

    return np.array(datasetFaces)


def preprocess_data(data):
    """Preprocess the raw data."""

    mean = np.zeros(data.shape)
    mean[:, :] = np.mean(data, axis=0)

    zero_mean = data - mean

    return zero_mean


def split_data(data):
    """Split the data into train and test sets."""
    train_size = 0.9  # Split Data with random sampling

    train_idx = np.random.choice(10, int(train_size*10), replace=False)

    X_train = []
    X_test = []
    for person in range(data.shape[0]):
        for poses in range(data.shape[1]):
            if poses in train_idx:
                X_train.append(data[person][poses].flatten())
            else:
                X_test.append(data[person][poses].flatten())

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array([i for i in range(40) for j in range(9)])
    y_test = np.array([i for i in range(40) for j in range(1)])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # Load raw data
    data = load_data(os.getcwd() + '/data/orl/raw/')

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # Save split data
    pickle.dump(X_train, open(os.getcwd() +
                              '/data/orl/split/X_train.pkl', 'wb'))
    pickle.dump(X_test, open(os.getcwd() +
                             '/data/orl/split/X_test.pkl', 'wb'))
    pickle.dump(y_train, open(os.getcwd() +
                              '/data/orl/split/y_train.pkl', 'wb'))
    pickle.dump(y_test, open(os.getcwd() +
                             '/data/orl/split/y_test.pkl', 'wb'))

    # Preprocess data
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    # Save preprocessed data
    pickle.dump(X_train, open(os.getcwd() +
                              '/data/orl/processed/X_train.pkl', 'wb'))
    pickle.dump(X_test, open(os.getcwd() +
                             '/data/orl/processed/X_test.pkl', 'wb'))
    pickle.dump(y_train, open(os.getcwd() +
                              '/data/orl/processed/y_train.pkl', 'wb'))
    pickle.dump(y_test, open(os.getcwd() +
                             '/data/orl/processed/y_test.pkl', 'wb'))

    print('Data saved in folder: ', '/data/orl/processed/')
