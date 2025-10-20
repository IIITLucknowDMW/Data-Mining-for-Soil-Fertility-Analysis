import numpy as np
import random


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split dataset into training and testing sets
    
    Parameters:
    - X: features
    - y: labels
    - test_size: proportion of test data (default 0.2)
    - random_state: random seed for reproducibility
    """
    if random_state is not None:
        random.seed(random_state)

    data = list(zip(X, y))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))

    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    return X_train, X_test, y_train, y_test


def load_dataset(file_path):
    """
    Load dataset from CSV file
    
    Parameters:
    - file_path: path to CSV file
    
    Returns:
    - dataset: numpy array of data
    - header: column names
    """
    dataset = np.genfromtxt(file_path, delimiter=',', dtype=float, skip_header=1)
    header = np.genfromtxt(file_path, delimiter=',', dtype=str, max_rows=1)
    return dataset, header
