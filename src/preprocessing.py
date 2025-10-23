import numpy as np
import copy as cp
import re
import statistics
from sklearn.linear_model import LinearRegression
import math


def calculate_median(attribute, dataset):
    dataset_clean = np.delete(dataset[:, attribute], missing_values(attribute, dataset))
    sorted_list = cp.deepcopy(dataset_clean)
    sorted_list.sort()
    if sorted_list.size % 2 != 0:
        median = sorted_list[((sorted_list.size + 1) // 2) - 1]
    else:
        median = (sorted_list[(sorted_list.size // 2) - 1] + sorted_list[sorted_list.size // 2]) / 2
    return median


def missing_values(attribute, dataset):
    missing_list = []
    for i in range(0, len(dataset[:, attribute])):
        if not re.fullmatch(r"\d+\.(:?\d+)?", str(dataset[i, attribute])):
            missing_list.append(i)
    return missing_list


def central_tendency_homemade(attribute, dataset):
    dataset_clean = np.delete(dataset[:, attribute], missing_values(attribute, dataset))
    mean = dataset_clean.sum() / dataset_clean.shape[0]
    median = calculate_median(attribute, dataset)
    unique_values, counts = np.unique(dataset_clean, return_counts=True)
    max_index = np.where(counts == max(counts))[0]
    mode = [unique_values[i] for i in max_index]
    return [mean, median, mode]


def quartiles_homemade(attribute, dataset):
    dataset_clean = np.delete(dataset[:, attribute], missing_values(attribute, dataset))
    sorted_list = cp.deepcopy(dataset_clean)
    sorted_list.sort()
    q0 = sorted_list[0]
    q1 = (sorted_list[sorted_list.size // 4 - 1] + sorted_list[sorted_list.size // 4]) / 2
    q3 = (sorted_list[sorted_list.size * 3 // 4 - 1] + sorted_list[sorted_list.size * 3 // 4]) / 2
    q2 = calculate_median(attribute, dataset)
    q4 = sorted_list[-1]
    return [q0, q1, q2, q3, q4]


def std_dev_homemade(attribute, dataset):
    dataset_clean = np.delete(dataset[:, attribute], missing_values(attribute, dataset))
    mean = np.mean(dataset_clean)
    deviations = [(val - mean) ** 2 for val in dataset_clean]
    variance = np.mean(deviations)
    return np.sqrt(variance)


def discretization(attribute, dataset):
    vals = dataset[:, attribute].copy()
    vals.sort()
    q = 1 + (10 / 3) * np.log10(dataset.shape[0])
    num_elements = math.ceil(dataset[:, attribute].shape[0] / q)
    
    for val in range(0, dataset[:, attribute].shape[0]):
        for i in range(0, vals.shape[0], num_elements):
            if(vals[i] > dataset[val, attribute]):
                upper_bound = i
                break
        dataset[val, attribute] = np.median(vals[upper_bound - num_elements:upper_bound])
    return dataset


def replace_missing_values(method, attribute, dataset):
    missing = missing_values(attribute, dataset)
    for i in missing:
        if method == 0:
            dataset[i, attribute] = statistics.mode(dataset[:, attribute])
        else:
            dataset[i, attribute] = np.mean([dataset[j, attribute] for j in range(0, len(dataset)) 
                                              if dataset[j, -1] == dataset[i, -1] and not j in missing])
    return dataset


def replace_outliers(method, attribute, dataset):
    outliers = []
    if method == 0:  # Linear Regression
        IQR = (quartiles_homemade(attribute, dataset)[-2] - quartiles_homemade(attribute, dataset)[1]) * 1.5
        for i in range(0, len(dataset[:, attribute])):
            if (dataset[i, attribute] > (quartiles_homemade(attribute, dataset)[-2] + IQR) or 
                dataset[i, attribute] < (quartiles_homemade(attribute, dataset)[1] - IQR)):
                outliers.append(i)
        
        X = np.delete(dataset, attribute, axis=1)
        X = np.delete(X, outliers, axis=0)
        y = dataset[:, attribute]
        y = np.delete(y, outliers, axis=0).reshape(-1, 1)

        model = LinearRegression().fit(X, y)
        
        for i in outliers:
            x2 = np.delete(dataset, attribute, axis=1)
            X_new = x2[i, :].T.reshape(1, -1)
            dataset[i, attribute] = model.predict(X_new)[0][0]
    else:  # Discretization
        dataset = discretization(attribute, dataset)
    return dataset


def replace_missing_general(method, dataset):
    for i in range(0, dataset.shape[1] - 1):
        dataset = replace_missing_values(method, i, dataset)
    return dataset


def replace_outliers_general(method, dataset):
    for i in range(0, dataset.shape[1] - 1):
        dataset = replace_outliers(method, i, dataset)
    return dataset


def remove_duplicates(dataset):
    print(f"Before: {len(dataset)} rows")
    dataset = np.unique(dataset, axis=0, return_index=False)
    print(f"After: {len(dataset)} rows")
    return dataset


def correlation_coef(attribute1, attribute2, dataset):
    mean1 = np.mean(dataset[:, attribute1])
    mean2 = np.mean(dataset[:, attribute2])
    std1 = std_dev_homemade(attribute1, dataset)
    std2 = std_dev_homemade(attribute2, dataset)
    return (dataset[:, attribute1].dot(dataset[:, attribute2]) - (len(dataset) * mean1 * mean2)) / ((len(dataset) - 1) * (std1 * std2))


def reduce_dimensions(threshold, dataset):
    to_delete = []
    for i in range(0, dataset.shape[1] - 1):
        for j in range(i + 1, dataset.shape[1]):
            if (np.abs(correlation_coef(i, j, dataset)) > threshold):
                print(i, j)
                to_delete.append(i)
    dataset = np.delete(dataset, to_delete, axis=1)
    return dataset


def normalize(method, attribute, dataset):
    if method:  # Min-Max normalization
        vmin = 0
        vmax = 1
        vmin_old = dataset[:, attribute].min()
        vmax_old = dataset[:, attribute].max()
        for val in range(0, dataset[:, attribute].shape[0]):
            dataset[val, attribute] = vmin + (vmax - vmin) * ((dataset[val, attribute] - vmin_old) / (vmax_old - vmin_old))
    else:  # Z-score normalization
        vmean = np.mean(dataset[:, attribute])
        s = np.mean((dataset[:, attribute] - vmean) ** 2)
        for val in range(0, dataset[:, attribute].shape[0]):
            dataset[val, attribute] = (dataset[val, attribute] - vmean) / s
    return dataset


def normalize_general(method, dataset):
    for i in range(0, dataset.shape[1] - 1):
        dataset = normalize(method, i, dataset)
    return dataset


def preprocess_dataset(dataset, missing_method=1, outlier_method=0, correlation_threshold=0.75, normalize_method=1):
    """
    Complete preprocessing pipeline
    
    Parameters:
    - missing_method: 0 for mode, 1 for mean
    - outlier_method: 0 for Linear Regression, 1 for Discretization
    - correlation_threshold: threshold for dimension reduction
    - normalize_method: 0 for Z-score, 1 for Min-Max
    """
    dataset = replace_missing_general(missing_method, dataset)
    dataset = replace_outliers_general(outlier_method, dataset)
    dataset = remove_duplicates(dataset)
    dataset = reduce_dimensions(correlation_threshold, dataset)
    dataset = normalize_general(normalize_method, dataset)
    return dataset
