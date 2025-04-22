#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install imblearn


# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, zscore
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
import keras
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, SimpleRNN, LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/1_Raw/"
label_dir = "/Users/ankithreddy/Downloads/SWANpre/2_Labels/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
labels = []

num_partitions = 5

# Load raw data
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + ".pkl", 'rb') as f:
        raw_data.append(pickle.load(f))

# Load labels
for i in range(num_partitions):
    labels.append(pd.read_csv(label_dir + "partition" + str(i+1) + "_labels.csv"))

# Processing data and labels
for i in range(num_partitions):
    # Transpose data to (num_samples, num_timestamps, num_features) and remove the first feature
    data = np.transpose(raw_data[i], (2, 0, 1))[:, :, 2:8]  # shape (num_samples, num_timestamps, 24 features)
    
    # Process FLARE_CLASS labels
    class_mapping = {'X': 5, 'M': 4, 'B': 3, 'C': 2, 'FQ': 1}
    flare_class_labels = labels[i]['FLARE_CLASS'].map(class_mapping).values

    # Process FLARE_TYPE labels
    type_mapping = {'FQ': 1, 'C': 10, 'B': 100, 'M': 1000, 'X': 10000}
    
    def calculate_flare_type(flare_type):
        if flare_type == 'FQ':
            return 1
        else:
            return type_mapping[flare_type[0]] * float(flare_type[1:])
    
    flare_type_labels = labels[i]['FLARE_TYPE'].apply(calculate_flare_type).values
    
    # Mean imputation and removal of invalid samples
    valid_samples = []
    for sample_idx in range(data.shape[0]):
        valid_sample = True
        for feature_idx in range(data.shape[2]):
            feature_data = data[sample_idx, :, feature_idx]
            n = len(feature_data)
            valid_values = feature_data[(feature_data != 0) & (~np.isnan(feature_data))]
            if len(valid_values) > 0:
                for t in range(n):
                    next_value_found = False
                    if feature_data[t] == 0 or np.isnan(feature_data[t]):
                        # Try to find the next available value
                        for j in range(t + 1, n):
                            if feature_data[j] != 0 and not np.isnan(feature_data[j]):
                                feature_data[t] = feature_data[j]
                                next_value_found = True
                                break
                    # If no next value is found, use the previous value
                    if not next_value_found:
                        for j in range(t - 1, -1, -1):
                            if feature_data[j] != 0 and not np.isnan(feature_data[j]):
                                feature_data[t] = feature_data[j]
                                break
    
            else:
                valid_sample = False
                break  # Exit the loop if the sample is invalid
            data[sample_idx, :, feature_idx] = feature_data
        if valid_sample:
            valid_samples.append(sample_idx)
    
    data = data[valid_samples]
    flare_class_labels = flare_class_labels[valid_samples]
    flare_type_labels = flare_type_labels[valid_samples]
    print(data.shape)
    
    unique, counts = np.unique(flare_class_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Partition {i+1} flare_class_labels count: {class_counts}")
    
    # Check for any NaN values before saving
    has_nan = np.isnan(data).any()
    print(f"Partition {i+1} has NaN values: {has_nan}")
    
    with open(processed_data_dir + "partition" + str(i+1) + "_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_flare_class_labels.pkl", 'wb') as f:
        pickle.dump(flare_class_labels, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_flare_type_labels.pkl", 'wb') as f:
        pickle.dump(flare_type_labels, f)


# In[4]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/1_Raw/"
label_dir = "/Users/ankithreddy/Downloads/SWANpre/2_Labels/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
labels = []

num_partitions = 5

# Load raw data
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + ".pkl", 'rb') as f:
        raw_data.append(pickle.load(f))

# Load labels
for i in range(num_partitions):
    labels.append(pd.read_csv(label_dir + "partition" + str(i+1) + "_labels.csv"))

# Processing data and labels
for i in range(num_partitions):
    # Transpose data to (num_samples, num_timestamps, num_features) and remove the first feature
    data = np.transpose(raw_data[i], (2, 0, 1))[:, :, 1:]  # shape (num_samples, num_timestamps, 24 features)
    
    # Process FLARE_CLASS labels
    class_mapping = {'X': 5, 'M': 4, 'B': 3, 'C': 2, 'FQ': 1}
    flare_class_labels = labels[i]['FLARE_CLASS'].map(class_mapping).values

    # Process FLARE_TYPE labels
    type_mapping = {'FQ': 1, 'C': 10, 'B': 100, 'M': 1000, 'X': 10000}
    
    def calculate_flare_type(flare_type):
        if flare_type == 'FQ':
            return 1
        else:
            return type_mapping[flare_type[0]] * float(flare_type[1:])
    
    flare_type_labels = labels[i]['FLARE_TYPE'].apply(calculate_flare_type).values
    
    # Mean imputation and removal of invalid samples
    valid_samples = []
    for sample_idx in range(data.shape[0]):
        valid_sample = True
        for feature_idx in range(data.shape[2]):
            feature_data = data[sample_idx, :, feature_idx]
            n = len(feature_data)
            valid_values = feature_data[(feature_data != 0) & (~np.isnan(feature_data))]
            if len(valid_values) > 0:
                for t in range(n):
                    next_value_found = False
                    if feature_data[t] == 0 or np.isnan(feature_data[t]):
                        # Try to find the next available value
                        for j in range(t + 1, n):
                            if feature_data[j] != 0 and not np.isnan(feature_data[j]):
                                feature_data[t] = feature_data[j]
                                next_value_found = True
                                break
                    # If no next value is found, use the previous value
                    if not next_value_found:
                        for j in range(t - 1, -1, -1):
                            if feature_data[j] != 0 and not np.isnan(feature_data[j]):
                                feature_data[t] = feature_data[j]
                                break
    
            else:
                valid_sample = False
                break  # Exit the loop if the sample is invalid
            data[sample_idx, :, feature_idx] = feature_data
        if valid_sample:
            valid_samples.append(sample_idx)
    
    data = data[valid_samples]
    flare_class_labels = flare_class_labels[valid_samples]
    flare_type_labels = flare_type_labels[valid_samples]
    print(data.shape)
    
    unique, counts = np.unique(flare_class_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Partition {i+1} flare_class_labels count: {class_counts}")
    
    # Check for any NaN values before saving
    has_nan = np.isnan(data).any()
    print(f"Partition {i+1} has NaN values: {has_nan}")
    
    with open(processed_data_dir + "partition" + str(i+1) + "_WFS_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_WFS_flare_class_labels.pkl", 'wb') as f:
        pickle.dump(flare_class_labels, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_WFS_flare_type_labels.pkl", 'wb') as f:
        pickle.dump(flare_type_labels, f)


# In[5]:


# NORMALIZATION


# In[7]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
raw_labels = []

num_partitions = 5

# Load processed data and labels
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_data.pkl", 'rb') as f:
        raw_data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_flare_class_labels.pkl", 'rb') as f:
        raw_labels.append(pickle.load(f))

# Function to apply normalization based on skewness
def normalize_feature(feature_data):
    feature_data_flat = feature_data.flatten()
    feature_data_normalized = zscore(feature_data_flat)
    
    return feature_data_normalized.reshape(feature_data.shape)

# Normalize data and convert labels to binary
for i in range(num_partitions):
    data = raw_data[i]
    labels = raw_labels[i]
    
    num_samples, num_timestamps, num_features = data.shape

    normalized_data = np.empty_like(data)

    for feature_idx in range(num_features):
        feature_data = data[:, :, feature_idx]
        normalized_feature_data = normalize_feature(feature_data)
        normalized_data[:, :, feature_idx] = normalized_feature_data

    # Check for any NaN values before saving
    has_nan = np.isnan(normalized_data).any()
    print(f"Partition {i+1} has NaN values: {has_nan}")

    # Convert labels to binary
    binary_labels = np.where(np.isin(labels, [4, 5]), 1, 0)

    # Save normalized data
    with open(processed_data_dir + "partition" + str(i+1) + "_normalized_data.pkl", 'wb') as f:
        pickle.dump(normalized_data, f)
    
    # Save binary labels
    with open(processed_data_dir + "partition" + str(i+1) + "_binary_labels.pkl", 'wb') as f:
        pickle.dump(binary_labels, f)

    print(f"Partition {i+1} normalized data shape: {normalized_data.shape}")
    print(f"Partition {i+1} binary labels distribution: {np.bincount(binary_labels)}")


# In[8]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
raw_labels = []

num_partitions = 5

# Load processed data and labels
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_WFS_data.pkl", 'rb') as f:
        raw_data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_WFS_flare_class_labels.pkl", 'rb') as f:
        raw_labels.append(pickle.load(f))

# Function to apply normalization based on skewness
def normalize_feature(feature_data):
    feature_data_flat = feature_data.flatten()
    feature_data_normalized = zscore(feature_data_flat)
    
    return feature_data_normalized.reshape(feature_data.shape)

# Normalize data and convert labels to binary
for i in range(num_partitions):
    data = raw_data[i]
    labels = raw_labels[i]
    
    num_samples, num_timestamps, num_features = data.shape

    normalized_data = np.empty_like(data)

    for feature_idx in range(num_features):
        feature_data = data[:, :, feature_idx]
        normalized_feature_data = normalize_feature(feature_data)
        normalized_data[:, :, feature_idx] = normalized_feature_data

    # Check for any NaN values before saving
    has_nan = np.isnan(normalized_data).any()
    print(f"Partition {i+1} has NaN values: {has_nan}")

    # Convert labels to binary
    binary_labels = np.where(np.isin(labels, [4, 5]), 1, 0)

    # Save normalized data
    with open(processed_data_dir + "partition" + str(i+1) + "_WFS_normalized_data.pkl", 'wb') as f:
        pickle.dump(normalized_data, f)
    
    # Save binary labels
    with open(processed_data_dir + "partition" + str(i+1) + "_WFS_binary_labels.pkl", 'wb') as f:
        pickle.dump(binary_labels, f)

    print(f"Partition {i+1} normalized data shape: {normalized_data.shape}")
    print(f"Partition {i+1} binary labels distribution: {np.bincount(binary_labels)}")


# In[9]:


# SMOTE Over Sampling


# In[10]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
labels = []

num_partitions = 5

# Load normalized data and labels
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_WFS_normalized_data.pkl", 'rb') as f:
        raw_data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_WFS_flare_class_labels.pkl", 'rb') as f:
        labels.append(pickle.load(f))

# Convert classes for binary classification and apply SMOTE
for i in range(num_partitions):
    data = raw_data[i]
    flare_class_labels = labels[i]
    
    # Convert classes
    binary_labels = np.where(flare_class_labels >= 4, 1, 0)
    
    # Reshape data to (num_samples, num_timestamps * num_features) for SMOTE
    num_samples, num_timestamps, num_features = data.shape
    reshaped_data = data.reshape((num_samples, num_timestamps * num_features))
    
    # Apply SMOTE
    smote = SMOTE()
    reshaped_data_smote, binary_labels_smote = smote.fit_resample(reshaped_data, binary_labels)
    
    # Reshape data back to (num_samples, num_timestamps, num_features)
    new_data = reshaped_data_smote.reshape((-1, num_timestamps, num_features))
    
    # Save new data and labels
    with open(processed_data_dir + "partition" + str(i+1) + "_smote_data.pkl", 'wb') as f:
        pickle.dump(new_data, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_smote_labels.pkl", 'wb') as f:
        pickle.dump(binary_labels_smote, f)
    
    print(f"Partition {i+1} new data shape: {new_data.shape}")
    print(f"Partition {i+1} new label distribution: {np.bincount(binary_labels_smote)}")


# In[11]:


# Balanced Sampling


# In[12]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
labels = []
flare_type_labels_list = []

num_partitions = 5

# Load processed data
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_WFS_normalized_data.pkl", 'rb') as f:
        raw_data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_WFS_flare_class_labels.pkl", 'rb') as f:
        labels.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_WFS_flare_type_labels.pkl", 'rb') as f:
        flare_type_labels_list.append(pickle.load(f))

# Gaussian Noise Injection
def gaussian_noise_injection(data, num_samples, noise_proportion=0.1):
    std_dev = np.std(data, axis=0)
    noise_level = std_dev * noise_proportion

    new_samples = []
    for _ in range(num_samples):
        sample_index = np.random.choice(len(data))
        sample = data[sample_index]
        noise = np.random.normal(0, noise_level, sample.shape)
        new_sample = sample + noise
        new_samples.append(new_sample)

    return np.array(new_samples)


# Process each partition
for i in range(num_partitions):
    data = raw_data[i]
    flare_class_labels = labels[i]
    flare_type_labels = flare_type_labels_list[i]

    # Oversampling
    augmented_data = []
    augmented_class_labels = []
    augmented_type_labels = []

    for class_label, factor in [(5, 10), (4, 1.5)]:
        class_indices = np.where(flare_class_labels == class_label)[0]
        class_data = data[class_indices]
        class_type_labels = flare_type_labels[class_indices]
        num_samples = int(len(class_indices) * factor)

        # Gaussian Noise Injection
        gni_data = gaussian_noise_injection(class_data, num_samples)
        gni_labels = np.full(num_samples, class_label)
        gni_type_labels = np.random.choice(class_type_labels, num_samples, replace=True)
        augmented_data.append(gni_data)
        augmented_class_labels.append(gni_labels)
        augmented_type_labels.append(gni_type_labels)

    # Combine original and augmented data
    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_class_labels = np.concatenate(augmented_class_labels, axis=0)
    augmented_type_labels = np.concatenate(augmented_type_labels, axis=0)
    data = np.concatenate((data, augmented_data), axis=0)
    flare_class_labels = np.concatenate((flare_class_labels, augmented_class_labels), axis=0)
    flare_type_labels = np.concatenate((flare_type_labels, augmented_type_labels), axis=0)

    # Calculate target number of samples for minority classes
    total_majority_class_samples = len(flare_class_labels[flare_class_labels == 5]) + len(flare_class_labels[flare_class_labels == 4])
    keep_1 = int(total_majority_class_samples * 1.2)
    keep_2_3 = int(total_majority_class_samples // 2)

    # Undersample minority classes to match the target number of samples
    minority_class_1_indices = np.where(flare_class_labels == 1)[0]
    minority_class_2_indices = np.where(flare_class_labels == 2)[0]
    minority_class_3_indices = np.where(flare_class_labels == 3)[0]

    minority_class_1_samples_to_keep = np.random.choice(minority_class_1_indices, min(keep_1, len(minority_class_1_indices)), replace=False)
    minority_class_2_samples_to_keep = np.random.choice(minority_class_2_indices, min(keep_2_3, len(minority_class_2_indices)), replace=False)
    minority_class_3_samples_to_keep = np.random.choice(minority_class_3_indices, min(keep_2_3, len(minority_class_3_indices)), replace=False)

    valid_indices = np.concatenate((minority_class_1_samples_to_keep, minority_class_2_samples_to_keep, minority_class_3_samples_to_keep, np.where(np.isin(flare_class_labels, [4, 5]))[0]))

    data = data[valid_indices]
    flare_class_labels = flare_class_labels[valid_indices]
    flare_type_labels = flare_type_labels[valid_indices]

    # Normalize data
    binary_labels = np.where(flare_class_labels >= 4, 1, 0)
    
    
    # Save normalized data and binary labels
    with open(processed_data_dir + "partition" + str(i+1) + "_OUS_normalized_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_OUS_binary_labels.pkl", 'wb') as f:
        pickle.dump(binary_labels, f)

    # Save flare_type_labels
    with open(processed_data_dir + "Partition" + str(i+1) + "_OUS_flare_type_labels.pkl", 'wb') as f:
        pickle.dump(flare_type_labels, f)

    print(f"Partition {i+1} normalized data shape: {data.shape}")
    print(f"Partition {i+1} binary labels distribution: {np.bincount(binary_labels)}")


# In[13]:


# Near Decision Boundary Sample Removal


# In[14]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
labels = []
flare_type_labels_list = []

num_partitions = 5

# Load processed data
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_WFS_normalized_data.pkl", 'rb') as f:
        raw_data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_WFS_flare_class_labels.pkl", 'rb') as f:
        labels.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_WFS_flare_type_labels.pkl", 'rb') as f:
        flare_type_labels_list.append(pickle.load(f))

# Gaussian Noise Injection
def smote_synthetic_samples(data, num_samples, k_neighbors=5):
    n_samples, n_timestamps, n_features = data.shape
    
    # Reshape the data for Nearest Neighbors to work on a 2D array
    reshaped_data = data.reshape(n_samples, -1)
    
    # Nearest Neighbors to determine the points for interpolation
    nn = NearestNeighbors(n_neighbors=k_neighbors+1)
    nn.fit(reshaped_data)
    
    synthetic_samples = []
    for _ in range(num_samples):
        # Randomly pick an index
        sample_index = np.random.randint(0, n_samples)
        sample = data[sample_index]
        
        # Find k-nearest neighbors
        neighbors = nn.kneighbors([reshaped_data[sample_index]], return_distance=False)[0]
        # Exclude the sample itself
        neighbors = neighbors[neighbors != sample_index]
        
        # Randomly select one of the neighbors
        neighbor_index = np.random.choice(neighbors)
        neighbor = data[neighbor_index]
        
        # Generate a synthetic sample
        diff = neighbor - sample
        gap = np.random.rand()
        synthetic_sample = sample + gap * diff
        synthetic_samples.append(synthetic_sample)
    
    return np.array(synthetic_samples)

# Process each partition
for i in range(num_partitions):
    data = raw_data[i]
    flare_class_labels = labels[i]
    flare_type_labels = flare_type_labels_list[i]

    # Oversampling
    augmented_data = []
    augmented_class_labels = []
    augmented_type_labels = []

    for class_label, factor in [(5, 10), (4, 1.5)]:
        class_indices = np.where(flare_class_labels == class_label)[0]
        class_data = data[class_indices]
        class_type_labels = flare_type_labels[class_indices]
        num_samples = int(len(class_indices) * factor)

        # Gaussian Noise Injection
        gni_data = smote_synthetic_samples(class_data, num_samples)
        gni_labels = np.full(num_samples, class_label)
        gni_type_labels = np.random.choice(class_type_labels, num_samples, replace=True)
        augmented_data.append(gni_data)
        augmented_class_labels.append(gni_labels)
        augmented_type_labels.append(gni_type_labels)

    # Combine original and augmented data
    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_class_labels = np.concatenate(augmented_class_labels, axis=0)
    augmented_type_labels = np.concatenate(augmented_type_labels, axis=0)
    data = np.concatenate((data, augmented_data), axis=0)
    flare_class_labels = np.concatenate((flare_class_labels, augmented_class_labels), axis=0)
    flare_type_labels = np.concatenate((flare_type_labels, augmented_type_labels), axis=0)

    # Remove class 2 and 3 samples
    valid_indices = np.where((flare_class_labels != 2) & (flare_class_labels != 3))[0]
    data = data[valid_indices]
    flare_class_labels = flare_class_labels[valid_indices]
    flare_type_labels = flare_type_labels[valid_indices]

    # Calculate target number of samples for minority class 1
    total_majority_class_samples = len(flare_class_labels[flare_class_labels == 5]) + len(flare_class_labels[flare_class_labels == 4])
    keep_1 = int(total_majority_class_samples * 1.2)

    # Undersample minority class 1 to match the target number of samples
    minority_class_1_indices = np.where(flare_class_labels == 1)[0]
    minority_class_1_samples_to_keep = np.random.choice(minority_class_1_indices, min(keep_1, len(minority_class_1_indices)), replace=False)

    valid_indices = np.concatenate((minority_class_1_samples_to_keep, np.where(np.isin(flare_class_labels, [4, 5]))[0]))

    data = data[valid_indices]
    flare_class_labels = flare_class_labels[valid_indices]
    flare_type_labels = flare_type_labels[valid_indices]

    # Update binary labels: classes 5 and 4 as 1, class 1 as 0
    binary_labels = np.where(np.isin(flare_class_labels, [4, 5]), 1, 0)

    # Shuffle the data and labels
    indices = np.random.permutation(len(data))
    data = data[indices]
    binary_labels = binary_labels[indices]
    flare_type_labels = flare_type_labels[indices]

    # Save normalized data and binary labels
    with open(processed_data_dir + "partition" + str(i+1) + "_CCBR_OUS_normalized_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_CCBR_OUS_binary_labels.pkl", 'wb') as f:
        pickle.dump(binary_labels, f)

    # Save flare_type_labels
    with open(processed_data_dir + "partition" + str(i+1) + "_CCBR_OUS_flare_type_labels.pkl", 'wb') as f:
        pickle.dump(flare_type_labels, f)

    print(f"Partition {i+1} normalized data shape: {data.shape}")
    print(f"Partition {i+1} binary labels distribution: {np.bincount(binary_labels)}")


# In[15]:


# Feature Selection


# In[16]:


data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
raw_data = []
labels = []
flare_type_labels_list = []

num_partitions = 5

# Load processed data
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_normalized_data.pkl", 'rb') as f:
        raw_data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_flare_class_labels.pkl", 'rb') as f:
        labels.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_flare_type_labels.pkl", 'rb') as f:
        flare_type_labels_list.append(pickle.load(f))

# Gaussian Noise Injection
def smote_synthetic_samples(data, num_samples, k_neighbors=5):
    n_samples, n_timestamps, n_features = data.shape
    
    # Reshape the data for Nearest Neighbors to work on a 2D array
    reshaped_data = data.reshape(n_samples, -1)
    
    # Nearest Neighbors to determine the points for interpolation
    nn = NearestNeighbors(n_neighbors=k_neighbors+1)
    nn.fit(reshaped_data)
    
    synthetic_samples = []
    for _ in range(num_samples):
        # Randomly pick an index
        sample_index = np.random.randint(0, n_samples)
        sample = data[sample_index]
        
        # Find k-nearest neighbors
        neighbors = nn.kneighbors([reshaped_data[sample_index]], return_distance=False)[0]
        # Exclude the sample itself
        neighbors = neighbors[neighbors != sample_index]
        
        # Randomly select one of the neighbors
        neighbor_index = np.random.choice(neighbors)
        neighbor = data[neighbor_index]
        
        # Generate a synthetic sample
        diff = neighbor - sample
        gap = np.random.rand()
        synthetic_sample = sample + gap * diff
        synthetic_samples.append(synthetic_sample)
    
    return np.array(synthetic_samples)

# Process each partition
for i in range(num_partitions):
    data = raw_data[i]
    flare_class_labels = labels[i]
    flare_type_labels = flare_type_labels_list[i]

    # Oversampling
    augmented_data = []
    augmented_class_labels = []
    augmented_type_labels = []

    for class_label, factor in [(5, 10), (4, 1.5)]:
        class_indices = np.where(flare_class_labels == class_label)[0]
        class_data = data[class_indices]
        class_type_labels = flare_type_labels[class_indices]
        num_samples = int(len(class_indices) * factor)

        # Gaussian Noise Injection
        gni_data = smote_synthetic_samples(class_data, num_samples)
        gni_labels = np.full(num_samples, class_label)
        gni_type_labels = np.random.choice(class_type_labels, num_samples, replace=True)
        augmented_data.append(gni_data)
        augmented_class_labels.append(gni_labels)
        augmented_type_labels.append(gni_type_labels)

    # Combine original and augmented data
    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_class_labels = np.concatenate(augmented_class_labels, axis=0)
    augmented_type_labels = np.concatenate(augmented_type_labels, axis=0)
    data = np.concatenate((data, augmented_data), axis=0)
    flare_class_labels = np.concatenate((flare_class_labels, augmented_class_labels), axis=0)
    flare_type_labels = np.concatenate((flare_type_labels, augmented_type_labels), axis=0)

    # Remove class 2 and 3 samples
    valid_indices = np.where((flare_class_labels != 2) & (flare_class_labels != 3))[0]
    data = data[valid_indices]
    flare_class_labels = flare_class_labels[valid_indices]
    flare_type_labels = flare_type_labels[valid_indices]

    # Calculate target number of samples for minority class 1
    total_majority_class_samples = len(flare_class_labels[flare_class_labels == 5]) + len(flare_class_labels[flare_class_labels == 4])
    keep_1 = int(total_majority_class_samples * 1.2)

    # Undersample minority class 1 to match the target number of samples
    minority_class_1_indices = np.where(flare_class_labels == 1)[0]
    minority_class_1_samples_to_keep = np.random.choice(minority_class_1_indices, min(keep_1, len(minority_class_1_indices)), replace=False)

    valid_indices = np.concatenate((minority_class_1_samples_to_keep, np.where(np.isin(flare_class_labels, [4, 5]))[0]))

    data = data[valid_indices]
    flare_class_labels = flare_class_labels[valid_indices]
    flare_type_labels = flare_type_labels[valid_indices]

    # Update binary labels: classes 5 and 4 as 1, class 1 as 0
    binary_labels = np.where(np.isin(flare_class_labels, [4, 5]), 1, 0)

    # Shuffle the data and labels
    indices = np.random.permutation(len(data))
    data = data[indices]
    binary_labels = binary_labels[indices]
    flare_type_labels = flare_type_labels[indices]

    # Save normalized data and binary labels
    with open(processed_data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_normalized_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    with open(processed_data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_binary_labels.pkl", 'wb') as f:
        pickle.dump(binary_labels, f)

    # Save flare_type_labels
    with open(processed_data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_flare_type_labels.pkl", 'wb') as f:
        pickle.dump(flare_type_labels, f)

    print(f"Partition {i+1} normalized data shape: {data.shape}")
    print(f"Partition {i+1} binary labels distribution: {np.bincount(binary_labels)}")


# In[ ]:




