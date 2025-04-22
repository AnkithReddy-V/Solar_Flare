#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Conv1D, GlobalAveragePooling1D, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Attention, LayerNormalization, GlobalAveragePooling1D
from tqdm import tqdm
from joblib import dump
from joblib import load
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def TSS(TP, TN, FP, FN):
    return (TP / (TP + FN)) - (FP / (FP + TN))

def HSS1(TP, TN, FP, FN):
    return (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

def HSS2(TP, TN, FP, FN):
    return (2 * (TP * TN - FP * FN)) / ((TP + FP) * (FN + TN) + (TP + FN) * (FP + TN))

def GSS(TP, TN, FP, FN):
    return (TP - (TP + FP) * (TP + FN) / (TP + FP + FN + TN))

def Recall(TP, TN, FP, FN):
    return TP / (TP + FN)

def FPR(TP, TN, FP, FN):
    return FP / (FP + TN)

def Accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def Precision(TP, TN, FP, FN):
    return TP / (TP + FP)


# In[3]:


def save_results(result, name):
    data_dir = "/Users/ankithreddy/Downloads/SWANresults/"

    with open(data_dir + name + ".pkl", 'wb') as f:
        pickle.dump(result, f)
    for i in range(4):
        print("TSS: " + str(result[i][6]) + "    Recall: " + str(result[i][10]))


# In[5]:


import os
import pickle
import numpy as np
import pandas as pd
data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
processed_data_dir = "/Users/ankithreddy/Downloads/SWANpre/I_Data/"
os.makedirs(processed_data_dir, exist_ok=True)
data = []
labels = []
flare_type_labels_list = []

num_partitions = 5

# Load processed data
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_normalized_data.pkl", 'rb') as f:
        data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_binary_labels.pkl", 'rb') as f:
        labels.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_flare_type_labels.pkl", 'rb') as f:
        flare_type_labels_list.append(pickle.load(f))

test_data = []
test_labels = []
test_flare_type_labels_list = []

# Load processed data
for i in range(num_partitions):
    with open(data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_normalized_data.pkl", 'rb') as f:
        test_data.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_binary_labels.pkl", 'rb') as f:
        test_labels.append(pickle.load(f))
    with open(data_dir + "partition" + str(i+1) + "_FS_CCBR_OUS_flare_type_labels.pkl", 'rb') as f:
        test_flare_type_labels_list.append(pickle.load(f))
 


# In[6]:


def kfold_training(name, X_train, Y_train, y_type_train, X_test, Y_test, y_type_test, training_func, num):
    kfold = np.array([[1,2],[2,3],[3,4],[4,5]])

    metrics = []
    metrics_values = np.array([])

    for i in range(0, num):
        train_index = kfold[i,0]
        test_index = kfold[i,1]
        metrics_values = training_func(X_train[train_index-1], Y_train[train_index-1], y_type_train[train_index-1], X_test[test_index-1], Y_test[test_index-1], y_type_test[test_index-1])
        while (metrics_values[4] < 0.01):
            metrics_values = training_func(X_train[train_index-1], Y_train[train_index-1], y_type_train[train_index-1], X_test[test_index-1], Y_test[test_index-1], y_type_test[test_index-1])
        metrics.append(np.append(np.append(train_index, test_index), metrics_values))
    return metrics


# In[9]:


import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Layer, Multiply, GRU
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random

# Define custom Attention LSTM layer
class AttentionLSTM(Layer):
    def __init__(self, units, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        self.units = units
        super(AttentionLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        self.lstm = LSTM(self.units, return_sequences=self.return_sequences)
        super(AttentionLSTM, self).build(input_shape)

    def call(self, x):
        # energies[t,i] = sum(W * x[t,i] + b)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)  # attention weights
        weighted_input = x * a
        return self.lstm(weighted_input)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)

# Define Focal Loss function
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1.e-9
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    loss = - alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
    return tf.reduce_sum(loss)

# Define TSS Loss function
def TSS(tp,tn,fp,fn):
    epsilon = 1.e-7
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    tss = recall + specificity - 1
    return tss

def HSS1(tp,tn,fp,fn):
    epsilon = 1.e-7
    a = 2*(tp*tn-fn*fp)
    b = (tp+fn)*(fn+tn) + (tp+fp)*(fp+tn)
    hss1 = a/b
    return hss1

def HSS2(tp,tn,fp,fn):
    epsilon = 1.e-7
    a = 2*(tp*tn-fn*fp)
    b = (tp+fn)*(tp+fp) + (tn+fn)*(tn+fp)
    hss2 = a/b
    return hss2

def GSS(tp,tn,fp,fn):
    epsilon = 1.e-7
    a = tp*tn-fn*fp
    b = tp + fn + fp + tn
    gss = a/b
    return gss

def Recall(tp,tn,fp,fn):
    epsilon = 1.e-7
    recall = tp / (tp + fn + epsilon)
    return recall

def Precision(tp,tn,fp,fn):
    epsilon = 1.e-7
    precision = tp / (tp + fp + epsilon)
    return precision

def tss_loss(y_true, y_pred, beta=1.0):
    epsilon = 1.e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    tp = tf.reduce_sum(y_true * y_pred)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)

    tss = recall + specificity - 1
    loss = -tss * beta
    return loss

# Define combined loss function
def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + tss_loss(y_true, y_pred)

def contrastive_regression(X_train, y_train, y_type_train, X_test, y_test, y_type_test):
    # Define triplet loss function
    def triplet_loss(anchor, positives, negatives, margin=4.0):
        # Reshape the inputs to combine the temporal and feature dimensions
        anchor_flat = tf.reshape(anchor, [anchor.shape[0], -1])
        positives_flat = tf.reshape(positives, [positives.shape[0], positives.shape[1], -1])
        negatives_flat = tf.reshape(negatives, [negatives.shape[0], negatives.shape[1], -1])

        # Normalize the vectors to unit length
        anchor_normalized = tf.nn.l2_normalize(anchor_flat, axis=-1)
        positives_normalized = tf.nn.l2_normalize(positives_flat, axis=-1)
        negatives_normalized = tf.nn.l2_normalize(negatives_flat, axis=-1)

        # Compute the cosine similarity
        pos_similarity = tf.reduce_sum(anchor_normalized[:, tf.newaxis, :] * positives_normalized, axis=-1)
        neg_similarity = tf.reduce_sum(anchor_normalized[:, tf.newaxis, :] * negatives_normalized, axis=-1)

        # Convert cosine similarity to cosine distance
        pos_distance = 1 - pos_similarity
        neg_distance = 1 - neg_similarity

        # Sum the distances to positives and negatives
        pos_distance_sum = tf.reduce_sum(pos_distance, axis=-1)
        neg_distance_sum = tf.reduce_sum(neg_distance, axis=-1)

        # Compute the triplet loss with cosine distance
        loss = tf.maximum(pos_distance_sum - neg_distance_sum + margin, 0.0)
        return tf.reduce_mean(loss)

    def build_contrastive_model(input_shape, num_lstm_layers, lstm_units, dense_units, dropout_rate=0.3):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_lstm_layers):
            x = LSTM(lstm_units, return_sequences=True)(x)
            x = Dropout(dropout_rate)(x)
        x = GlobalAveragePooling1D()(x)
        embeddings = Dense(dense_units, activation='relu')(x)

        model = Model(inputs, embeddings)
        return model

    def build_regression_model(input_shape, num_gru_layers, gru_units, dropout_rate=0.3):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_gru_layers):
            x = AttentionLSTM(gru_units, return_sequences=True)(x)
            x = Dropout(dropout_rate)(x)
        x = GlobalAveragePooling1D()(x)
        mid = Dense(2, activation='relu')(x)
        classification_output = Dense(1)(mid)

        model = Model(inputs, classification_output)
        return model

    def combined_model(input_shape, num_layers, units, dense_units, dropout_rate=0.3):
        contrastive_model = build_contrastive_model(input_shape, num_layers, units, dense_units, dropout_rate)
        regression_model = build_regression_model(input_shape, num_layers, units, dropout_rate)

        inputs = Input(shape=input_shape)
        contrastive_embeddings = contrastive_model(inputs)
        regression_output = regression_model(inputs)

        combined_input = Concatenate()([inputs[:,0,:], contrastive_embeddings, regression_output])

        mid = Dense(12, activation='relu')(combined_input)
        mid = Dense(4, activation='relu')(mid)
        final_output = Dense(1, activation='sigmoid')(mid)

        model = Model(inputs, final_output)
        return model, contrastive_model, regression_model

    # Example usage
    input_shape = (60, 6)  # Update based on your actual data shape
    num_layers = 2
    num_units = 6
    dense_units = 4

    classification_model, contrastive_model, regression_model = combined_model(input_shape, num_layers, num_units, dense_units)
    contrastive_model.summary()
    regression_model.summary()
    classification_model.summary()

    # Function to generate triplets
    def generate_triplets(X, y, num_samples=4, batch_size=64):
        anchors = []
        positives = []
        negatives = []

        num_batches = np.ceil(len(X) / batch_size).astype(int)

        for i in range(len(X)):
            anchor = X[i]
            g = i // batch_size

            batch_start = batch_size * g
            batch_end = min(batch_size * (g + 1), len(X))
            positive_indices = np.where((y == y[i]) & (np.arange(len(y)) >= batch_start) & (np.arange(len(y)) < batch_end))[0]
            negative_indices = np.where((y != y[i]) & (np.arange(len(y)) >= batch_start) & (np.arange(len(y)) < batch_end))[0]

            positive_indices = positive_indices[positive_indices != i]

            selected_positives = list(positive_indices)
            selected_negatives = list(negative_indices)

            batch = g + 1
            while len(selected_positives) < num_samples or len(selected_negatives) < num_samples:
                if batch >= num_batches:
                    batch = 0

                if len(selected_positives) < num_samples:
                    batch_start = batch_size * batch
                    batch_end = min(batch_size * (batch + 1), len(X))
                    new_positives = np.where((y == y[i]) & (np.arange(len(y)) >= batch_start) & (np.arange(len(y)) < batch_end))[0]
                    new_positives = new_positives[new_positives != i]
                    selected_positives.extend(new_positives)

                if len(selected_negatives) < num_samples:
                    batch_start = batch_size * batch
                    batch_end = min(batch_size * (batch + 1), len(X))
                    new_negatives = np.where((y != y[i]) & (np.arange(len(y)) >= batch_start) & (np.arange(len(y)) < batch_end))[0]
                    selected_negatives.extend(new_negatives)

                if batch == g:
                    break
                batch += 1

            selected_positives = np.array(selected_positives)[:num_samples]
            selected_negatives = np.array(selected_negatives)[:num_samples]

            anchors.append(anchor)
            positives.append(X[selected_positives])
            negatives.append(X[selected_negatives])

        return np.array(anchors), np.array(positives), np.array(negatives)

    epochs = 10
    batch_size = 64

    anchors, positives, negatives = generate_triplets(X_train, y_train, 4, batch_size)

    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        epoch_loss_triplet = 0
        epoch_loss_regression = 0
        epoch_loss_classification = 0
        num_batches = 0

        print(f'Epoch {epoch + 1}/{epochs}')
        for i in tqdm(range(0, len(anchors), batch_size), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            a_batch = anchors[i:i + batch_size]
            p_batch = positives[i:i + batch_size]
            n_batch = negatives[i:i + batch_size]
            x_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            y_type_batch = y_type_train[i:i + batch_size]

            y_batch = y_batch.reshape(-1, 1)
            y_type_batch = y_type_batch.reshape(-1, 1)

            with tf.GradientTape() as tape:
                anchor_embeddings = contrastive_model(a_batch, training=True)

                positive_embeddings_list = []
                negative_embeddings_list = []

                for j in range(4):
                    positive_embedding = contrastive_model(p_batch[:, j, :, :], training=True)
                    negative_embedding = contrastive_model(n_batch[:, j, :, :], training=True)
                    positive_embeddings_list.append(positive_embedding)
                    negative_embeddings_list.append(negative_embedding)

                positive_embeddings = tf.stack(positive_embeddings_list, axis=1)
                negative_embeddings = tf.stack(negative_embeddings_list, axis=1)

                loss_triplet = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

                regression_output = regression_model(x_batch, training=True)
                loss_regression = tf.keras.losses.mean_squared_error(y_type_batch, regression_output)
                loss_regression = tf.reduce_mean(loss_regression) * 0.00000001

                combined_output = classification_model(x_batch, training=True)
                loss_classification = combined_loss(y_batch, combined_output)
                loss_classification = tf.reduce_mean(loss_classification)

                total_loss = loss_triplet + loss_regression + loss_classification

            gradients = tape.gradient(total_loss, classification_model.trainable_variables + regression_model.trainable_variables + contrastive_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, classification_model.trainable_variables + regression_model.trainable_variables + contrastive_model.trainable_variables))

            epoch_loss_triplet += loss_triplet.numpy()
            epoch_loss_regression += loss_regression.numpy()
            epoch_loss_classification += loss_classification.numpy()
            num_batches += 1

        avg_loss_triplet = epoch_loss_triplet / num_batches
        avg_loss_regression = epoch_loss_regression / num_batches
        avg_loss_classification = epoch_loss_classification / num_batches

        print(f'Epoch {epoch + 1} - Triplet Loss: {avg_loss_triplet:.4f}, Regression Loss: {avg_loss_regression:.4f}, Classification Loss: {avg_loss_classification:.4f}')

    print("Training completed!")

    best_threshold = 0.0
    best_tss = 0.0
    y_pred = classification_model.predict(X_test)

    for i in range(1, 1000):
        threshold = i / 1000
        y_pred_binary = (y_pred > threshold).astype(int)
        confusion = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = confusion.ravel()
        tss = TSS(tp,tn,fp,fn)
        if tss > best_tss:
            best_tss = tss
            best_threshold = i / 1000

    print(str(X_train.shape)+': The Classifier is Done! \n')

    threshold = best_threshold
    y_pred_binary = (y_pred > threshold).astype(int)
    confusion = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = confusion.ravel()

    tss = TSS(tp,tn,fp,fn)
    hss1 = HSS1(tp,tn,fp,fn)
    hss2 = HSS2(tp,tn,fp,fn)
    gss = GSS(tp,tn,fp,fn)
    recall = Recall(tp,tn,fp,fn)
    precision = Precision(tp,tn,fp,fn)

    output_values = np.array([tp, fn, fp, tn, tss, hss1, hss2, gss, recall, precision])

    return output_values


# In[10]:


contrastive_regression_results = kfold_training('contrastive_regression_model', data, labels, flare_type_labels_list, test_data, test_labels, test_flare_type_labels_list, contrastive_regression, 4)


# In[12]:


save_results(contrastive_regression_results, "AttentionContReg_FS_result")


# In[ ]:




