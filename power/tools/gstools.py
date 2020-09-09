#!/usr/bin/python
"""
tools related to training the network
"""

import h5py
import numpy as np
from os import listdir,makedirs
from os.path import isfile, join, exists
from tqdm import tqdm, tqdm_notebook
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.markers
import time
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D
from keras.models import Model, load_model
from keras.layers import Input
import keras
from sklearn.preprocessing import LabelEncoder
from keras import backend as K

def load_dataset(data_path=Path('/dovilabfs/work/tommaria/gw/data/gravityspy'), verbose=1, with_times=False):
    """
        this function loads the dataset of gravity spy, after preprocessing and split into training, test and validation sets
        inputs:
            data_path - path to where the training set file 'training_set.hdf5' is located
            verbose - verbosity flag
            with_times - flag that indicates if loading with times or not
        outputs:
            x_train, x_test, x_val - numpy arrays of the training, test and validation images
            y_train, y_test, y_val - lists of corresponding labels
    """
    start_time = time.time()
    
    if with_times:
        file_name = 'training_set_with_times.hdf5'
    else:
        file_name = 'training_set.hdf5'
    
    with h5py.File(join(data_path, file_name), 'r') as f:
        x_train = np.asarray(f['x_train'])
        x_test = np.asarray(f['x_test'])
        x_val = np.asarray(f['x_val'])
        y_train = [item.decode('ascii') for item in list(f['y_train'])]
        y_test = [item.decode('ascii') for item in list(f['y_test'])]
        y_val = [item.decode('ascii') for item in list(f['y_val'])]
        if with_times:
            times_train = np.asarray(f['times_train'])
            times_test = np.asarray(f['times_test'])
            times_val = np.asarray(f['times_val'])
    
    if verbose:
        print("--- Data loading time is %.7s seconds ---\n" % (time.time() - start_time))
    
    if with_times:
        return x_train, x_test, x_val, y_train, y_test, y_val, times_train, times_test, times_val
    else:
        return x_train, x_test, x_val, y_train, y_test, y_val

def encode_labels(y_train, y_test=None, y_val=None):
    labels = np.unique(y_train)
    # encode labels to integers
    encoder = LabelEncoder().fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    y_val = encoder.transform(y_val)

    # one-hot encode the labels
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    return labels, y_train, y_test, y_val, num_classes

def preproc(x_train, x_test=None, x_val=None):
    train_mean = np.mean(x_train, axis=0)
    x_train = x_train - train_mean
    x_test = x_test - train_mean
    x_val = x_val - train_mean
    return x_train, x_test, x_val

def create_embedding(x, pca=False):
    """
        this function performs t-SNE embedding with or without PCA first
        inputs:
            x - numpy array - data to be embedded
            pca - bool - if True first reduce dimensions using PCA and then embed the principal components. default False
        outputs:
            x_tsne - numpy array - embedded data using t-SNE
    """

    # if trying to embed an image turn it into a vector
    if len(x.shape) > 2:
        x = np.reshape(x, (x.shape[0],-1))

    if pca:
        x = PCA(n_components=50).fit_transform(x)

    x_tsne = TSNE(n_components=2, verbose=0, random_state=0).fit_transform(x)
    return x_tsne

def plot_embedding(x, y=None, title=None):
    """
        this functions scatterplots embedded data
        inputs:
            x - numpy array - embedded data to be plot
            y - list - labels of the data used to color and select markers fo the data (if provided). default None
            title - string - title of the plot (if provided). default None
    """

    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x_embedded = (x - x_min) / (x_max - x_min)

    plt.figure(figsize=(15, 15))

    if y is None:
        sns.scatterplot(x_embedded[:,0], x_embedded[:,1], s=150, alpha=0.5)
    else:
        sns.scatterplot(x_embedded[:,0], x_embedded[:,1],
                        hue=y, style=y, markers=gen_stylemap(y), s=150, alpha=0.5)
    plt.title(title, size=20)
    plt.show()
    return

def gen_stylemap(y):
    """
        this function generates markers to be used in plotting embedded data
        input: y - list - labels of the data to be used to generate the markers
        output: stylemap - dictionary - mapping between label to marker
    """

    labels = np.unique(y)
    num_styles = len(Line2D.filled_markers)
    stylemap = dict()
    for k, label in enumerate(labels):
        stylemap.update({label: Line2D.filled_markers[k % num_styles]})

    return stylemap

def embed_plot(x, pca=False, y=None, title=None):
    """
    """

    x_tsne = create_embedding(x, pca)
    plot_embedding(x_tsne, y, title)
    return x_tsne

def load_fe_model(model_name, weights_file=None):
    """
        this function loads feature extraction model
        inputs:
            model_name - string - name of the model to load
            weights_file - string - name of file containing weights to load to the model (if provided). default None
        output:
            fe_model - keras Model - feature extraction model
    """

    model = load_model(model_name)
    if weights_file is not None:
        model.load_weights(weights_file)

    # the feature extraction model is the network in the trained model without the last classification layer
    fe_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return fe_model

def load_models(model_name, weights_file=None):
    """
        this function loads feature extraction model and prediction model
        inputs:
            model_name - string - name of the model to load
            weights_file - string - name of file containing weights to load to the model (if provided). default None
        output:
            features_model - keras Model - feature extraction model
            predictions_model - keras Model - prediction model on extracted features
    """

    model = load_model(model_name)
    if weights_file is not None:
        model.load_weights(weights_file)

    # the feature extraction model is the network in the trained model without the last classification layer
    features_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    pred_input = Input(shape=model.layers[-2].output_shape[1:])
    pred_output = model.layers[-1](pred_input)
    predictions_model = Model(inputs=pred_input, outputs=pred_output)
    return model, features_model, predictions_model

def extract_features(x, model_name, weights_file=None):
    """

    """

    fe_model = load_fe_model(model_name, weights_file) # load feature extraction model
    features = fe_model.predict(x, verbose=0) # extract features
    return features

def extract_embed_plot(x, model_name, weights_file=None, pca=False, y=None, title=None):
    """
    """

    features = extract_features(x, model_name, weights_file)
    embed_plot(x, pca, y, title)
    return embedded_features, features

def gram_matrix(x):
    """This function computes the gram matrices of a given input image.
    """
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def p_gram(features, p):
    """Compute p-order Gram matrix of features of a single layer.
    """
    if features.ndim == 1:
        temp = np.empty((1, features.shape[0]))
        temp[0,:] = features
    else:
        temp = features
    features = temp ** p
    G = np.dot(features, np.transpose(features)) ** (1/p)
    G_bar = G[np.triu_indices(G.shape[0])]
    return G_bar

def mins_maxs(x, y, p_list, functor, batch_size=32):
    """Compute arrays of minimum and maximum values of the Gram matrices for each class, layer and order.
    """
    mins = []
    maxs = []
    labels = np.unique(y)
    for c, label in enumerate(labels):
        idx = [j for j, temp in enumerate(y) if temp == label]
        temp_mins, temp_maxs = class_mins_maxs(x[idx], p_list, functor, batch_size)
        mins.append(temp_mins)
        maxs.append(temp_maxs)
    return mins, maxs

# def class_mins_maxs(x, p_list, functor, batch_size=32):
#     """Compute arrays of minimum and maximum values of the Gram matrices for specific given class.
#     """
#     mins = []
#     maxs = []
#     for sample in x:
#         layer_outs = functor([sample[np.newaxis,...], 0])
#         for l, features in enumerate(layer_outs):
#             if features[0].ndim != 3:
#                 continue
                
#             features = np.transpose(features[0], (2, 0, 1))
#             features = np.reshape(features, (features.shape[0], -1))
#             if l == len(mins):
#                 mins.append([None]*len(p_list))
#                 maxs.append([None]*len(p_list))
            
#             for j, p in enumerate(p_list):
#                 G_bar = p_gram(features, p)
#                 if mins[l][j] is None:
#                     mins[l][j] = G_bar
#                     maxs[l][j] = G_bar
#                 else:
#                     mins[l][j] = np.minimum(mins[l][j], G_bar)
#                     maxs[l][j] = np.maximum(maxs[l][j], G_bar)
#     return mins, maxs

def class_mins_maxs(x, p_list, functor, batch_size=32):
    mins = []
    maxs = []
    for i in range(0, x.shape[0], batch_size):
        batch_layer_outs = functor([x[i:i+batch_size], 0])
        for l, batch_features in enumerate(batch_layer_outs):
            if batch_features[0].ndim != 3:
                continue
            
            for features in batch_features:
                features = np.transpose(features, (2, 0, 1))
                features = np.reshape(features, (features.shape[0], -1))
                if l == len(mins):
                    mins.append([None]*len(p_list))
                    maxs.append([None]*len(p_list))

                for j, p in enumerate(p_list):
                    G_bar = p_gram(features, p)
                    if mins[l][j] is None:
                        mins[l][j] = G_bar
                        maxs[l][j] = G_bar
                    else:
                        mins[l][j] = np.minimum(mins[l][j], G_bar)
                        maxs[l][j] = np.maximum(maxs[l][j], G_bar)
    return mins, maxs

# def deviations_features_predictions(x, labels, p_list, mins, maxs, functor):
#     """Compute Gram matrix deviations, latent features and predictionsfor given input.
#     """
#     delta = []
#     latent = []
#     y_hat = []
#     for i, sample in enumerate(x):
#         layer_outs = functor([sample[np.newaxis,...], 0])
#         latent.append(layer_outs[-2][0])
#         y_hat.append(layer_outs[-1][0])
#         c = np.argmax(layer_outs[-1][0])        
#         dev = []
#         for l, features in enumerate(layer_outs):
#             dev_l = 0
#             if features[0].ndim != 3:
#                 continue
            
#             features = np.transpose(features[0], (2, 0, 1))
#             features = np.reshape(features, (features.shape[0], -1))
#             for j, p in enumerate(p_list):
#                 G_bar = p_gram(features, p)
#                 for k, val in enumerate(G_bar):
#                     if val < mins[c][l][j][k]:
#                         dev_l += (mins[c][l][j][k] - val) / (np.abs(mins[c][l][j][k]) + 1e-6)
#                     elif val > maxs[c][l][j][k]:
#                         dev_l += (val - maxs[c][l][j][k]) / (np.abs(maxs[c][l][j][k]) + 1e-6)

#             dev.append(dev_l)

#         dev = np.asarray(dev)
#         delta.append(dev)

#     latent = np.asarray(latent)
#     y_hat = np.asarray(y_hat)
#     delta = np.asarray(delta)
#     return delta, latent, y_hat

def deviations_features_predictions(x, labels, p_list, mins, maxs, functor, batch_size=32):
    """Compute Gram matrix deviations, latent features and predictionsfor given input.
    """
    delta = []
    latent = []
    y_hat = []
    for i in range(0, x.shape[0], batch_size):
        batch_layer_outs = functor([x[i:i+batch_size], 0])
        latent.extend(batch_layer_outs[-2])
        y_hat.extend(batch_layer_outs[-1])
        cs = np.argmax(batch_layer_outs[-1], axis=1)
        print(np.asarray(y_hat).shape)
        print(cs.shape)
        for idx in range(cs.shape[0]):
            c = cs[idx]
            dev = []
            for l in range(len(batch_layer_outs)):
                features = batch_layer_outs[l][idx]
                dev_l = 0
                if features.ndim != 3:
                    continue

                features = np.transpose(features, (2, 0, 1))
                features = np.reshape(features, (features.shape[0], -1))
                for j, p in enumerate(p_list):
                    G_bar = p_gram(features, p)
                    for k, val in enumerate(G_bar):
                        if val < mins[c][l][j][k]:
                            dev_l += (mins[c][l][j][k] - val) / (np.abs(mins[c][l][j][k]) + 1e-6)
                        elif val > maxs[c][l][j][k]:
                            dev_l += (val - maxs[c][l][j][k]) / (np.abs(maxs[c][l][j][k]) + 1e-6)

                dev.append(dev_l)

            dev = np.asarray(dev)
            delta.append(dev)

    latent = np.asarray(latent)
    y_hat = np.asarray(y_hat)
    delta = np.asarray(delta)
    return delta, latent, y_hat

def layer_conditions(layer):
    """Conditions to choose the layer for Gram matrix comutation (convolution and activation layers).
    """
    ll = layer.name.split('_')
    if len(ll) < 3:
        return False
    return (ll[2] == 'out' and int(ll[1][-1]) % 2 == 1)
    # if len(ll) < 4:
    #     return False
    # return (ll[0].startswith('conv') and 
    #         ll[1].startswith('block') and 
    #         ll[2] != 'preact' and
    #         (ll[3] == 'conv' or ll[3] == 'relu'))

def p_gram_K(features, p):
    """Compute Gram matrices using keras backend.
    """
    G_bar = []
    assert(features.ndim == 4)
    features = K.permute_dimensions(features, (0, 3, 1, 2))
    features = K.reshape(features, (features.shape[0], features.shape[1], -1)) ** p
    G = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1))) ** (1/p)
    for g in G:
        G_bar.append(np.asarray(g)[np.triu_indices(g.shape[0])])
    return np.asarray(G_bar)

def p_gram_new(features, p):
    """Compute Gram matrices using keras backend.
    """
    G_bar = []
    assert(features.ndim == 3)
    features = np.transpose(features, (2, 0, 1))
    features = np.reshape(features, (features.shape[0], -1)) ** p
    G = np.dot(features, np.transpose(features)) ** (1/p)
    G_bar = G[np.triu_indices(G.shape[0])]
    return np.asarray(G_bar)

def mins_maxs_new(x, y, p_list, functor, batch_size=32):
    """Compute mins, maxs using keras backend
    """
    mins = []
    maxs = []
    first = True
    labels = np.unique(y)
    for c, label in enumerate(labels):
        idx = [j for j, temp in enumerate(y) if temp == label]
        temp_mins, temp_maxs, temp_latent, temp_y_hat = class_mins_maxs_new(x[idx], p_list, functor, batch_size)
        mins.append(temp_mins)
        maxs.append(temp_maxs)
        if first:
            latent = np.zeros((x.shape[0], temp_latent.shape[1]))
            y_hat = np.zeros((x.shape[0], temp_y_hat.shape[1]))
            first = False

        latent[idx] = temp_latent
        y_hat[idx] = temp_y_hat
    return mins, maxs, latent, y_hat

def class_mins_maxs_new(x, p_list, functor, batch_size=32):
    """Compute class mins, maxs using keras backend.
    """
    mins = []
    maxs = []
    latent = []
    y_hat = []
    for i in range(0, x.shape[0], batch_size):
        batch_layer_outs = functor([x[i:i+batch_size], 0])
        latent.extend(batch_layer_outs[-2])
        y_hat.extend(batch_layer_outs[-1])
        for l, batch_features in enumerate(batch_layer_outs):
            if batch_features[0].ndim != 3:
                continue
            
            if l == len(mins):
                mins.append([None]*len(p_list))
                maxs.append([None]*len(p_list))

            for j, p in enumerate(p_list):
                G_bar = p_gram_K(batch_features, p)
                current_min = G_bar.min(axis=0)
                current_max = G_bar.max(axis=0)
                if mins[l][j] is None:
                    mins[l][j] = current_min
                    maxs[l][j] = current_max
                else:
                    mins[l][j] = np.minimum(mins[l][j], current_min)
                    maxs[l][j] = np.maximum(maxs[l][j], current_max)
    return mins, maxs, np.asarray(latent), np.asarray(y_hat)

def deviations_features_predictions_new(x, labels, p_list, mins, maxs, functor, batch_size=32):
    """Compute Gram matrix deviations, latent features and predictionsfor given input, using keras backend.
    """
    delta = []
    latent = []
    y_hat = []
    for i in range(0, x.shape[0], batch_size):
        batch_layer_outs = functor([x[i:i+batch_size], 0])
        latent.extend(batch_layer_outs[-2])
        y_hat.extend(batch_layer_outs[-1])
        cs = np.argmax(batch_layer_outs[-1], axis=1)
        # print(np.asarray(y_hat).shape)
        # print(cs.shape)
        for idx in range(cs.shape[0]):
            c = cs[idx]
            dev = []
            for l in range(len(batch_layer_outs)):
                features = batch_layer_outs[l][idx]
                dev_l = 0
                if features.ndim != 3:
                    continue

                for j, p in enumerate(p_list):
                    G_bar = p_gram_K(features[np.newaxis, ...], p)
                    # G_bar = p_gram_new(features, p)
                    dev_l += np.sum(K.relu(mins[c][l][j] - G_bar) / (np.abs(mins[c][l][j]) + 1e-6))
                    dev_l += np.sum(K.relu(G_bar - maxs[c][l][j]) / (np.abs(maxs[c][l][j]) + 1e-6))

                dev.append(dev_l)

            dev = np.asarray(dev)
            delta.append(dev)

    latent = np.asarray(latent)
    y_hat = np.asarray(y_hat)
    delta = np.asarray(delta)
    return delta, latent, y_hat

def load_group(grp):
    """Load group containing lists of mins/maxs from hdf5 file.
    """
    classes = np.unique(sorted([int(key.split('_')[0]) for key in grp.keys()]))
    layers = np.unique(sorted([int(key.split('_')[1]) for key in grp.keys()]))
    grp_list = []
    for c in classes:
        c_list = []
        for l in layers:
            c_list.append(np.asarray(grp[str(c) + '_' + str(l)]))
        
        grp_list.append(c_list)
    return grp_list