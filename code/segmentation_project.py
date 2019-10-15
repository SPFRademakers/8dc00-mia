#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project code+scripts for 8DC00 course
"""

# Imports

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
import scipy
from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from mpl_toolkits.mplot3d import Axes3D


def create_dataset(image_number, slice_number, task):
    # create_dataset creates a dataset for a particular subject (image), slice and task
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    # Output:
    # X           - Nxk feature matrix, where N is the number of pixels and k is the number of features
    # Y           - Nx1 vector with labels
    # feature_labels - kx1 cell array with descriptions of the k features

    # extract features from the subject/slice
    X, feature_labels = extract_features(image_number, slice_number)

    # create labels
    Y = util.create_labels(image_number, slice_number, task)

    return X, Y, feature_labels


def extract_features(image_number, slice_number):
    # extracts features for [image_number]_[slice_number]_t1.tif and [image_number]_[slice_number]_t2.tif
    # Input:
    # image_number - Which subject (scalar)
    # slice_number - Which slice (scalar)
    # Output:
    # X            - N x k dataset, where N is the number of pixels and k is the total number of features
    # features     - k x 1 cell array describing each of the k features

    base_dir = '../data/dataset_brains/'

    t1 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif')
    t2 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t2.tif')

    n = t1.shape[0]
    features = ()

    t2f = t2.flatten().T.astype(float)
    t2f = t2f.reshape(-1, 1)
    print(np.unique(t2f))
    for i in range(len(t2f)):
        if t2f[i] > 200:
            t2f[i] = 255
        else:
            t2f[i] = 1

    # addition of features T1 and T2 intensities
    t1f = t1.flatten().T.astype(float)
    t1f = t1f.reshape(-1, 1)
    t2f = t2.flatten().T.astype(float)
    t2f = t2f.reshape(-1, 1)
    X = np.concatenate((t1f, t2f), axis=1)

    features += ('T1 intensity',)
    features += ('T2 intensity',)
    #
    # # addition of feature coordinates
    # c, _ = seg.extract_coordinate_feature(t1)
    # X1 = np.concatenate((X, c), axis=1)
    # features += ('coordinates',)
    #
    # addition of feature T1 gaussian blurred
    t1_g = scipy.ndimage.gaussian_filter(t1, sigma=2)
    t1_gf = t1_g.flatten().T.astype(float)
    t1_gf = t1_gf.reshape(-1, 1)
    X2 = np.concatenate((X, t1_gf), axis=1)
    features += ('guassian blurred (T1)',)
    #
    # # addition of feature T2 gaussian blurred
    # t2_g = ndimage.gaussian_filter(t2, sigma=2)
    # t2_gf = t2_g.flatten().T.astype(float)
    # t2_gf = t2_gf.reshape(-1, 1)
    # X3 = np.concatenate((X2, t2_gf), axis=1)
    # features += ('guassian blurred (T2)',)
    #
    # # addition of laplacian kernel on T1
    # k_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    # Ic1 = ndimage.convolve(t1, k_laplace, mode='reflect')
    # Ic1_f = Ic1.flatten().T.astype(float)
    # Ic1_f = Ic1_f.reshape(-1, 1)
    # X4 = np.concatenate((X3, Ic1_f), axis=1)
    # features += ('laplacian kernel (T1)',)

    # addition of laplacian kernel on T2
    k_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    Ic2 = scipy.ndimage.convolve(t2, k_laplace, mode='reflect')
    Ic2_f = Ic2.flatten().T.astype(float)
    Ic2_f = Ic2_f.reshape(-1, 1)
    X3 = np.concatenate((X2, Ic2_f), axis=1)
    features += ('laplacian kernel (T2)',)

    return t2f, features

def cost_kmeans(X, w_vector):
    # Computes the cost of assigning data in X to clusters in w_vector
    # Input:
    # X         - data
    # w_vector  - means of clusters
    # Output:
    # J         - new means of clusters

    # Get the data dimensions
    n, m = X.shape

    # Number of clusters per feature
    K = int(len(w_vector)/m)

    # Reshape cluster centers into dataset format
    W = w_vector.reshape(K, m)

    D = pairwise_distances(X, W, metric='euclidean')
    min_index = np.argmin(D, axis=1)
    min_dist = D[np.arange(D.shape[0]), min_index]
    J = np.sum(min_dist**2)

    return J


def kmeans_clustering(test_data, K=2):
    # Returns the labels for test_data, predicted by the kMeans
    # classifier which assumes that clusters are ordered by intensity
    #
    # Input:
    # test_data          num_test x p matrix with features for the test data
    # k                  Number of clusters to take into account (2 by default)
    # Output:
    # predicted_labels    num_test x 1 predicted vector with labels for the test data

    N, M = test_data.shape

    # link to the cost function of kMeans
    fun = lambda w: cost_kmeans(test_data, w)

    # the learning rate
    mu = 0.00001

    # iterations
    num_iter = 100

    # Initialize cluster centers and store them in w_initial
    idx = np.random.randint(N, size=2)
    w_initial = test_data[idx, :]

    # Reshape centers to a vector (needed by ngradient)
    w_vector = w_initial.reshape(K*M, 1)

    for i in np.arange(num_iter):

        # gradient ascent
        g = util.ngradient(fun, w_vector)
        change = mu * g
        w_vector = w_vector - change[:, np.newaxis]

    # Reshape back to dataset
    w_final = w_vector.reshape(K, M)

    # Find min_dist and min_index
    D = scipy.spatial.distance.cdist(test_data, w_final, metric='euclidean')
    min_index = np.argmin(D, axis=1)

    # Sort by intensity of cluster center
    sorted_order = np.argsort(w_final[:, 0], axis=0)

    # Update the cluster indices based on the sorted order and return results in predicted_labels
    predicted_labels = np.empty(*min_index.shape)
    predicted_labels[:] = np.nan

    for i in np.arange(len(sorted_order)):
        predicted_labels[min_index == sorted_order[i]] = i

    return predicted_labels


def segmentation_mymethod(train_data_matrix, train_labels_matrix, test_data, task='brain'):
    # segments the image based on your own method!

    # Input:
    # train_data_matrix     num_pixels x num_features x num_subjects matrix of features
    # train_labels_matrix   num_pixels x num_subjects matrix of labels
    # test_data             num_pixels x num_features test data
    # task                  String corresponding to the segmentation task: either 'brain' or 'tissue'

    # Output:
    # predicted_labels      Predicted labels for the test slice

    predicted_labels = kmeans_clustering(test_data, K=2)

    return predicted_labels


def segmentation_demo():

    # Data name specification
    train_subject = 1
    test_subject = 2
    train_slice = 1
    test_slice = 1
    task = 'tissue'

    # Load data
    train_data, train_labels, train_feature_labels = util.create_dataset(train_subject, train_slice, task)
    test_data, test_labels, test_feature_labels = util.create_dataset(test_subject, test_slice, task)

    # Normalize and feed data through X_pca
    train_norm, _ = seg.normalize_data(train_data)
    Xpca, v, w, fraction_variance, ix = seg.mypca(train_norm)
    relevant_feature = int(np.sum(fraction_variance < 0.95)) + 1
    train_norm_ord = train_norm[:, ix]
    train_norm = train_norm_ord[:, :relevant_feature]

    # find the predicted labels (here: the train_labels)
    predicted_labels = seg.segmentation_atlas(None, train_labels, None)

    # Calculate the error and dice score of these predicted labels in comparison to test labels
    err = util.classification_error(test_labels, predicted_labels)
    dice = util.dice_multiclass(test_labels, predicted_labels)

    # Display results
    true_mask = test_labels.reshape(240, 240)
    predicted_mask = predicted_labels.reshape(240, 240)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(true_mask, 'gray')
    ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
    print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

    # COMPARE METHODS
    num_images = 5
    num_methods = 3
    im_size = [240, 240]

    # make space for error and dice data
    all_errors = np.empty([num_images, num_methods])
    all_errors[:] = np.nan
    all_dice = np.empty([num_images, num_methods])
    all_dice[:] = np.nan

    # data name specification
    all_subjects = np.arange(num_images)
    train_slice = 1
    task = 'tissue'

    # make space for data
    all_data_matrix = np.empty([train_norm.shape[0], train_norm.shape[1], num_images])
    all_labels_matrix = np.empty([train_labels.size, num_images])
    all_data_matrix_kmeans = np.empty([train_norm.shape[0], train_norm.shape[1], num_images])
    all_labels_matrix_kmeans = np.empty([train_labels.size, num_images])

    # Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')
    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub, train_slice, task)
        train_norm, _ = seg.normalize_data(train_data)
        Xpca, v, w, fraction_variance, ix = seg.mypca(train_norm)
        relevant_labels = int(np.sum(fraction_variance < 0.95)) + 1
        train_norm_ord = train_norm[:, ix]
        train_norm = train_norm_ord[:, :relevant_labels]
        all_data_matrix[:, :, i] = train_norm
        all_labels_matrix[:, i] = train_labels.flatten()

    # Load datasets for kmeans
    print('Loading data for ' + str(num_images) + ' subjects...')
    for i in all_subjects:
        sub = i + 1
        train_data_kmeans, train_labels_kmeans, train_feature_labels_kmeans = create_dataset(sub, train_slice, task)
        train_norm_kmeans, _ = seg.normalize_data(train_data_kmeans)
        all_data_matrix_kmeans[:, :, i] = train_norm_kmeans
        all_labels_matrix_kmeans[:, i] = train_labels_kmeans.flatten()

    print('Finished loading data.\nStarting segmentation...')

    # Go through each subject, taking i-th subject as the test
    for i in np.arange(num_images):
        sub = i+1

        # Define training subjects as all, except the test subject
        train_subjects = all_subjects.copy()
        train_subjects = np.delete(train_subjects, i)

        # Obtain data about the chosen amount of subjects
        train_data_matrix = all_data_matrix[:, :, train_subjects]
        train_labels_matrix = all_labels_matrix[:, train_subjects]
        test_data = all_data_matrix[:, :, i]
        test_labels = all_labels_matrix[:, i]
        test_shape_1 = test_labels.reshape(im_size[0], im_size[1])

        fig = plt.figure(figsize=(15, 5))

        # Get predicted labels from atlas method
        predicted_labels = seg.segmentation_combined_atlas(train_labels_matrix)
        all_errors[i, 0] = util.classification_error(test_labels, predicted_labels)
        all_dice[i, 0] = util.dice_multiclass(test_labels, predicted_labels)

        # Plot atlas method
        predicted_mask_1 = predicted_labels.reshape(im_size[0], im_size[1])
        ax1 = fig.add_subplot(151)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 0], all_dice[i, 0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined atlas'.format(sub))

        # Get predicted labels from kNN method
        predicted_labels = seg.segmentation_combined_knn(train_data_matrix, train_labels_matrix, test_data, k=10)
        all_errors[i, 1] = util.classification_error(test_labels, predicted_labels)
        all_dice[i, 1] = util.dice_multiclass(test_labels, predicted_labels)

        # Plot kNN method
        predicted_mask_2 = predicted_labels.reshape(im_size[0], im_size[1])
        ax2 = fig.add_subplot(152)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 1], all_dice[i, 1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: Combined k-NN'.format(sub))

        # Get predicted labels from my own method
        # all_data_matrix_bnb = np.empty([train_norm.shape[0], train_norm.shape[1], num_images])
        # all_labels_matrix_bnb = np.empty([train_labels.size, num_images])

        # for ii in all_subjects:
        #     sub = i + 1
        #     task = 'brain'
        #     train_data_bnb, train_labels_bnb, train_feature_labels_bnb = util.create_dataset(sub, train_slice, task)
        #     train_norm_bnb, _ = seg.normalize_data(train_data_bnb)
        #     Xpca, v, w, fraction_variance, ix = seg.mypca(train_norm_bnb)
        #     relevant_labels_bnb = int(np.sum(fraction_variance < 0.95)) + 1
        #     train_norm_ord_bnb = train_norm_bnb[:, ix]
        #     train_norm_bnb = train_norm_ord_bnb[:, :relevant_labels_bnb]
        #     all_data_matrix_bnb[:, :, ii] = train_norm_bnb
        #     all_labels_matrix_bnb[:, ii] = train_labels_bnb.flatten()
        #
        # qw, we, er = all_data_matrix.shape
        # for iii in np.arange(qw):
        #     for j in np.arange(er):
        #         if all_labels_matrix_bnb[iii, j] == 0:
        #             for k in np.arange(we):
        #                 all_data_matrix[iii, k, j] = 0

        # train_data_matrix = all_data_matrix[:, :, train_subjects]
        # test_data = all_data_matrix[:, :, i]

        train_data_matrix_kmeans = all_data_matrix_kmeans[:, :, train_subjects]
        train_labels_matrix_kmeans = all_labels_matrix[:, train_subjects]
        test_data_kmeans = all_data_matrix_kmeans[:, :, i]

        predicted_labels = segmentation_mymethod(train_data_matrix_kmeans, train_labels_matrix_kmeans, test_data_kmeans, task)
        all_errors[i, 2] = util.classification_error(test_labels, predicted_labels)
        all_dice[i, 2] = util.dice_multiclass(test_labels, predicted_labels)

        # Plot my own method
        predicted_mask_3 = predicted_labels.reshape(im_size[0], im_size[1])
        ax3 = fig.add_subplot(153)
        ax3.imshow(test_shape_1, 'gray')
        ax3.imshow(predicted_mask_3, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 2], all_dice[i, 2])
        ax3.set_xlabel(text_str)
        ax3.set_title('Subject {}: My method'.format(sub))

        ax4 = fig.add_subplot(154)
        ax4.imshow(predicted_mask_3, 'viridis')
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 2], all_dice[i, 2])
        ax4.set_xlabel(text_str)
        ax4.set_title('Subject {}: My method'.format(sub))

        ax5 = fig.add_subplot(155)
        ax5.imshow(test_shape_1, 'gray')
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 2], all_dice[i, 2])
        ax5.set_xlabel(text_str)
        ax5.set_title('Subject {}: My method'.format(sub))
