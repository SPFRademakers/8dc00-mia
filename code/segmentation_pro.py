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


def cost_kmeans(X, w_vector):
    # Computes the cost of assigning data in X to clusters in w_vector
    # Input:
    # X         - data
    # w_vector  - means of clusters
    # Output:
    # J         - cost

    # Get the data dimensions
    n, m = X.shape

    # Number of clusters per feature
    K = int(len(w_vector)/m)

    # Reshape cluster centers into dataset format
    W = w_vector.reshape(K, m)

    D = scipy.spatial.distance.cdist(X, W, metric='euclidean')
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

    X_norm, _ = seg.normalize_data(test_data)
    N, M = X_norm.shape

    clusters = 4

    # link to the cost function of kMeans
    fun = lambda w: cost_kmeans(test_data, w)

    # the learning rate
    mu = 0.01

    # iterations
    num_iter = 100

    # Initialize cluster centers and store them in w_initial
    idx = np.random.randint(N, size=clusters)
    w_initial = X_norm[idx, :]

    # Reshape centers to a vector (needed by ngradient)
    w_vector = w_initial.reshape(K*M, 1)

    for i in np.arange(num_iter):
        # gradient ascent
        change = mu * util.ngradient(fun, w_vector)
        w_vector = w_vector - change[:, np.newaxis]

    # Reshape back to dataset
    w_final = w_vector.reshape(K, M)
    print(w_final.shape)
    print(w_final)

    # Find min_dist and min_index
    D = scipy.spatial.distance.cdist(test_data, w_final, metric='euclidean')
    min_index = np.argmin(D, axis=1)

    # Sort by intensity of cluster center
    sorted_order = np.argsort(w_final[:, 0], axis=0)

    # Update the cluster indices based on the sorted order and return results in predicted_labels
    predicted_labels = np.empty(*min_index.shape)
    predicted_labels[:] = np.nan

    for i in np.arange(len(sorted_order)):
        print(sorted_order[i])
        predicted_labels[min_index == sorted_order[i]] = i
        print(np.unique(predicted_labels))

    print("after loop")
    print(predicted_labels.shape)
    print(np.unique(predicted_labels))
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

    predicted_labels = kmeans_clustering(test_data, K=4)

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

    # find the predicted labels (here: the train_labels)
    predicted_labels = seg.segmentation_atlas(None, train_labels, None)

    # Calculate the error and dice score of these predicted labels in comparison to test labels
    err = util.classification_error(test_labels, predicted_labels)
    dice = util.dice_overlap(test_labels, predicted_labels)

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
    all_data_matrix = np.empty([train_data.shape[0], train_data.shape[1], num_images])
    # all_labels_matrix = np.empty([train_labels.size, num_images], dtype=bool)
    all_labels_matrix = np.empty([train_labels.size, num_images])

    # Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')

    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub, train_slice, task)
        all_data_matrix[:, :, i] = train_data
        all_labels_matrix[:, i] = train_labels.flatten()

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
        all_dice[i, 0] = util.dice_overlap(test_labels, predicted_labels)

        # Plot atlas method
        predicted_mask_1 = predicted_labels.reshape(im_size[0], im_size[1])
        ax1 = fig.add_subplot(131)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 0], all_dice[i, 0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined atlas'.format(sub))

        # Get predicted labels from kNN method
        predicted_labels = seg.segmentation_combined_knn(train_data_matrix, train_labels_matrix, test_data)
        all_errors[i, 1] = util.classification_error(test_labels, predicted_labels)
        all_dice[i, 1] = util.dice_overlap(test_labels, predicted_labels)

        # Plot kNN method
        predicted_mask_2 = predicted_labels.reshape(im_size[0], im_size[1])
        ax2 = fig.add_subplot(132)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 1], all_dice[i, 1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: Combined k-NN'.format(sub))

        # Get predicted labels from my own method
        predicted_labels = segmentation_mymethod(train_data_matrix, train_labels_matrix, test_data, task)
        print(predicted_labels.shape)
        print(np.unique(predicted_labels))
        all_errors[i, 2] = util.classification_error(test_labels, predicted_labels)
        all_dice[i, 2] = util.dice_overlap(test_labels, predicted_labels)

        # Plot my own method
        predicted_mask_3 = predicted_labels.reshape(im_size[0], im_size[1])
        ax3 = fig.add_subplot(133)
        ax3.imshow(test_shape_1, 'gray')
        ax3.imshow(predicted_mask_3, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i, 2], all_dice[i, 2])
        ax3.set_xlabel(text_str)
        ax3.set_title('Subject {}: My method'.format(sub))
