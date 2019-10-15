"""
Utility functions for segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import segmentation as seg
from scipy import ndimage


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    g = np.zeros(len(x))
    for k in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[k] = xp[k] + h/2
        xn[k] = xn[k] - h/2
        g[k] = (fun(xp)-fun(xn))/h

    return g


def scatter_data(X, Y, feature0=0, feature1=1, ax=None):
    # scatter_data displays a scatter-plot of at most 1000 samples from dataset X, and gives each point
    # a different color based on its label in Y.
    # Input:
    # X             - samples of image
    # Y             - labels
    # feature0      - first feature, normally 0
    # feature1      - second feature, normally 1
    # ax            - ?
    # Output:
    # ax            - scatter-plot data, in samples and features

    # visualization can only go up to 1000 pixels
    k = 1000
    if len(X) > k:
        idx = np.random.randint(len(X), size=k)
        X = X[idx, :]
        Y = Y[idx]

    # showing all labels and indices of the samples
    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)

    # choosing to plot or not
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()

    # an amount of colors are chosen for the labels
    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))

    # plotting the scatter of all samples X, with an added color and label from Y
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'X, class ' + str(i)
        ax.scatter(X[idx2, feature0], X[idx2, feature1], color=c, label=lbl)
        plt.xlabel('feature0')
        plt.ylabel('feature1')
        ax.legend()

    return ax


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
    Y = create_labels(image_number, slice_number, task)

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

    # addition of features T1 and T2 intensities
    t1f = t1.flatten().T.astype(float)
    t1f = t1f.reshape(-1, 1)
    t2f = t2.flatten().T.astype(float)
    t2f = t2f.reshape(-1, 1)
    X = np.concatenate((t1f, t2f), axis=1)

    features += ('T1 intensity',)
    features += ('T2 intensity',)

    # addition of feature coordinates
    c, _ = seg.extract_coordinate_feature(t1)
    X1 = np.concatenate((X, c), axis=1)
    features += ('coordinates',)

    # addition of feature T1 gaussian blurred
    t1_g = ndimage.gaussian_filter(t1, sigma=2)
    t1_gf = t1_g.flatten().T.astype(float)
    t1_gf = t1_gf.reshape(-1, 1)
    X2 = np.concatenate((X1, t1_gf), axis=1)
    features += ('guassian blurred (T1)',)

    # addition of feature T2 gaussian blurred
    t2_g = ndimage.gaussian_filter(t2, sigma=2)
    t2_gf = t2_g.flatten().T.astype(float)
    t2_gf = t2_gf.reshape(-1, 1)
    X3 = np.concatenate((X2, t2_gf), axis=1)
    features += ('guassian blurred (T2)',)

    # addition of laplacian kernel on T1
    k_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    Ic1 = ndimage.convolve(t1, k_laplace, mode='reflect')
    Ic1_f = Ic1.flatten().T.astype(float)
    Ic1_f = Ic1_f.reshape(-1, 1)
    X4 = np.concatenate((X3, Ic1_f), axis=1)
    features += ('laplacian kernel (T1)',)

    # addition of laplacian kernel on T2
    k_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    Ic2 = ndimage.convolve(t2, k_laplace, mode='reflect')
    Ic2_f = Ic2.flatten().T.astype(float)
    Ic2_f = Ic2_f.reshape(-1, 1)
    X5 = np.concatenate((X4, Ic2_f), axis=1)
    features += ('laplacian kernel (T2)',)

    return X5, features


def create_labels(image_number, slice_number, task):
    # Creates labels for a particular subject (image), slice and task

    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task         - String corresponding to the task, either 'brain' or 'tissue'

    # Output:
    # Y            - Nx1 vector with labels

    # Original labels reference:
    # 0 background
    # 1 cerebellum
    # 2 white matter hyperintensities/lesions
    # 3 basal ganglia and thalami
    # 4 ventricles
    # 5 white matter
    # 6 brainstem
    # 7 cortical grey matter
    # 8 cerebrospinal fluid in the extracerebral space

    # Read the ground-truth image
    global Y
    base_dir = '../data/dataset_brains/'

    I = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_gt.tif')

    if task == 'brain':
        Y = I > 0
    elif task == 'tissue':
        Y = I.copy()
        Y[I == 0] = 0
        Y[I == 1] = 0
        Y[I == 6] = 0
        Y[I == 2] = 1
        Y[I == 5] = 1
        Y[I == 7] = 2
        Y[I == 3] = 2
        Y[I == 4] = 3
        Y[I == 8] = 3
    else:
        print(task)
        raise ValueError("Variable 'task' must be one of two values: 'brain' or 'tissue'")

    Y = Y.flatten().T
    Y = Y.reshape(-1, 1)

    return Y


def dice_overlap(true_labels, predicted_labels, smooth=1.):
    # returns the Dice coefficient for two binary label vectors
    # Input:
    # true_labels         Nx1 binary vector with the true labels
    # predicted_labels    Nx1 binary vector with the predicted labels
    # smooth              smoothing factor that prevents division by zero
    # Output:
    # dice                Dice coefficient

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    # AB = 0
    # A = 0
    # B = 0
    # for i in range(len(t)):
    #     if t[i] == 1 and p[i] == 1:
    #         AB = AB + 1
    #     if t[i] == 1:
    #         A = A + 1
    #     if p[i] == 1:
    #         B = B + 1
    #
    # dice = 2 * AB / (A + B)

    dice = np.sum(p[t == 1]*2.0 / (np.sum(p) + np.sum(t)))

    return dice


def dice_multiclass(true_labels, predicted_labels):
    # dice_multiclass.m returns the Dice coefficient for two label vectors with
    # multiple classses
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # dice_score          Dice coefficient

    all_classes, indices1, indices2 = np.unique(true_labels, return_index=True, return_inverse=True)

    dice_score = np.empty((len(all_classes), 1))
    dice_score[:] = np.nan

    #Consider each class as the foreground class
    for i in np.arange(len(all_classes)):
        idx2 = indices2 == all_classes[i]
        lbl = 'X, class '+ str(all_classes[i])
        temp_true = true_labels.copy()
        temp_true[true_labels == all_classes[i]] = 1  #Class i is foreground
        temp_true[true_labels != all_classes[i]] = 0  #Everything else is background

        temp_predicted = predicted_labels.copy();
        temp_predicted[predicted_labels == all_classes[i]] = 1
        temp_predicted[predicted_labels != all_classes[i]] = 0
        dice_score[i] = dice_overlap(temp_true.astype(int), temp_predicted.astype(int))

    dice_score_mean = dice_score.mean()

    return dice_score_mean


def classification_error(true_labels, predicted_labels):
    # classification_error.m returns the classification error for two vectors
    # with labels
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # error         Classification error

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    error = np.sum(t != p) / len(t)

    # TP = 0
    # FN = 0
    # FP = 0
    # TN = 0
    # for i in range(len(t)):
    #     if t[i] == 1 and p[i] == 1:
    #         TP = TP + 1
    #     if t[i] == 1 and p[i] == 0:
    #         FN = FN + 1
    #     if t[i] == 0 and p[i] == 1:
    #         FP = FP + 1
    #     if t[i] == 0 and p[i] == 0:
    #         TN = TN + 1
    #
    # accuracy = (TP + TN) / (TP + FP + FN + TN)
    # error = 1-accuracy

    return error





