"""
Registration project code.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output


def rigid_corr(I, Im, x, MI = True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with a rigid transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle and the remaining two elements
    #     are the translation
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)

    SCALING = 100

    # the first element is the rotation angle
    T = rotate(x[0])

    # the remaining two element are the translation
    #
    # the gradient ascent/descent method work best when all parameters
    # of the function have approximately the same range of values
    # this is  not the case for the parameters of rigid registration
    # where the transformation matrix usually takes  much smaller
    # values compared to the translation vector this is why we pass a
    # scaled down version of the translation vector to this function
    # and then scale it up when computing the transformation matrix
    Th = util.t2h(T, x[1:]*SCALING)

    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed moving image
    # ADD CODE TO DIFFERENTIATE BETWEEN C AND MI
    C = correlation(I, Im_t)

    return C, Im_t, Th


def affine_corr(I, Im, x, MI = True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)

    NUM_BINS = 64
    SCALING = 100

    Tro = rotate(x[0])
    Tsc = scale(x[1], x[2])
    Tsh = shear(x[3], x[4])
    Trss = Tro.dot(Tsc).dot(Tsh)
    Th = util.t2h(Trss, [x[5], x[6]])

    Im_t, Xt = image_transform(Im, Th)

    # ADD CODE TO DIFFERENTIATE BETWEEN C AND MI
    C = correlation(I, Im_t)

    return C, Im_t, Th


def intensity_based_registration(I, Im, Affine = True, MI = True):

    # calling whether to use affine or rigid-based transformation
    if Affine:
        x = np.array([0., 1., 1., 0., 0., 0., 0.])
        # don't know for sure if there needs to be an if statement here or
        # in the affine_corr rigid_corr functions
        if MI:
            fun = lambda x: reg.affine_corr(I, Im, x, MI)
        else:
            fun = lambda x: reg.affine_corr(I, Im, x, MI)
    else:
        x = np.array([0., 0., 0.])
        fun = lambda x: reg.rigid_corr(I, Im, x)

    # the learning rate
    mu = 0.0005  # making code to make it best?

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)

    # change this line to MI or CC
    if MI:
      p = reg.joint_histogram(I, Im)
      similarity = reg.mutual_information(p)
    else:
      similarity = reg.correlation(I, Im)

    # similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14, 6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = fun(x)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
