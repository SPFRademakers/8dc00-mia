"""
Registration project code.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output 
import scipy.optimize as opt

def joint_histogram(I, J, num_bins=16, minmax_range=None):
    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")
    p, xedge, yedge = np.histogram2d(I.ravel(), J.ravel(), num_bins)
    p = p/np.sum(p)

    return p

def rigid_corr(I, Im, x, MI):
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
    T = reg.rotate(x[0])
    Th = util.t2h(T, x[1:]*SCALING)

    # transform the moving image
    Im_t, Xt = reg.image_transform(Im, Th)

    # compute the similarity between the fixed and transformed moving image
    if MI:
        p = joint_histogram(I, Im_t)
        S = reg.mutual_information(p)
    else:
        S = reg.correlation(I, Im_t)

    return S, Im_t, Th

def optim_rigid_corr(x, I, Im, MI):
    # Identical to rigid_corr in functionality but the x variable is placed
    # in front when calling for use in scipy.optimize function.
    # Output is changed to only give the similarity because scipy.optimize
    # uses this metric to optimize the parameters

    SCALING = 100

    # the first element is the rotation angle
    T = reg.rotate(x[0])
    Th = util.t2h(T, x[1:]*SCALING)

    # transform the moving image
    Im_t, Xt = reg.image_transform(Im, Th)

    # compute the similarity between the fixed and transformed moving image
    if MI:
        p = joint_histogram(I, Im_t)
        S = reg.mutual_information(p)
    else:
        S = reg.correlation(I, Im_t)

    return S


def affine_corr(I, Im, x, MI):
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

    Tro = reg.rotate(x[0])
    Tsc = reg.scale(x[1], x[2])
    Tsh = reg.shear(x[3], x[4])
    Trss = Tro.dot(Tsc).dot(Tsh)
    Th = util.t2h(Trss, [x[5], x[6]])

    Im_t, Xt = reg.image_transform(Im, Th)

    if MI:
        p = joint_histogram(I, Im_t)
        S = reg.mutual_information(p)
    else:
        S = reg.correlation(I, Im_t)

    return S, Im_t, Th

def optim_affine_corr(x, I, Im, MI):
    # Identical to affine_corr in functionality but the x variable is placed
    # in front when calling for use in scipy.optimize function.
    # Output is changed to only give the similarity because scipy.optimize
    # uses this metric to optimize the parameters

    Tro = reg.rotate(x[0])
    Tsc = reg.scale(x[1], x[2])
    Tsh = reg.shear(x[3], x[4])
    Trss = Tro.dot(Tsc).dot(Tsh)
    Th = util.t2h(Trss, [x[5], x[6]])

    Im_t, Xt = reg.image_transform(Im, Th)

    if MI:
        p = joint_histogram(I, Im_t)
        S = reg.mutual_information(p)
    else:
        S = reg.correlation(I, Im_t)

    return S

def intensity_based_registration(I, Im, mu=0.0005, num_iter=200, Affine=True, MI=True, Gradient=False):
    # Performs fully automatic intensity based image registration, providing
    # options for Rigid and Affine transformation, 
    # options for Cross-correlation (CC) and Mutual Information (MI) as similarity metrics,
    # and options for Gradient ascent and Nelder-Mead as optimization algorithms.
    # Input:
    # I - Fixed image
    # Im - Moving image
    # mu - step size for Gradient ascent algorithm
    # num_iter - number of iterations for Gradient ascent algorithm
    # Affine - transformation method (standard = Affine)
    # MI - similarity metric (standard = MI)
    # Gradient - optimization algorithm (standard = Nelder-Mead)
    # Output:
    # similarity - final similarity of the images
    # time_elapsed - time elapsed for image registration
    # Im_t - registered image
    
    start_time = time.clock()

    if Affine:
        x = np.array([0., 1., 1., 0., 0., 0., 0.])
        fun = lambda x: affine_corr(I, Im, x, MI)
    else:
        x = np.array([0., 0., 0.])
        fun = lambda x: rigid_corr(I, Im, x, MI)

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14, 6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt1 = ax1.text(0.05, 0.95, " ",
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    txt2 = ax2.text(1.48, 0.18," ",
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)

    if Gradient:
        # perform 'num_iter' gradient ascent updates
        for k in np.arange(num_iter):

            # gradient ascent
            g = reg.ngradient(fun, x)
            x += g*mu
        
            #for visualization of the result
    
            S, Im_t, _ = fun(x)
            clear_output(wait = True)

            current_time = time.clock() - start_time

            # update moving image and parameters
            im2.set_data(Im_t)
            txt1.set_text(np.array2string(x, precision=5, floatmode='fixed'))
            txt2.set_text("mu = " + str(mu) + ", num_iter = " + str(num_iter) + ", time = " + str(round(current_time, 2)))

            # update 'learning' curve
            similarity[k] = S
            learning_curve.set_ydata(similarity)
            
            display(fig)
            
            time_elapsed = (time.clock() - start_time)

        if Affine:
            if MI:
                print("MI = " + str(similarity[-1]) + " for the affine intensity-based registration after " + str(num_iter) + " iterations with a computation time of " + str(time_elapsed))
            else:
                print("CC = " + str(similarity[-1]) + " for the affine intensity-based registration after " + str(num_iter) + " iterations with a computation time of " + str(time_elapsed))
        else:
            if MI:
                print("MI = " + str(similarity[-1]) + " for the rigid intensity-based registration after " + str(num_iter) + " iterations with a computation time of " + str(time_elapsed))
            else:
                print("CC = " + str(similarity[-1]) + " for the rigid intensity-based registration after " + str(num_iter) + " iterations with a computation time of " + str(time_elapsed))

        return similarity[-1], time_elapsed, Im_t
            
    else:    
        x0 = x
        # Optimizing the parameters using nelder-mead
        if Affine:
            res = opt.minimize(lambda x: -optim_affine_corr(x, I, Im, MI), x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': False})
        else:
            res = opt.minimize(lambda x: -optim_rigid_corr(x, I, Im, MI), x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': False})
        
        x = res.x
        S, Im_t, _ = fun(x)
        clear_output(wait = True)

        current_time = time.clock() - start_time

        # update moving image and parameters
        im2.set_data(Im_t)
        txt1.set_text(np.array2string(x, precision=5, floatmode='fixed'))
        txt2.set_text("mu = " + str(mu) + ", num_iter = " + str(res.nit) + ", time = " + str(round(current_time, 2)))

        # update 'learning' curve
        similarity = S
        learning_curve.set_ydata(similarity)

        display(fig)

        time_elapsed = (time.clock() - start_time)
        # print the performance of the function
        if Affine:
            if MI:
                print("MI = " + str(similarity) + " for the affine intensity-based registration after " + str(res.nit) + " iterations with a computation time of " + str(time_elapsed))
            else:
                print("CC = " + str(similarity) + " for the affine intensity-based registration after " + str(res.nit) + " iterations with a computation time of " + str(time_elapsed))
        else:
            if MI:
                print("MI = " + str(similarity) + " for the rigid intensity-based registration after " + str(res.nit) + " iterations with a computation time of " + str(time_elapsed))
            else:
                print("CC = " + str(similarity) + " for the rigid intensity-based registration after " + str(res.nit) + " iterations with a computation time of " + str(time_elapsed))

        return similarity, time_elapsed, Im_t