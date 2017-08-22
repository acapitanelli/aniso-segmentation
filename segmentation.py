from filtering import imfilter
import numpy as np


def run(img,num_classes=4):

    map_segmented,post_prob = map_segmentation(img,num_classes)

    anisotropic_segmented = anisotropic_segmentation(img,post_prob,num_classes)

    return map_segmented,anisotropic_segmented


def map_segmentation(img,num_classes):

    np.seterr(all='raise')

    # output class matrix
    height,width = img.shape
    num_pixels = height*width

    # error tolerance
    err = 10**-3;

    # max iterations
    max_iters = 5000

    # vectorize data
    data = img.ravel()

    # intensity statistics
    min_I = np.min(data)
    max_I = np.max(data);
    int_I = (max_I-min_I)/num_classes;

    # initial estimation
    class_mean = np.zeros((num_classes,max_iters))
    class_var = np.zeros((num_classes,max_iters))

    for ii in range(num_classes):

        start_I = min_I + ii*int_I
        end_I = start_I + int_I

        # mean value init
        class_mean[ii,0] = (start_I+end_I)/2

        # variance init
        class_var[ii,0] = 10

    # prior probability for pixel belonging to each class
    prob_ante = 1/num_classes*np.ones(num_classes)

    # posterior probability for pixel belonging to each class
    prob_post = np.zeros((num_classes,num_pixels))

    next_iter = True
    curr_iter = 0
    while next_iter:

        for ii in range(num_classes):

            est_mean = class_mean[ii,curr_iter]
            est_var = class_var[ii,curr_iter]

            try:
                norm_pdf = np.exp(-(data-est_mean)**2/(2*est_var))/np.sqrt(2*np.pi*est_var)

            except FloatingPointError:

                norm_pdf = np.zeros(num_pixels)
                for jj in range(num_pixels):

                    try:
                        norm_pdf[jj] = np.exp(-(data[jj]-est_mean)**2/(2*est_var))/np.sqrt(2*np.pi*est_var)

                    except FloatingPointError:
                        pass

            prob_post[ii,:] = prob_ante[ii]*norm_pdf

        # normalization
        post_prob = np.divide(prob_post,np.tile(np.sum(prob_post,axis=0),(num_classes,1)))

        # pixel assignment to classes with MAP criterion
        class_idx = np.argmax(post_prob,axis=0)

        # ML estimation of class parameters
        for ii in range(num_classes):

            class_pixels = data[class_idx==ii]

            class_mean[ii,curr_iter+1] = np.mean(class_pixels)
            class_var[ii,curr_iter+1] = np.var(class_pixels)

        # next iteration
        curr_iter += 1

        # exit condition
        if curr_iter>1:

            mean_err = abs(class_mean[:,curr_iter]-class_mean[:,curr_iter-1])
            var_err = abs(class_var[:,curr_iter]-class_var[:,curr_iter-1])

            if (curr_iter>max_iters) or (max(mean_err)<err and  max(var_err)<err):
                next_iter = False

    segmented_data = np.reshape(class_idx,(height,width))

    return segmented_data,post_prob


def anisotropic_segmentation(img,post_prob,num_classes=4):
    """ Adapted from work of P.D. Kovesi. Matlab code,
        see [http://www.peterkovesi.com/matlabfns/]
    """
    height,width = img.shape

    prob_matrix = np.zeros((height,width,num_classes))

    for ii in range(num_classes):

        prob_matrix[:,:,ii] = np.reshape(post_prob[ii,:],(height,width))

    # number of iterations
    num_iters = 9

    # scale factor
    T = 1/8

    # conduction factor k
    kappa = 10

    # convolution kernels
    hN = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    hS = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    hE = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    hW = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    hNE = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]])
    hSE = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
    hSW = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])
    hNW = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

    # pixel distance
    dx = 1;
    dy = 1;
    dd = np.sqrt(2);

    # iterative evalution of PDE
    for ii in range(num_iters):
        for kk in range(num_classes):

            P = prob_matrix[:,:,kk]

            # nabla evaluation on neighborhood
            nablaN = imfilter(P,hN,mode='conv')
            nablaS = imfilter(P,hS,mode='conv')
            nablaW = imfilter(P,hW,mode='conv')
            nablaE = imfilter(P,hE,mode='conv')
            nablaNE = imfilter(P,hNE,mode='conv')
            nablaSE = imfilter(P,hSE,mode='conv')
            nablaSW = imfilter(P,hSW,mode='conv')
            nablaNW = imfilter(P,hNW,mode='conv')

            # diffusion functions
            cN = np.exp(-np.square(nablaN/kappa))
            cS = np.exp(-np.square(nablaS/kappa))
            cW = np.exp(-np.square(nablaW/kappa))
            cE = np.exp(-np.square(nablaE/kappa))
            cNE = np.exp(-np.square(nablaNE/kappa))
            cSE = np.exp(-np.square(nablaSE/kappa))
            cSW = np.exp(-np.square(nablaSW/kappa))
            cNW = np.exp(-np.square(nablaNW/kappa))

            # update of posterior probability
            P += T*(
                    (1/np.square(dy))*np.multiply(cN,nablaN) + (1/np.square(dy))*np.multiply(cS,nablaS) +
                    (1/np.square(dx))*np.multiply(cW,nablaW) + (1/np.square(dx))*np.multiply(cE,nablaE) +
                    (1/np.square(dd))*np.multiply(cNE,nablaNE) + (1/np.square(dd))*np.multiply(cSE,nablaSE) +
                    (1/np.square(dd))*np.multiply(cSW,nablaSW) + (1/np.square(dd))*np.multiply(cNW,nablaNW)
                )

            prob_matrix[:,:,kk] = P

    # pixel assignment to classes with MAP criterion
    segmented_data = np.argmax(prob_matrix,axis=2)

    return segmented_data
