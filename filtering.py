import numpy as np

def imfilter_old(im,kernel,mode='corr'):

    height,width = im.shape

    # add check for even-sized kernels
    num_rows,num_cols = kernel.shape

    if mode == 'conv':
        kernel = np.flipud(np.fliplr(kernel))

    anchor_r = int((num_rows-1)/2)
    anchor_c = int((num_cols-1)/2)

    filtered_im = np.zeros((height,width))

    for ii in range(height):
        for jj in range(width):

            x = 0

            for nn in range(num_rows):
                for mm in range(num_cols):

                    loc_r = ii+nn-anchor_r
                    loc_c = jj+mm-anchor_c

                    if (loc_r>=0 and loc_r<height) and (loc_c>=0 and loc_c<width):
                        x += kernel[nn,mm] * im[loc_r,loc_c]

            filtered_im[ii,jj] = x

    return filtered_im



def imfilter(im,kernel,mode='corr'):

    if mode not in ['corr','conv']:
        raise ValueError('mode can be "corr" or "conv"')

    num_rows,num_cols = kernel.shape

    if num_rows%2 == 0 or num_cols%2 == 0:
        raise ValueError('kernel must be odd-sized')

    height,width = im.shape

    if mode == 'conv':
        kernel = np.flipud(np.fliplr(kernel))

    anchor_r = int((num_rows-1)/2)
    anchor_c = int((num_cols-1)/2)

    # image zero padding
    height_padded = height+2*anchor_r
    width_padded = width+2*anchor_c
    im_padded = np.zeros( (height_padded,width_padded) )
    im_padded[anchor_r:anchor_r+height,anchor_c:anchor_c+width] = im

    # fill temporary matrix
    tmp_matrix = np.zeros( (num_rows*num_cols, height*width) )

    for ii in range(anchor_r,height+anchor_r):
        for jj in range(anchor_c,width+anchor_c):

            data_block = im_padded[ii-anchor_r:ii+anchor_r+1,jj-anchor_c:jj+anchor_c+1]

            idx = (jj-anchor_c) + (ii-anchor_r)*width
            tmp_matrix[:,idx] = data_block.ravel()

    # fill kernel matrix
    tmp_kernel = np.tile(kernel.reshape(num_rows*num_cols,1),(1,height*width))

    # correlation/convolution
    prodsum = np.sum(np.multiply(tmp_matrix,tmp_kernel),axis=0)

    segmented_data = np.reshape(prodsum,(height,width))

    return segmented_data
