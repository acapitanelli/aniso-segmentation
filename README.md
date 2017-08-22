# aniso-segmentation #

A python-coded algorithm which uses **MAP classification** and **anisotropic heat diffusion equation** for image segmentation of speckled images.


### references ###

This very simple implementation is based on the work of Gui Gao et al., *A segmentation algorithm for SAR images based on the anisotropic heat diffusion equation (2008)*. Implementation of kernel convolution for the evaluation of anisotropic equations is adapted from P.D. Kovesi [matlab implementation](http://www.peterkovesi.com/matlabfns/).


### main concepts ###

Segmentation works with gray images and is divided into two stages:

* **coarse segmentation**: an iterative maximum-a-posteriori (MAP) estimation is performed in order to classify image pixels
* **fine segmentation**: anisotropic heat diffusion model is used for despeckling and smoothing the posterior probability matrixes (for details see  P.Perona and J.Malik, *Scale-Space and Edge Detection Using Anisotropic Diffusion (1990)*)

This approach was originally conceived for the elaboration of SAR (Synthetic Aperture Radar) images. In that context, valid distributions for pixel intensity include the negative exponential distribution or, more generally, the gamma distribution.

Dealing with ordinary picture, here pixel intensity is assumed to follow a normal distribution, so that evaluation of PDF and maximum-likelihood estimation can be easily handled with no need for external libraries (such as scipy).


### example ###

A test image with four different textures is generated, then moltiplicative gaussian noise (speckle) is inserted.

[Test Image](test/test_image.png)

Then just type:

> \>>> python test.py


Result of segmentation will be saved in `\test` folder.

[Coarse segmented](test/map.png)

[Fine segmented](test/anisotropic.png)
