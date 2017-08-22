from PIL import Image
import numpy as np

# generate test image
# ------------------------------------------------------- #

# init
data = np.zeros((512,512))
noise = np.zeros((512,512))

# intensity/noise variance
I_1 = 60
I_2 = 110
I_3 = 160
I_4 = 210

sigma = 0.2

# pixel intensity
data[0:256,0:256] = I_1*np.ones((256,256))
data[0:256,256:] = I_2*np.ones((256,256))
data[256:,0:256] = I_3*np.ones((256,256))
data[256:,256:] = I_4*np.ones((256,256))

# speckle noise
noise[0:256,0:256] = np.random.normal(loc=0,scale=sigma,size=(256,256))
noise[0:256,256:] = np.random.normal(loc=0,scale=sigma,size=(256,256))
noise[256:,0:256] = np.random.normal(loc=0,scale=sigma,size=(256,256))
noise[256:,256:] = np.random.normal(loc=0,scale=sigma,size=(256,256))

img_array = data + np.multiply(data,noise)

# save image
# ------------------------------------------------------- #

img = Image.fromarray(img_array.astype('uint8'))
img.save('test_image.png')
