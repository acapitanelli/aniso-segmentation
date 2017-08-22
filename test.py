import matplotlib.pyplot as plt
from PIL import Image
import segmentation
from os import path
import numpy as np
# read image
# ------------------------------------------------------- #
img = Image.open( path.join('test','test_image.png') )

pixels = img.load()
width, height = img.size

data = np.zeros((width,height),dtype='uint8')
for ii in range(width):
    for jj in range(height):
        data[jj,ii] = pixels[ii,jj]

fig, ax = plt.subplots()
ax.imshow(data, extent=[0,width-1,0,height-1])
ax.set_title('original image')

plt.savefig(path.join('test','original.png'))

# image segmentation
# ------------------------------------------------------- #

map_segmented,anisotropic_segmented = segmentation.run(data,num_classes=4)


# show results
# ------------------------------------------------------- #

fig1, ax1 = plt.subplots()
ax1.imshow(map_segmented, extent=[0,width-1,0,height-1])
ax1.set_title('map segmentation')

plt.savefig(path.join('test','map.png'))

fig2, ax2 = plt.subplots()
ax2.imshow(anisotropic_segmented, extent=[0,width-1,0,height-1])
ax2.set_title('map/anisotropic segmentation')

plt.savefig(path.join('test','anisotropic.png'))

#plt.tight_layout()
#plt.show()
