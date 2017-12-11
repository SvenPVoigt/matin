import os
import numpy as np
import math
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from scipy.ndimage.morphology import binary_erosion
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def ComTomLabelPoreMatrixSurface(image, minVal = 0, maxVal = 255):
    import numpy as np
    from scipy import ndimage
    from scipy.ndimage.morphology import binary_erosion
    
    # See "template_masking_ver1 Developing a Function" for documentation
    label_im, nb_labels = ndimage.label(image==minVal)
    sizes = ndimage.sum((image==minVal), label_im, range(nb_labels + 1))
    image = (image == maxVal).astype(np.uint8)
    image[label_im==np.argmax(sizes)] = 2
    image[binary_erosion(image==2, border_value=1)] = 3
    
    return image

def showSlices(struc, xInd, yInd, zInd):
    fig = plt.figure(figsize=(16,8));
    a = fig.add_subplot(1,3,1)
    a.imshow(struc[:,:,zInd])
    b = fig.add_subplot(1,3,2)
    b.imshow(struc[:,yInd,:])
    c = fig.add_subplot(1,3,3)
    c.imshow(struc[xInd,:,:])
    plt.show(fig)

def getGaugeIndex(array3d, minLev = 0.75, cross_axes = (1,2)):
    # see 'template_segmentation_ver2 Making a Function' for documentation
    hProf = array3d.mean(axis = cross_axes)
    x = np.where(hProf<np.mean(sorted(hProf)[0:round(len(hProf)*minLev)]))[0]
    return [x[0], x[-1]]

def getAngle(array3d, topStartInd, botEndInd, axis_rot = 0):
    # see 'template_segmentation_ver2 Making a Function' for documentation

    topProf = array3d[topStartInd,:,:].mean(axis = axis_rot)
    topProfInd = [j for (i,j) in zip(topProf,np.arange(len(topProf))) if i > np.mean(topProf)]
    botProf = array3d[botEndInd,:,:].mean(axis = axis_rot)
    botProfInd = [j for (i,j) in zip(botProf,np.arange(len(botProf))) if i > np.mean(botProf)]
    rise = botEndInd-topStartInd
    run = np.mean(topProfInd) - np.mean(botProfInd)

    return math.degrees(math.atan(run/rise))

def rotate3dArrayAboutAxis0(array3d, theta):
    # see 'template_segmentation_ver2 Making a Function' for documentation
    arrayRot = rotate(array3d[:,0,:], angle = theta)
    arrayRot = np.expand_dims(arrayRot, axis = 1)

    for i in np.arange(1,array3d.shape[1]):
        sliceRot = rotate(array3d[:,i,:], angle = theta)
        sliceRot = np.expand_dims(sliceRot, axis = 1)
        arrayRot = np.concatenate((arrayRot,sliceRot), axis = 1)

    return arrayRot

def degauss(imgBlur, sigma = 1, laplacian = 30):
    # see 'template_segmentation_ver2 Making a Function' for documentation
    filGaus = ndimage.gaussian_filter(imgBlur, sigma)
    sharpGaus = imgBlur + laplacian * (imgBlur - filGaus)

    return sharpGaus > threshold_otsu(sharpGaus)


