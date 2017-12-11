import numpy as np
import pyfftw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# The complete correlations use numpy to get the FFT of an array
# These are slow but are precise within machine precision
def autoCorrComplete(arr, n = 0):
    H1 = np.fft.fftn(arr == n)
    return np.fft.fftshift(np.fft.ifftn(np.conj(H1)*H1)).real

def crossCorrComplete(arr, n = 0, p = 1):
    H1 = np.fft.fftn(arr == n)
    H2 = np.fft.fftn(arr == p)
    return np.fft.fftshift(np.fft.ifftn(np.conj(H1)*H2)).real


# The wisdom correlations use pyfftw to get the FFT of an array
# These are much faster but are only precise within the definitions of np.close
# However, this is still very close to the real result
def autoCorrW(perStruc, n = 0):
    # Plan fft of a (real space array) into b (frequency space array)
    a = pyfftw.empty_aligned(perStruc.shape, dtype='complex128')
    a[:,:,:] = (perStruc == n).astype(int)
    fft_object = pyfftw.builders.fftn(a)
    # Save fft into b
    b = fft_object()
    del(a)
    b = np.conj(b)*b
    # Plan inverse fft of b (frequency space array)
    fft_object = pyfftw.builders.ifftn(b)
    # Return inverse fft of b as real space array
    return np.fft.fftshift(fft_object()).real

def crossCorrW(perStruc, n = 0, p = 1):
    a = pyfftw.empty_aligned(perStruc.shape, dtype='complex128')
    a[:,:,:] = (perStruc == n).astype(int)
    fft_object0 = pyfftw.builders.fftn(a)
    b = fft_object0()
    del(a)
    del(fft_object0)
    c = pyfftw.empty_aligned(perStruc.shape, dtype='complex128')
    c[:,:,:] = (perStruc == p).astype(int)
    fft_object1 = pyfftw.builders.fftn(c)
    d = fft_object1()
    del(c)
    del(fft_object1)
    b = np.conj(b)*d
    del(d)
    fft_object2 = pyfftw.builders.ifftn(b)
    return np.fft.fftshift(fft_object2()).real


# For the nstats- it is important that the center cell corresponds to the volume fraction
# Therefore, the following function makes sure that all axes are odd, meaning the center
# cell will correspond to volume fraction
def makeAxesOdd(struc):
    start = [0]*len(struc.shape)
    
    for i in range(len(struc.shape)):
        if struc.shape[i] % 2 == 0:
            start[i] = 1
    
    struc = struc[start[0]:struc.shape[0],start[1]:struc.shape[1],start[2]:struc.shape[2]]
            
    return struc


# To get nonperiodic statistics requires padding the array
def nonPPad(struc, value, padAxes = (0,1,2), cutoff = None):
    padDef = [(0,0)]*len(struc.shape)
    
    for i in padAxes:
        if cutoff is None:
            rad = struc.shape[i]
        else:
            rad = cutoff
        padDef[i] = (0, rad + 1)
        
    return np.pad(struc, pad_width = padDef, mode = 'constant', constant_values = value)


# Grab slice returns a select part of the stats centered around the midpoint
# In the following function rad takes the place of cutoff for better readability
# Midpoint is typically defined as (arr.shape[i]//2 + 1) for all axes i in an odd array
def grabSlice(arr, midP, cutoff):
    rad = cutoff
    del(cutoff)
    return arr[midP[0]-1-rad:midP[0]+rad,midP[1]-1-rad:midP[1]+rad,midP[2]-1-rad:midP[2]+rad]


# Print information about the stats
def printStatsInfo(stats):
    print('Shape:', stats.shape)
    print('Index of maximum:', np.argwhere(stats[:,:,:] == np.max(stats[:,:,:])))
    print('Maximum value:', np.max(stats[:,:,:]))
    print('Midpoint value:', stats[stats.shape[0]//2,stats.shape[1]//2,
                                   stats.shape[2]//2])

# Visualizing statistics is possible by showing 3d slices
def plot3DStats(gg, cutoff, alpha = 0.5, projection = '3D'):
    if projection is '3D':
        # create a 21 x 21 vertex mesh
        xx, yy = np.meshgrid(range(-cutoff,cutoff+1), range(-cutoff,cutoff+1))

        fig = plt.figure(figsize = (9,9))

        # show the 3D rotated projection
        ax0 = fig.add_subplot(111, projection='3d')
        cset00 = ax0.contourf(xx, yy, np.around(gg[cutoff + 1,:,:], 4), 100, zdir='z', alpha = alpha)
        cset01 = ax0.contourf(xx, np.around(gg[:,cutoff + 1,:], 4), yy, 100, zdir='y', alpha = alpha)
        cset02 = ax0.contourf(np.around(gg[:,:,cutoff + 1], 4), xx, yy, 100, zdir='x', alpha = alpha)

        ax0.set_xlabel('X')
        ax0.set_xlim3d(-cutoff,cutoff)
        ax0.set_ylim3d(-cutoff,cutoff)
        ax0.set_ylabel('Y')
        ax0.set_zlim3d(-cutoff,cutoff)
        ax0.set_zlabel('Z')


        plt.colorbar(cset00, ax = ax0, fraction = 0.03)

        plt.show()
        
    elif projection is '2D':
        fig = plt.figure(figsize = (8,16))

        ax0 = fig.add_subplot(131, projection='3d')
        ax0.imshow(gg[:,:,cutoff + 1])
        ax0.set_xlabel('X')
        ax0.set_xlim(-cutoff,cutoff)
        ax0.set_ylim(-cutoff,cutoff)
        ax0.set_ylabel('Y')
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.imshow(gg[:,cutoff + 1,:])
        ax1.set_xlabel('X')
        ax1.set_xlim(-cutoff,cutoff)
        ax1.set_ylim(-cutoff,cutoff)
        ax1.set_ylabel('Z')
        
        ax2 = fig.add_subplot(131, projection='3d')
        ax2.imshow(gg[cutoff + 1],:,:)
        ax2.set_xlabel('Y')
        ax2.set_xlim(-cutoff,cutoff)
        ax2.set_ylim(-cutoff,cutoff)
        ax2.set_ylabel('Z')
        
        
        plt.colorbar(cset00, ax = ax0, fraction = 0.03)

        plt.show()
        
    else:
        print('projection has to be either 3D or 2D')