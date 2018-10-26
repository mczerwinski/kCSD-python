#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:42:54 2018

@author: mczerwinski
"""

import numpy as np
import scipy.signal as ss
from matplotlib import pyplot as plt
from copy import deepcopy as cp
import os
loc = os.getcwd()

#os.chdir("/home/mczerwinski/opt/git_kCSD/kCSD-python/corelib")
#from KCSD import KCSD1D, KCSD2D

def gauss2d(axis, p):
    """
     p:     list of parameters of the Gauss-function
            [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE]
            SIGMA = FWHM / (2*sqrt(2*log(2)))
            ANGLE = rotation of the X,Y direction of the Gaussian in radians
    Returns
    -------
    the value of the Gaussian described by the parameters p
    axis is the positions (x,y)
    """
    x,y = axis
    x = x.reshape(x.size, 1)
    y = y.reshape(1, x.size)
    rcen_x = p[0] * np.cos(p[5]) - p[1] * np.sin(p[5])
    rcen_y = p[0] * np.sin(p[5]) + p[1] * np.cos(p[5])
    xp = x * np.cos(p[5]) - y * np.sin(p[5])
    yp = x * np.sin(p[5]) + y * np.cos(p[5])
    g = p[4]*np.exp(-(((rcen_x-xp)/p[2])**2+
                      ((rcen_y-yp)/p[3])**2)/2.)
    g = g/np.max(abs(g))
    return g

def gauss(axis, pos, std):
    '''
    axis is a numpy array or a tuple
    '''
    _gauss = 1./(np.sqrt(2*np.pi)*std)*np.e**(-0.5*((axis-pos)/std)**2)

    #_gauss = _gauss/np.max(abs(_gauss))
    return _gauss

def normalize(data):
    data/= np.max(abs(data))
    #data -= np.sum(data)/np.size(data)
    return data

def weight_width(x):
    '''
    transforms the amplitude of the gaussian
    '''
    y = 1./(1.5+np.exp(-10*(x-0.5)))+0.1
    return y

def SingleGauss(counter, dim=1, placement=(0.5, 0.5, 0.5), libsize='medium', resolution=100, ReturnMaxSize=False):
    '''
    single gauss
    placement is a tuple, with placement.size>=dim
    '''
    axis = np.linspace(-0.5, 1.5, resolution*2)
    if libsize is 'large':
        sizes1 = np.arange(0.01, 0.2,0.01)
        sizes2 = np.arange(0.2, 0.81, 0.05)
        sizes = np.hstack((sizes1[:-1], sizes2))
    elif libsize is 'medium':
        sizes1 = np.arange(0.01, 0.2,0.05)
        sizes2 = np.arange(0.2, 0.81, 0.1)
        sizes = np.hstack((sizes1, sizes2))
    elif libsize is 'small':
        sizes1 = np.arange(0.01, 0.2,0.05)
        sizes2 = np.arange(0.2, 0.81, 0.25)
        sizes = np.hstack((sizes1, sizes2))
    if counter>np.size(sizes)-1:
        return False
    if dim==1:
        source = gauss(axis, placement[0], sizes[counter])
    elif dim==2:
        # [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE]
        source =gauss2d((axis, axis), [placement[0], placement[1], sizes[counter], sizes[counter], 1, 0])

    source = normalize(source)
    if ReturnMaxSize:
        return source, np.size(sizes)
    else:
        return source

def dipol(counter, dim=1,same_sign=True, libsize='medium', resolution=100, ReturnMaxSize=False):
    '''
    twice less sizes than the single gauss
    returns the csd from -0.5 to 1.5 in resolution*2
    if ReturnMaxSize==True: will return the max iterations it will give, if counter-1>ReturnMaxSize it will return False
    '''
    if libsize is 'small':
        distances = np.hstack((np.arange(0.1,0.51,0.15), np.array([.666, 0.8333])))
        distances = np.arange(0.1, 1.51,0.2)
        libsize_1gauss = 'small'
    if libsize is 'medium':
        distances = np.hstack((np.arange(0.05,0.51,0.05)[:-1], np.arange(0.5,1.01,0.25)))
        libsize_1gauss = 'small'
    if libsize is 'large':
        distances = np.hstack((np.arange(0.01,0.51,0.05)[:-1], np.arange(0.5,1.01,0.1)))
        libsize_1gauss = 'medium'

    if libsize_1gauss is 'small':
        sizes1 = np.arange(0.05, 0.12,0.05)
        sizes2 = np.arange(0.1, 0.61, 0.25)
        sizes = np.hstack((sizes1[:-1], sizes2))

    if libsize_1gauss is 'medium':
        sizes1 = np.arange(0.05, 0.2,0.05)
        sizes2 = np.arange(0.2, 0.61, 0.1)
        sizes = np.hstack((sizes1, sizes2))

    size_index = counter//np.size(distances)
    distance_index = counter%np.size(distances)

    axis = np.linspace(-0.5, 1.5, resolution*2)

    if dim==1:
        pos1 = 0.5+distances[distance_index]
        pos2 = 0.5
    elif dim==2:
        pos1 = (0.5, 0.5+distances[distance_index])
        pos2 = (0.5, 0.5)
        par1 = [ pos1[0], pos1[1], sizes[size_index], sizes[size_index], 1, 0]
        par2 = [ pos2[0], pos2[1], sizes[size_index], sizes[size_index], 1, 0]

        # [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE]
    if same_sign==True:
        a = 1
    else:
        a = -1

    if dim==1:
        source = gauss(axis, pos1, sizes[size_index])+a*gauss(axis, pos2, sizes[size_index])
    elif dim==2:
        source = gauss2d((axis, axis), par1)+a*gauss2d((axis, axis), par2)

    source = normalize(source)
    if ReturnMaxSize:
        counter_max = np.size(distances)*np.size(sizes)
        return source, counter_max
    else:
        return source


def tripole(counter, libsize='medium', dim=1, resolution=100, ReturnMaxSize=False, sign_way=1):
    '''
    sign_way can be 1 or 2
    'large' and 'medium' are the same
    '''
    if libsize is 'small':
        distances = np.arange(0.1,0.51,0.1)
        libsize_1gauss = 'small'
    if libsize is 'medium':
        distances = np.hstack((np.arange(0.05,0.61,0.05)[:-1], np.arange(0.5,1.01,0.25)))
        libsize_1gauss = 'small'
    if libsize is 'large':
        distances = np.hstack((np.arange(0.01,0.61,0.05)[:-1], np.arange(0.5,1.01,0.1)))
        libsize_1gauss = 'medium'

    # sizes from SingleGauss, but shorter
    if libsize_1gauss is 'medium':
        sizes = np.arange(0.02, 0.42,0.1)
        #sizes2 = np.arange(0.2, 0.61, 0.1)
        #sizes = np.hstack((sizes1, sizes2))
    elif libsize_1gauss is 'small':
        sizes = np.arange(0.02, 0.61,0.05)
        #sizes2 = np.arange(0.40, 0.61, 0.02)
        #sizes = np.hstack((sizes1, sizes2))
    elif libsize_1gauss=='large':
        sizes = np.arange(0.02, 0.61,0.05)

    size_index = counter//np.size(distances)
    distance_index = counter%np.size(distances)

    axis = np.linspace(-0.5, 1.5, resolution*2)

    if dim==1:
        pos1 = 0.5+distances[distance_index]
        pos2 = 0.5
        pos3 = 0.5-distances[distance_index]
    elif dim==2:
        pos1 = 0.5+distances[distance_index]
        pos2 = 0.5
        pos3 = 0.5-distances[distance_index]
        xpos = 0.5
        par1 = [ xpos, pos1, sizes[size_index], sizes[size_index], 1, 0]
        par2 = [ xpos, pos2, sizes[size_index], sizes[size_index], 1, 0]
        par3 = [ xpos, pos3, sizes[size_index], sizes[size_index], 1, 0]

    if sign_way ==1:
        a,b,c = 1, 1, -1
    elif sign_way==2:
        a,b,c = 1, -1, 1

    if dim==1:
        source = a*gauss(axis, pos1, sizes[size_index])+ b*gauss(axis, pos2, sizes[size_index])+c*gauss(axis, pos3, sizes[size_index])
    elif dim==2:
        source = a*gauss2d((axis, axis), par1)+b*gauss2d((axis, axis), par2)+c*gauss2d((axis, axis), par3)

    source = normalize(source)

    if ReturnMaxSize:
        counter_max = np.size(distances)*np.size(sizes)
        return source, counter_max
    else:
        return source

def Source1D(Counter, libsize = 'medium'):
    #original_counter = cp(Counter)
    if libsize=='small':
        nRandom = 20
    if libsize=='medium':
        nRandom = 80
    if libsize=='small':
        nRandom = 200

    source, Nrepetitions = SingleGauss(0,libsize=libsize, ReturnMaxSize=True)
    if Counter< Nrepetitions:
        source, Nrepetitions = SingleGauss(Counter,libsize=libsize, ReturnMaxSize=True)

        return source
    Counter-=Nrepetitions

    #two sources:
    source, Nrepetitions = dipol(0,libsize=libsize, ReturnMaxSize=True)
    if Counter< Nrepetitions:
        source= dipol(Counter,libsize=libsize)
        return source
    Counter-=Nrepetitions

    source, Nrepetitions = dipol(0,libsize=libsize, ReturnMaxSize=True)
    if Counter< Nrepetitions:
        source= dipol(Counter,libsize=libsize, same_sign=False)
        return source
    Counter-=Nrepetitions

    source, Nrepetitions = dipol(0,libsize=libsize, ReturnMaxSize=True)
    if Counter< Nrepetitions:
        source= tripole(Counter,libsize=libsize, sign_way=1)
        return source
    Counter-=Nrepetitions

    source, Nrepetitions = dipol(0,libsize=libsize, ReturnMaxSize=True)
    if Counter< Nrepetitions:
        source= tripole(Counter,libsize=libsize, sign_way=2)
        return source
    Counter-=Nrepetitions


    if Counter<nRandom:
        source = RandomCSD1d(Counter, 100)
        return source
    return False


def RandomCSD1d(seed, resolution=100):
    '''
    resolution for space from 0 to 1, so 2*resolution from -0.5 to 1.5
    seed/the counter if for random sources
    '''
    xaxis = np.linspace(-0.5,1.5, resolution*2)

    np.random.seed(seed)
    z = np.random.random(size=3*4+1)
    if z[0]<0.25:
        n_gauss = 1
    elif z[0]<0.5:
        n_gauss = 2
    elif z[0]<0.75:
        n_gauss = 3
    else:
        n_gauss = 4

    csd = np.zeros(xaxis.size)
    counter = 1
    for i in range(n_gauss):
        pos = z[counter]
        counter+=1
        std = z[counter]
        counter+=1
        amp = z[counter]-0.5
        counter+=1
        csd += gauss(xaxis, pos, weight_width(std) )*amp

    csd = normalize(csd)
    #csd *= ss.tukey(xaxis.size)
    #csd *= ss.gaussian(xaxis.size, xaxis.size/2.)
    #csd *= ss.blackman(xaxis.size)

    return csd

def RandomCSD2d(seed, resolution=100):
    '''
    xaxis = np.linspace(-1,2, 3*N)

    '''
    #window = Window2d(2*N, 2*N)
    #start_x, end_x, res_x = -0.5,1.5, 2*resolution
    #start_y, end_y, res_y = -0.5,1.5, 2*resolution

    #xaxis = np.linspace(start_x,end_x, res_x)
    xaxis = np.linspace(-0.5, 1.5, resolution*2)
    #yaxis = cp(xaxis)

    #csd_x, csd_y = np.mgrid[start_x:end_x:np.complex(0,res_x),
    #                        start_y:end_y:np.complex(0,res_y)]
    np.random.seed(seed)
    z = np.random.random(size=6*4+1)
    if z[0]<0.25:
        n_gauss = 1
    elif z[0]<0.5:
        n_gauss = 2
    elif z[0]<0.75:
        n_gauss = 3
    else:
        n_gauss = 4
    csd = np.zeros((xaxis.size, xaxis.size))
    counter = 1
    for i in range(n_gauss):
        XCEN = z[counter]
        counter+=1
        YCEN = z[counter]
        counter+=1
        SIGMAX = weight_width(z[counter])
        counter+=1
        SIGMAY = weight_width(z[counter])
        counter+=1
        ANGLE = z[counter]
        counter+=1
        AMP = z[counter]-0.5
        counter+=1
        #csd += gauss2d(csd_x, csd_y, XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE  )
        p = [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE ]
        csd += gauss2d((xaxis, xaxis), p)

    csd = normalize(csd)

    #csd *= ss.tukey(xaxis.size)
    #csd *= ss.gaussian(xaxis.size, xaxis.size/2.)
    #csd *= window

    return csd
#
#def gauss2d(x,y,XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE):
#    """
#     p:     list of parameters of the Gauss-function
#            [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE]
#            SIGMA = FWHM / (2*sqrt(2*log(2)))
#            ANGLE = rotation of the X,Y direction of the Gaussian in radians
#    Returns
#    -------
#    the value of the Gaussian described by the parameters p
#    at position (x,y)
#    """
#
#    rcen_x = XCEN * np.cos(ANGLE) - YCEN * np.sin(ANGLE)
#    rcen_y = XCEN * np.sin(ANGLE) + YCEN * np.cos(ANGLE)
#    xp = x * np.cos(ANGLE) - y * np.sin(ANGLE)
#    yp = x * np.sin(ANGLE) + y * np.cos(ANGLE)
#    g = AMP*np.exp(-(((rcen_x-xp)/SIGMAX)**2+
#                      ((rcen_y-yp)/SIGMAY)**2)/2.)
#    return g

def Window2d(NX, NY):

    window1 = np.ones((NX, NY))*ss.blackman(NX).reshape((NX,1))
    window2 = np.ones((NX, NY))*ss.blackman(NY).reshape((1,NY))
    return np.minimum(window1, window2)

#Edited and made similar:

def calculate_potential_1D(csd, measure_locations, csd_space_x, h, sigma=1.):
    pots = np.zeros(len(measure_locations))
    for ii in range(len(measure_locations)):
        pots[ii] = integrate_1D(measure_locations[ii], csd_space_x, csd, h)
    pots *= 1/(2.*sigma) #eq.: 26 from Potworowski et al
    return pots

def calculate_potential_2D(csd, ele_xx, ele_yy, csd_x, csd_y, h, sigma=1.):
    """
    copied from test_CSD
    CHANGE - add the h to the variables called
    """
    #xlin = csd_x[:,0]
    #ylin = csd_y[0,:]

#    xlims = [xlin[0], xlin[-1]]
#    ylims = [ylin[0], ylin[-1]]

    gridx, gridy = np.meshgrid(csd_x, csd_y)
    xlims = [csd_x[0], csd_x[-1]]
    ylims = [csd_y[0], csd_y[-1]]

    pots = np.zeros(len(ele_xx))
    for ii in range(len(ele_xx)):
        pots[ii] = integrate_2D(ele_xx[ii], ele_yy[ii],
                                xlims, ylims, csd, h,
                                csd_x, csd_y, gridx, gridy)
    pots /= 2*np.pi*sigma
    return pots

def calculate_potential_3D(true_csd, ele_xx, ele_yy, ele_zz,
                           csd_x, csd_y, csd_z, sigma = 1.0):
    """
    For Mihav's implementation to compute the LFP generated
    not corrected
    """
    xlin = csd_x[:,0,0]
    ylin = csd_y[0,:,0]
    zlin = csd_z[0,0,:]
    xlims = [xlin[0], xlin[-1]]
    ylims = [ylin[0], ylin[-1]]
    zlims = [zlin[0], zlin[-1]]
    pots = np.zeros(len(ele_xx))
    #tic = time.time()
    for ii in range(len(ele_xx)):
        pots[ii] = integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                xlims, ylims, zlims, true_csd,
                                xlin, ylin, zlin,
                                csd_x, csd_y, csd_z)
    #    print 'Electrode:', ii
    pots /= 4*np.pi*sigma
    #toc = time.time() - tic
    #print toc, 'Total time taken - series, sims'
    return pots



if __name__ == "__main__":
    save_dir = '/home/mczerwinski/results/csdPaper/'
    #os.chdir(save_dir)

    if True:
        os.chdir(save_dir)
        size = 'small'
        N = 100
        xaxis = np.linspace(-0.5,1.5, 2*N)
        counter = 0
        csd=True
        plot_counter = 1
        for i in range(20):
            subplot_counter = 1
            if csd is False:
                break
            plt.figure('picture 1d '+ str(plot_counter), figsize=(12,12))
            for j in range(6):
                for k in range(6):
                    csd = Source1D(counter, libsize=size)
                    if csd is False:
                        continue
                    plt.subplot(6,6,subplot_counter)
                    plt.axvline(1, lw=0.2, color='k')
                    plt.axvline(0, lw=0.2, color='k')
                    subplot_counter+=1

                    plt.plot(xaxis, csd)
                    plt.title(counter)

                    counter+=1
                if csd is False:
                        continue
            plot_counter+=1
            plt.savefig('test '+size+' 1d '+str(plot_counter-1)+'.png')
            plt.close()

    # Plot all 1D:
    if False:
        N = 100
        xaxis = np.linspace(-0.5,1.5, 2*N)
        counter = 0
        plot_counter = 1
        for i in range(20):
            subplot_counter = 1
            plt.figure('picture 1d '+ str(plot_counter), figsize=(12,12))
            for j in range(6):
                for k in range(6):
                    plt.subplot(6,6,subplot_counter)
                    plt.axvline(1, lw=0.2, color='k')
                    plt.axvline(0, lw=0.2, color='k')
                    subplot_counter+=1
                    plt.plot(xaxis, RandomCSD1d(counter, 100))
                    plt.title(counter)

                    counter+=1
            plot_counter+=1
            plt.savefig('1d '+str(plot_counter-1)+'.png')
            plt.close()

    # Plot all 2D
    if False:
        N = 50
        xaxis = np.linspace(-0.5,1.5, 2*N)
        counter = 0
        plot_counter = 1
        for i in range(20):
            subplot_counter = 1
            plt.figure('picture 2d '+ str(plot_counter), figsize=(12,12))
            for j in range(6):
                for k in range(6):
                    plt.subplot(6,6,subplot_counter)
                    plt.axvline(1, lw=0.2, color='k')
                    plt.axvline(0, lw=0.2, color='k')
                    plt.axhline(1, lw=0.2, color='k')
                    plt.axhline(0, lw=0.2, color='k')

                    subplot_counter+=1
                    csd = TrueCSD2d(counter, N)
                    plt.pcolor(xaxis, xaxis, csd, cmap='bwr', vmin=-1, vmax=1)
                    plt.title(counter)

                    counter+=1
            plot_counter+=1
            plt.savefig('2d '+str(plot_counter-1)+'.png')
            plt.close()
    os.chdir(loc)