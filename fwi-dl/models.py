from __future__ import division

import math
import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter


def basic_model(config, vBase=1.0, pos=[0.5], amp=[-100.0], var=0.5e-4):
    """Returns 1D derivative of Gaussian reflector model.

    Given a configuration dictionary that specifies:

    1) 'x_limits': A tuple specifying the left and right bound of the domain
    2) 'nx': The number of nodes or degrees of freedom, including the end
             points
    3) 'dx': The spatial step

    returns a 2-tuple containing the true velocity model C and 
    the background velocity C0.

    Keyword parameters:
    vBase  background constant velocity
    pos    reflector position(s)
    amp    reflector amplitude(s)
    var    variance of Gaussian function (=sigma**2)

    """

    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    xs = x_limits[0] + np.arange(nx,dtype=np.float)*dx
    sigma = math.sqrt( var )
    coef = 1. 	# 1./(sigma*var*math.sqrt(2.*math.pi))
    
    C0 = np.ones(nx)*vBase
    C = C0.copy()
    for i in range(len(pos)):   # All reflector positions
        xi = xs - pos[i]
        dC = xi * coef * np.exp( -(xi**2)/(2.*var) )
        dC[np.where(abs(dC) < 1e-7)] = 0
        C += dC * amp[i]

    return C, C0


def gauss_model(config, positions=[(0.4,0.55)], amp=[(0.2,0.3)]):

    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    xs = x_limits[0] + np.arange(nx)*dx
    C0 = np.ones(nx)
    C = C0
    for i in range(len(positions)):
        dC = amp[i][0]*np.exp(-((xs-positions[i][0])**2)/(1e-4)) -  \
             amp[i][1]*np.exp(-((xs-positions[i][1])**2)/(1e-3))
        dC[np.where(abs(dC) < 1e-7)] = 0

        C = C + dC

    return C, C0


def basic_seismic_model(config, positions=[0.5], amp=[0.3]):

    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']
    x = x_limits[1]-x_limits[0]

    xs = x_limits[0] + np.arange(nx)*dx
    C0 = np.ones(nx)
    C = C0
    for i in range(len(positions)):
        dC = np.zeros(nx)
        dC[int((positions[i]/x)*nx):] = amp[i]

        C = C + dC

    p = [0,1]
    for i in range(int(len(positions)/2)):
        dC = np.zeros(nx)
        p[0] = random.randint(10, nx-10)
        p[1] = random.randint(10, nx-10)
        p.sort()
        val = random.uniform(0.1,0.3)
        dC[p[0]:p[1]] = -val
        C = C + dC

    C = gaussian_filter(C, sigma=3)
    C0 = gaussian_filter(C, sigma=21)
    return C, C0

def random_model(config):
    if (random.random() < 1.0):
        num_layer = random.randint(1, 5) 
        positions=[]
        amp = []
        for i in range(num_layer):
            amp.append(random.uniform(0.1, 0.5))
            positions.append(random.uniform(0.1,0.9))
        positions.sort()
        print("A random basic seismic model: ", positions, amp)
        return basic_seismic_model(config, positions, amp)
    elif (random.random() < 0.75):
        num_layer = random.randint(1, 5) 
        positions=[]
        amp = []
        for i in range(num_layer):
            amp.append(random.uniform(20.0, 100.0))
            positions.append(random.uniform(0.1,0.9))
        amp.sort()
        positions.sort()
        print("A random basic model: ", positions, amp)
        return basic_model(config, positions, amp)
    else:
        num_layer = random.randint(1, 3) 
        positions=[]
        amp = []
        for i in range(num_layer*2):
            amp.append(random.uniform(0.1,0.5))
            positions.append(random.uniform(0.1,0.9))
        positions.sort()
        amp.sort()
        pos_pair=[]
        amp_pair=[]
        for i in range(num_layer):
            pos_pair.append((positions[i*2], positions[i*2+1]))
            amp_pair.append((amp[i*2], amp[i*2+1]))
        print("A random gaussian model: ", pos_pair, amp_pair)
        return gauss_model(config, pos_pair, amp_pair)

