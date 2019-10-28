#  PySIT Tutorial exercises

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from models import basic_model

config = dict()


##############################################################################
# Problem 1.1

def ricker(t, config):

    nu0 = config['nu0']
    sigmaInv = math.pi * nu0 * np.sqrt(2)
    cut = 1.e-6
    t0 = 6. / sigmaInv
    tdel = t - t0
    expt = (math.pi * nu0 * tdel) ** 2
    w = np.zeros([t.size])
    w[:] = (1. - 2. * expt) * np.exp( -expt )
    w[np.where(abs(w) < 1e-7)] = 0

    return w

# Configure source wavelet
config['nu0'] = 10  # Hz

# Evaluate wavelet and plot it
'''
ts = np.linspace(0, 0.5, 1000)
ws = ricker(ts, config)

plt.figure()
plt.plot(ts, ws,
         color='green',
         label=r'$\nu_0 =\,{0}$Hz'.format(config['nu0']),
         linewidth=2)

plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$w(t)$', fontsize=18)
plt.title('Ricker Wavelet', fontsize=22)

plt.legend()
'''

##############################################################################
# Problem 1.2

def point_source(value, position, config):

    nx = config['nx']
    dx = config['dx']
    xbgn, xend = config['x_limits']
    f = np.zeros([nx])
    # Find the first spatial sample point beyond the source location
    xpos = position - xbgn
    ixs = int(max(1, np.ceil( xpos/dx )))
    # Distribute the unit amplitude proportionally 
    # between the two neighboring sample positions
    frac = ixs*dx - xpos
    f[ixs] = (1. - frac) * value
    f[ixs-1] = frac * value

    return f

# Domain parameters
config['x_limits'] = [0.0, 1.0]
config['nx'] = 201
config['dx'] = (config['x_limits'][1] - config['x_limits'][0]) / (config['nx']-1)

# Source parameter
config['x_s'] = 0.1


##############################################################################
# Problem 1.3

def wave_matrices(C, config):

    # Operator  ( M Dtt + A Dt + K ) u(x,t) = f(x,t)

    dt = config['dt']
    dx = config['dx']
    nx = config['nx']

    # Stiffness (Left,Middle,Right)
    left = 0; mid = 1; right = 2  # K-indices - spatial neighbors
    lbdry = 0; rbdry = -1  # boundary indices

    #  boundaries:
    Kx = np.zeros([3,nx])
    Kx[mid,  lbdry] = -C[lbdry] / (dx * dt)
    Kx[right,lbdry] = +C[lbdry] / (dx * dt)
    Kx[left, rbdry] = +C[rbdry] / (dx * dt)
    Kx[mid,  rbdry] = -C[rbdry] / (dx * dt)

    #  interior:
    Cwgt = (C[1:-1] / dx)**2
    Kxx = np.zeros([3,nx])
    Kxx[left, 1:-1] = +1. * Cwgt
    Kxx[mid,  1:-1] = -2. * Cwgt
    Kxx[right,1:-1] = +1. * Cwgt

    K = Kxx + Kx

    # Attenuation (Current time step)
    A = np.zeros([nx])
    A[lbdry] = 1./dt 
    A[rbdry] = 1./dt 
 
    '''
    # Mass (0 Previous, 1 Current)
    M = np.zeros([2,nx])
    M[0,1:-1] = -1.
    M[1,1:-1] = +2.
    '''
    M = np.zeros([nx])
    M[1:-1] = 1.
    
    return M, A, K

# Load the model
C, C0 = basic_model(config)

# Build an example set of matrices
# M, A, K = wave_matrices(C, config)


##############################################################################
# Problem 1.4 -- modified as first-derivative ODE system

def advance(C, sources, config):

    # 1D wavefield propagation [Tutorial 'leap_frog' function]
    dt = config['dt']
    nt = config['nt']
    dx = config['dx']
    nx = config['nx']
    Csq = C**2
    M, A, K = wave_matrices(C, config)
    left = 0; mid = 1; right = 2  # K-indices
    lbdry = 0; rbdry = -1  # boundary indices

    # Set up first-derivative system, 
    # time positions Current (C) and Next (N)
    vC = np.zeros([nx])
    uC = np.zeros([nx])
    us = list() 	# list of output wavefields
    
    # loop over time-steps
    for it in range(nt):
        us.append( uC )

        f = sources[it] * Csq  # C2dt2
        # m = np.sum( np.stack([uP,uC]) * M, axis=0 )
        # m = M  Mass matrix 1/Csq is not used in this version

        # Stiffness -- interior points 
        # (computing all, but endpoints will be replaced)
        k = np.sum( np.stack([np.roll(uC,1), uC, np.roll(uC,-1)] ) * K, axis=0)
        
        # boundary points (0 lbdry, -1 rbdry)
        k[lbdry] = uC[lbdry+1]*K[right,lbdry] \
                 + uC[lbdry  ]*K[mid,  lbdry]
        k[rbdry] = uC[rbdry-1]*K[left, rbdry] \
                 + uC[rbdry  ]*K[mid,  rbdry]
        
        # Attenuation
        a = Csq * A * vC

        vN = vC + dt*(f + k - a)
        uN = uC + dt*vN
        
        vC = vN
        uC = uN
            
    return us

# Define time step parameters
config['alpha'] = 1./6.    # CFL stability coefficient
config['T'] = 3    # seconds
config['dt'] = config['alpha'] * config['dx'] / C.max()
config['nt'] = int(config['T']/config['dt'])

# Generate the sources
sources = list()
for it in range(config['nt']):
    t = it*config['dt']
    f = point_source(ricker(t, config), config['x_s'], config)
    sources.append(f)
#print( 'Built %d sources of dimension %d'%(len(sources), len(sources[0])) )

# Generate wavefields
us = advance(C, sources, config)
print( 'Generated %d wavefields of shape'%len(us), us[0].shape)

##############################################################################
# Problem 1.5

def plot_space_time(u, config, overlay=None, title=None):

    # Plots a wavefield u
    xlim = config['x_limits']
    xlen = xlim[1] - xlim[0]
    T = config['T']
    dt = config['dt']
    nt = len(us)
    nx = len(us[0])
    img = np.zeros([nt,nx])
    for it in range(nt):
        img[it,:] = u[it]
    if overlay:
        ovr = np.zeros([nt,nx])
        for it in range(nt):
            ovr[it,:] = overlay[it]

    f = plt.figure(tight_layout=True)
    ax = plt.axes()
    ax.set_title(title,fontsize=14)
    plt.imshow(img, cmap='gray', aspect='auto')
    if overlay:
        plt.imshow(ovr, cmap='seismic', alpha=0.6, aspect='auto')

    plt.xlabel(r'space($x$) km', fontsize=14)
    xtlabel = np.linspace(xlim[0],xlim[1],11,endpoint=True)
    xtpos = np.linspace(0,nx,11,endpoint=True) 
    plt.xticks( ticks=xtpos, labels=np.around(xtlabel,decimals=1) )

    plt.ylabel(r'time($t$) sec', fontsize=14)
    ytlabel = np.arange(0., T+0.5, 0.5)
    delt = 0.5/dt
    ytpos = np.arange(0, nt+delt, delt) 
    plt.yticks( ticks=ytpos, labels=ytlabel )
    
# Plot wavefield
plot_space_time(us, config, title=r'Wavefield u($x,t$)')

##############################################################################
# Problem 1.6

def record_data(u, config):

    nx = config['nx']
    dx = config['dx']
    nt = config['nt']
    x_r = config['x_r']
    xbgn, xend = config['x_limits']

    xpos = x_r - xbgn
    ixr = int(max(1, np.ceil( xpos/dx )))
    frac = ixr*dx - xpos
    trc = np.zeros([nt])
    for it in range(nt):
        trc[it] = frac*u[it][ixr-1] + (1. - frac)*u[it][ixr]

    return trc

# Receiver position
config['x_r'] = 0.15

##############################################################################
# Problem 1.7

def forward_operator(C, config):

    # Set up the propagation matrices
    M, A, K = wave_matrices(C, config)

    # Define the sources
    sources = list()
    for it in range(config['nt']):
        t = it*config['dt']
        f = point_source(ricker(t, config), config['x_s'], config)
        sources.append(f)

    # Generate wavefields
    us = advance(C, sources, config)

    # Extract wave data at receiver
    trace = record_data(us, config)

    return us, trace


us, d = forward_operator(C, config)

# The last argument False excludes the end point from the list
ts = np.linspace(0, config['T'], config['nt'], False)
'''
plt.figure()
plt.plot(ts, d, label=r'$x_r =\,{0}$'.format(config['x_r']), linewidth=2)
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$d(t)$', fontsize=16)
plt.title('Trace at $x_r={0}$'.format(config['x_r']), fontsize=18)
plt.legend()
'''
##############################################################################
# Problem 2.1
# Residual trace from proposal model C0

u0s, d0 = forward_operator(C0, config)
r = d - d0
'''
plt.figure()
plt.plot(ts, r, label=r'$x_r =\,{0}$'.format(config['x_r']), linewidth=2)
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$d(t)$', fontsize=16)
plt.title('Residual trace at $x_r={0}$'.format(config['x_r']), fontsize=18)
plt.legend()
'''

##############################################################################
# Problem 2.2
# Adjoint wavefield from 'sources' at receiver positions

# Generate adjoint sources
nt = config['nt']
adjsrc = list()
for it in range(nt):
    t = it*config['dt']
    # Source in reverse-time order
    f = point_source(r[nt-it-1], config['x_r'], config)
    adjsrc.append(f)
#print( 'Built %d adjoint sources of dimension %d'%(len(adjsrc), len(adjsrc[0])) )

# Generate adjoint wavefields
qs = advance(C0, adjsrc, config)
print( 'Generated %d adjoint wavefields of shape'%len(qs), qs[0].shape)


##############################################################################
# Problem 2.3

def imaging_condition(qs, u0s, config):

    # Imaging condition of trial wavefield u0s with adjoint field qs
    nx = config['nx']
    nt = config['nt']
    dt = config['dt']
    dtscl = 1./dt**3
    upad = np.zeros( nx )
    uprev = upad.copy()
    ucurr = upad.copy()
    unext = u0s[0]
    image = np.zeros( nx )

    # Integration loop, reversed through adjoint wavefield
    for it in range(nt):
        uprev = ucurr
        ucurr = unext
        if it==nt-1:
            unext = upad.copy()
        else:
            unext = u0s[it+1]
        Irtm = qs[nt-it-1]*Dtt( uprev, ucurr, unext ) 
        image -= Irtm*dtscl 

    return image

def Dtt( uprev, ucurr, unext ):
    # Second time derivative
    return uprev - 2.*ucurr + unext

# Compute the image
I_rtm = imaging_condition(qs, u0s, config)

'''
# Plot the comparison
xs = np.arange(config['nx'])*config['dx']
dC = C-C0

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xs,  C, label=r'$C$')
plt.plot(xs, C0, label=r'$C0$')
plt.plot(xs, dC, label=r'$\delta C$')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(xs, I_rtm, label=r'$I_{RTM}$')
plt.legend()
'''

##############################################################################
# Problem 2.4
# Wavefield comparisons for imaging condition

# Trial wavefield second derivative
dtscl = 1./config['dt']**2
upad = np.zeros( config['nx'] )

uprev = upad.copy()
ucurr = upad.copy()
unext = u0s[0]
DttU0 = list()
qsrev = list()

for it in range(config['nt']):
    uprev = ucurr
    ucurr = unext
    if it==nt-1:
        unext = upad.copy()
    else:
        unext = u0s[it+1]
    DttU0.append( Dtt( uprev, ucurr, unext ) * dtscl )
    qsrev.append( qs[nt-it-1] )

plot_space_time(qsrev, config, overlay=DttU0, \
    title=r'Imaging wavefields, DttU0 (color) and adjoint (gray)')


##############################################################################
# Problem 2.5

def adjoint_operator(C0, d, config):

    # Generate adjoint sources
    nt = config['nt']
    adjsrc = list()
    for it in range(nt):
        t = it*config['dt']
        # Source in reverse-time order
        f = point_source(d[nt-it-1], config['x_r'], config)
        adjsrc.append(f)

    # Generate adjoint wavefields
    qs = advance(C0, adjsrc, config)

    # Compute the image
    image = imaging_condition(qs, u0s, config)

    return image


##############################################################################
# Problem 3.1

def linear_sources(dm, u0s, config):

    # Source wavefields for linear modeling equations
    nx = config['nx']
    nt = config['nt']
    dt = config['dt']
    upad = np.zeros( nx )

    uprev = upad.copy()
    ucurr = upad.copy()
    unext = u0s[0]
    sources = list()

    for it in range(nt):
        uprev = ucurr
        ucurr = unext
        if it==nt-1:
            unext = upad.copy()
        else:
            unext = u0s[it+1]
        sources.append( -dm*Dtt( uprev, ucurr, unext ) )

    return sources

'''
##############################################################################
# Problem 3.2

def linear_forward_operator(C0, dm, config):

    # Propagator with dm as linearized source

    # Set up the propagation matrices
    M, A, K = wave_matrices(C0, config)

    # Generate wavefields
    u1s = advance(C0, dm, config)

    # Extract wave data at receiver
    trace = record_data(us, config)

    return u1s, trace

linsrc = linear_sources(dm, u0s, config)
u1s, d = linear_forward_operator(C0, linsrc, config)


##############################################################################
# Problem 3.3

def adjoint_condition(C0, config):
    
    # Test for equality of:
    #    dot( Sampled(Forward(dm)), data )  [in data domain] and
    #    dot( dm, Adjoint(Sampled(data)) )  [in model domain]

    # LHS (data domain)
    u0s, data = forward_operator(C0, config)
    linsrc = linear_sources(dm, u0s, config)
    u1s, d = linear_forward_operator(C0, linsrc, config)


##############################################################################
# Problem 4.1

def gradient_descent(C0, d, k, config):

    # implementation goes here
    pass


##############################################################################
# Problem 4.2

'''
plt.show()

