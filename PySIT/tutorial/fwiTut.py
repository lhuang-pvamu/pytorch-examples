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
'''
# Evaluate wavelet and plot it
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
    bgn, end = config['x_limits']
    xv = np.linspace( bgn, end, nx ) - position
    f = np.zeros([nx])
    for ix in range(len(xv)):
        if xv[ix] >= 0. :
            frac = xv[ix] / dx
            f[ix] = (1. - frac) * value
            f[ix-1] = frac * value
            break

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
    #  boundaries:
    Kx = np.zeros([3,nx])
    Kx[1,0]  = +C[0]*dt / dx
    Kx[2,0]  = -C[0]*dt / dx
    Kx[0,-1] = -C[-1]*dt / dx
    Kx[1,-1] = +C[-1]*dt / dx

    #  interior:
    Cwgt = (C * dt / dx)**2
    Kxx = np.zeros([3,nx])
    Kxx[0][1:-1] = +1. * Cwgt[0:-2]
    Kxx[1][1:-1] = -2. * Cwgt[1:-1]
    Kxx[2][1:-1] = +1. * Cwgt[2:]

    K = Kx + Kxx

    # Attenuation (Current)
    A = np.zeros([nx])
    A[0]  = C[0]*dt/dx
    A[-1] = -C[-1]*dt/dx

    # Mass (Previous,Current)
    M = np.zeros([2,nx])
    M[0,:] = -1.
    M[1,:] = +2.
    
    return M, A, K

# Load the model
C, C0 = basic_model(config)

# Build an example set of matrices
# M, A, K = wave_matrices(C, config)


##############################################################################
# Problem 1.4

def advance(C, sources, config):

    # 1D wavefield propagation
    dt = config['dt']
    nt = config['nt']
    dx = config['dx']
    nx = config['nx']
    C2dt2 = (C * dt)**2
    M, A, K = wave_matrices(C, config)

    # Initial conditions (time-steps are Previous, Current, Next)
    prev = 0; curr = 1; next = 2
    u = np.zeros([3,nx])
    us = list()
    
    # loop over time-steps
    for it in range(nt):
        temp = prev; prev = curr; curr = next; next = temp
        
        # interior points
        m = u[curr, 1:-1]*M[1, 1:-1] + u[prev, 1:-1]*M[0, 1:-1]
        k = u[curr, 0:-2]*K[0, 1:-1] + u[curr, 1:-1]*K[1, 1:-1] \
          + u[curr, 2:  ]*K[2, 1:-1]
        f = sources[it][1:-1] * C2dt2[1:-1]

        u[next, 1:-1] = m + k + f

        # boundary points
        u[next, 0] = u[curr, 0] + A[ 0]*(u[curr, 1] - u[curr, 0])
        u[next,-1] = u[curr,-1] + A[-1]*(u[curr,-1] - u[curr,-2])

        us.append( u[next,:].copy() )
            
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
print( 'Built %d sources of dimension %d'%(len(sources), len(sources[0])) )

# Generate wavefields
us = advance(C, sources, config)
print( 'Generated %d wavefields of shape'%len(us), us[0].shape)

##############################################################################
# Problem 1.5

def plot_space_time(u, config, title=None):

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

    f = plt.figure(tight_layout=True)
    ax = plt.axes()
    #ax.set_xlim(xlim)
    #ax.set_ylim((0,T))
    ax.set_title(title,fontsize=16)
    plt.imshow(img, cmap='gray', aspect='auto')
    plt.colorbar()

    plt.xlabel(r'space($x$) km', fontsize=14)
    xtlabel = np.linspace(xlim[0],xlim[1],5,endpoint=True)
    xtpos = np.linspace(0,nx,5,endpoint=True) 
    plt.xticks( ticks=xtpos, labels=xtlabel )

    plt.ylabel(r'time($t$) sec', fontsize=14)
    ytlabel = np.arange(0., T+0.5, 0.5)
    delt = 0.5/dt
    ytpos = np.arange(0, nt+delt, delt) 
    plt.yticks( ticks=ytpos, labels=ytlabel )
    

# Plot wavefields
#plot_space_time(sources, config, title=r'f(x,t)')
plot_space_time(us, config, title=r'wavefield u($x,t$)')

'''
##############################################################################
# Problem 1.6

def record_data(u, config):

    # implementation goes here
def point_source(value, position, config):

    nx = config['nx']
    dx = config['dx']
    x_r = config['x_r']
    bgn, end = config['x_limits']
    xv = np.linspace( bgn, end, nx ) - x_r
    f = np.zeros([nx])
    for ix in range(len(xv)):
        if xv[ix] >= 0. :
            frac = xv[ix] / dx
            f[ix] = (1. - frac) * value
            f[ix-1] = frac * value
            break

    return f

    return d

# Receiver position
config['x_r'] = 0.15


##############################################################################
# Problem 1.7

def forward_operator(C, config):

    # implementation goes here

    return us, trace


us, d = forward_operator(C, config)

# The last argument False excludes the end point from the list
ts = np.linspace(0, config['T'], config['nt'], False)

plt.figure()
plt.plot(ts, d, label=r'$x_r =\,{0}$'.format(config['x_r']), linewidth=2)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$d(t)$', fontsize=18)
plt.title('Trace at $x_r={0}$'.format(config['x_r']), fontsize=22)
plt.legend()


##############################################################################
# Problem 2.1



##############################################################################
# Problem 2.2



##############################################################################
# Problem 2.3

def imaging_condition(qs, u0s, config):

    # implementation goes here

    return image

# Compute the image
I_rtm = imaging_condition(qs, u0s, config)

# Plot the comparison
xs = np.arange(config['nx'])*config['dx']
dC = C-C0

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xs, dC, label=r'$\delta C$')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(xs, I_rtm, label=r'$I_\text{RTM}$')
plt.legend()


##############################################################################
# Problem 2.4



##############################################################################
# Problem 2.5

def adjoint_operator(C0, d, config):

    # implementation goes here

    return image



##############################################################################
# Problem 3.1

def linear_sources(dm, u0s, config):

    # implementation goes here

    return sources


##############################################################################
# Problem 3.2

def linear_forward_operator(C0, dm, config):

    # implementation goes here

    return u1s


##############################################################################
# Problem 3.3

def adjoint_condition(C0, config):

    # implementation goes here
    pass


##############################################################################
# Problem 4.1

def gradient_descent(C0, d, k, config):

    # implementation goes here

    return sources


##############################################################################
# Problem 3.2

def linear_forward_operator(C0, dm, config):

    # implementation goes here

    return u1s


##############################################################################
# Problem 3.3

def adjoint_condition(C0, config):

    # implementation goes here
    pass


##############################################################################
# Problem 4.1

def gradient_descent(C0, d, k, config):

    # implementation goes here
    pass


##############################################################################
# Problem 4.2

'''
plt.show()

