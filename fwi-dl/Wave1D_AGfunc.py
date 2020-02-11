#  One-D wave propagation using PyTorch 

import sys
import math
import argparse
import time
import numpy as np
#import numpy.random
import torch
#import torch.nn.functional as F
#from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from models import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pgmDesc = 'Physics-Guided Neural Network (PGNN) of 1D Wave Propagation from PySIT Exercises'

def parseArgs():
    ''' Parameter setup for PGNN 1D Wave Propagation '''
    note = '''Note: TODO: Make parameters instead of hardwired values.
    '''
    ap = argparse.ArgumentParser( description='%(prog)s -- '+pgmDesc, epilog=note )

    # TODO: Add arguments logic here

    return ap.parse_args()

##############################################################################

def ricker(t, config):

    nu0 = config['nu0']
    sigmaInv = math.pi * nu0 * math.sqrt(2)
    cut = 1.e-6
    t0 = 6. / sigmaInv
    tdel = t - t0
    expt = (math.pi * nu0 * tdel) ** 2
    w = torch.zeros(t.size()).to(device)
    w[:] = (1. - 2. * expt) * torch.exp( -expt )
    w[torch.where(abs(w) < 1e-7)] = 0

    return w


##############################################################################

class Point_Source(object):
  def __init__(self, position, config):
    ''' Configures interpolation of source in sampling grid
    '''
    self.nx = config['nx']
    dx = config['dx']
    xbgn, xend = config['x_limits']
    # Find the first spatial sample point beyond the source location
    xpos = position - xbgn
    ixs = max(1, int(np.ceil( xpos/dx )) )
    # Distribute the unit amplitude proportionally 
    # between the two neighboring sample positions
    frac = ixs - xpos/dx
    # Force zero on  boundary
    frac=0. if ixs==1    else frac
    frac=1. if ixs==self.nx-1 else frac
    self.ixs = ixs
    self.frac = frac

  def set(self, value ):
    ''' Sets amplitude at source point for current time 
    '''
    f = torch.zeros([self.nx]).to(device)
    f[self.ixs] = (1. - self.frac) * value
    f[self.ixs-1] = self.frac * value

    return f

##############################################################################

class Receivers(object):
  def __init__(self, config):
    ''' Configures interpolation of receivers in sampling grid
    '''
    nx = config['nx']
    dx = config['dx']
    xbgn, _ = config['x_limits']
    x_r = np.asarray( config['x_r'] )
    self.nrec = len(x_r)

    # Find the first spatial sample point beyond each receiver location
    xpos = (x_r - xbgn)
    ixr = ( np.clip( np.ceil(xpos/dx), 1, nx-1) ).astype(np.int64)
    frac = (ixr - xpos/dx)
    self.ixr = torch.from_numpy(ixr).long().to(device)
    self.frac = torch.from_numpy(frac).float().to(device)

  def sample(self, u ):
    ''' Interpolates amplitude at each receiver point for current time 
    '''
    unext = torch.gather(u, 0, self.ixr  )
    uprev = torch.gather(u, 0, self.ixr-1)
    samples = self.frac * uprev + (1.-self.frac) * unext
    return samples

##############################################################################
# Problem 1.3

def wave_matrices(C, config):

    # Operator  ( M Dtt + A Dt + K ) u(x,t) = f(x,t)

    dt = config['dt']
    dx = config['dx']
    nx = config['nx']
    bc = config['bc']
    fixedBC = bc=='fixed'

    # Stiffness 
    # spatial second-order accurate centered and forward differences
    spcOrder = 2
    spcCtr = torch.FloatTensor([   1., -2.,    1. ]).to(device)
    spcFwd = torch.FloatTensor([-3./2., 2., -1./2.]).to(device)
    lbdry = 0; rbdry = -1  # boundary indices

    #  interior points:
    Cwgt = ((C/dx)**2).to(device)
    Kxx = torch.repeat_interleave(spcCtr,nx).reshape(3,nx).to(device) * Cwgt
    Kxx[:,lbdry] = 0.
    Kxx[:,rbdry] = 0. 

    #  boundaries:
    Kx = torch.zeros([3,nx]).to(device)
    if not fixedBC:
        Kx[:,lbdry] =            spcFwd  * C[lbdry] / dx
        Kx[:,rbdry] = torch.flip(spcFwd,[0]) * C[rbdry] / dx

    # K = Kxx + Kx

    # Attenuation 
    timOrder = 2  # second-order backward differencees
    timBak = torch.FloatTensor([3./2., -2., 1./2.]).to(device)

    A = torch.zeros([3]).to(device)
    A[0] = 1./timBak[0]
    A[1] = -timBak[1]
    A[2] = -timBak[2]
 
    '''
    # Mass (0 Previous, 1 Current)
    M = np.zeros([2,nx])
    M[0,1:-1] = -1.
    M[1,1:-1] = +2.
    '''
    M = torch.zeros([nx]).to(device)
    M[1:-1] = 1.
    
    return M, A, Kx, Kxx

##############################################################################

class Wave1D_PGNNcell(torch.nn.Module):
    ''' A single time-step of the 1D wave equation
    '''
    def __init__(self, C, config):

        super().__init__()

        _, self.A, self.Kx, self.Kxx = wave_matrices(C, config)
        self.dt = config['dt']
        nx = config['nx']
        self.fixedBC = config['bc']=='fixed'

        # Injecting at source position (single source only)
        self.ptsrc = Point_Source( config['x_s'], config )
        self.fscale = torch.FloatTensor( C**2 / config['dx'] ).to(device)

    def forward(self, H, src ):

        A = self.A;  Kx = self.Kx;  Kxx = self.Kxx
        dt = self.dt

        # Retrieve the internal (hidden) state
        # Using first-derivative system, 
        #   time positions  Previous (P)  Current (C)  and  Next (N)
        uC,vC,uP,vP = [ H[0], H[1], H[2], H[3] ]

        # Inject the input signal
        f = self.ptsrc.set(src) * self.fscale

        # Stiffness -- interior points 
        #   Using  Kxx uN = Kxx (uC + dt * vC)
        # (computing all, but endpoints will be replaced)
        k = torch.sum( \
            torch.stack([torch.roll(uC,1), uC, torch.roll(uC,-1)]) * Kxx,\
            axis=0)
        k += dt * torch.sum( \
            torch.stack([torch.roll(vC,1), vC, torch.roll(vC,-1)]) * Kxx,\
            axis=0)
        
        # boundary points (0 lbdry, -1 rbdry)
        lbdry = 0; rbdry = -1  # boundary indices
        nc = 3  # spatial derivative coefficients
        if self.fixedBC:
            k[lbdry] = 0.
            k[rbdry] = 0.
        else:
            # Using  Dx uN = Dx (uC + dt * vC)
            k[lbdry] =      torch.sum( uC[ :+nc] * Kx[:,lbdry], axis=0) \
                     + dt * torch.sum( vC[ :+nc] * Kx[:,lbdry], axis=0)
            k[rbdry] =      torch.sum( uC[-nc: ] * Kx[:,rbdry], axis=0) \
                     + dt * torch.sum( vC[-nc: ] * Kx[:,rbdry], axis=0)

        vN = A[0] * ( dt*(f+k) + A[1]*vC + A[2]*vP )
        
        vN[lbdry] = k[lbdry]
        vN[rbdry] = k[rbdry]
 
        uN = A[0] * ( dt*vN   + A[1]*uC + A[2]*uP )

        # Return the new internal state
        return [uN,vN,uC,vC]


##############################################################################

class Wave1D_Propagator(torch.nn.Module):
    #def forward_operator(C, config):

    def __init__(self, C, config):
    
        super().__init__()
        self.C = C
        self.config = config
        self.nx = config['nx']
        self.nt = config['nt']
        self.x_s = config['x_s']
        self.nrec = len(config['x_r'])

        # Define the source amplitudes
        ts = torch.linspace(0, config['T'], config['nt']).to(device)
        self.ws = ricker(ts, config)

        # Receiver geometry
        self.rcvrs = config['rcv']

        # Set up the PGNN cell stepper
        self.cell = Wave1D_PGNNcell(C, config)
        H1 = torch.zeros([self.nx], device=device)
        self.H = [H1, H1.clone(), H1.clone(), H1.clone()]

    def forward(self):

        # Generate wavefields
        us = [] 	# list of output wavefields
        traces = [] 
        rcv = self.rcvrs

        # 4 hidden state components
    
        for it in range(self.nt):

            self.H = self.cell.forward( self.H, self.ws[it] )

            us.append( self.H[0].cpu().detach().numpy() )

            # Extract wavefield sample at each receiver position
            if self.nrec>0:
                samps = rcv.sample( self.H[0].clone() )
                traces.append( samps )

        trc = None if self.nrec==0 else torch.stack(traces,dim=1)

        return us, trc 


##############################################################################
# Problem 1.5

def plot_space_time(u, config, overlay=None, title=None):

    # Plots a wavefield u
    xlim = config['x_limits']
    xlen = xlim[1] - xlim[0]
    T = config['T']
    dt = config['dt']
    nt = len(u)
    nx = len(u[0])
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
    plt.colorbar()
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

##############################################################################
# Problem 2.3

def imaging_condition(qs, u0s, config):

    # Imaging condition of trial wavefield u0s with adjoint field qs
    nx = config['nx']
    nt = config['nt']
    dt = config['dt']
    dx = config['dx']
    dtscl = 1./dt**2
    uprev = np.zeros( nx )
    ucurr = np.zeros( nx )
    unext = u0s[0]
    image = np.zeros( nx )

    # Integration loop, reversed through adjoint wavefield
    for it in range(nt):
        uprev = ucurr
        ucurr = unext
        if it==nt-1:
            unext = np.zeros( nx )
        else:
            unext = u0s[it+1]
        Irtm = qs[nt-it-1]*Dtt( uprev, ucurr, unext ) 
        image -= Irtm*dtscl 
    image[0] = 0.;  image[-1] = 0.

    return image

def Dtt( uprev, ucurr, unext ):
    # Second time derivative
    return uprev - 2.*ucurr + unext

##############################################################################
# Problem 2.5

def adjoint_operator(C0, d, config):

    # Forward wavefields
    u0s, _ = forward_operator(C0, config)

    # Generate adjoint sources
    nt = config['nt']
    adjsrc = []
    for it in range(nt):
        t = it*config['dt']
        # Source in reverse-time order
        f = point_source(d[nt-it-1], config['x_r'], config)
        f[0] = 0.;  f[-1] = 0.
        adjsrc.append( f.copy() )

    # Generate adjoint wavefields
    qs = advance(C0, adjsrc, config)

    # Compute the image
    image = imaging_condition(qs, u0s, config)

    return image

##############################################################################
# Problem 3.1

def linear_sources(dm, u0s, config):

    # Source wavefields for linear modeling equations:
    #  -dm(x) * Dtt u0(x,t)
    nx = config['nx']
    nt = config['nt']
    dt = config['dt']
    dmp = dm.copy()
    dmp[0] = 0.;  dmp[-1] = 0.
    dtscl = 1./dt**2

    uprev = np.zeros( nx )
    ucurr = np.zeros( nx ) 
    unext = u0s[0]
    sources = []

    for it in range(nt):
        uprev = ucurr.copy()
        ucurr = unext.copy()
        if it==nt-1:
            unext = np.zeros( nx )
        else:
            unext = u0s[it+1]
        srct = -dmp*dtscl*Dtt( uprev, ucurr, unext )
        sources.append( srct.copy() )

    return sources

##############################################################################
# Problem 3.2

def linear_forward_operator(C0, dm, config):

    # Propagate with linearized source from model perturbation
    u0s, _ = forward_operator(C0, config)
    linsrc = linear_sources(dm, u0s, config)

    # Generate wavefields
    u1s = advance(C0, linsrc, config)

    # Extract wave data at receiver
    trace = record_data(u1s, config)

    return u1s, trace

##############################################################################
# Problem 3.3

def adjoint_condition(C0, config):
    '''
    Test for equality of:
        dot( Sampled(Forward(dm)), data )  [in data domain] and
        dot( dm, Adjoint(Sampled(data)) )  [in model domain]
    where 'dm' and 'data' are random model and data values.
    '''

    # using random values
    nt = config['nt']
    nx = config['nx']
    data = np.random.random((nt,))
    data[0] = 0.;  data[-1] = 0.
    dm = np.random.random((nx,))
    dm[0] = 0.;  dm[-1] = 0.

    # LHS (data domain)
    _, dfs = linear_forward_operator(C0, dm, config)
    LHS = np.dot( dfs, data )

    # RHS (model domain)
    adjt = adjoint_operator(C0, data, config)
    RHS = np.dot( dm, adjt )

    relerr = abs(LHS-RHS)/abs(LHS+RHS)

    print("Adjoint condition: LHS= %g. RHS= %g"%(LHS,RHS) ) 
    return relerr

##############################################################################
##############################################################################
class Wave1D_AGfunc(object):    
    ''' Autogradient implementation of 1D wave equation
    '''
    # Model and survey parameters are maintained in 'config'
    config = dict()
    C = None
    C0 = None

    def __init__(self,args):

    # Default configuration values
        flow = 'forward'
        bc = 'absorbing' # boundary conditions
        nu0 = 10.  # central frequency, Hz

        # Model domain
        x_limits = [0.0, 1.0]
        vBase = 1.0
        nx = 201
        dx = 0.

        # Source and receiver positions
        x_s = 0.1
        x_r = [0.15]

        # Time step parameters
        alpha = 1./6.    # CFL stability coefficient
        T = 3.0    # seconds


   # Parsed arguments are retrieved here...
        # x_limits[-1] = 2.
        # bc = 'fixed'
        T = 2.
        # nx = 301
        # vBase = 1.5
        x_r = [0.15, 0.25, 0.35, 0.8]
        flow = 'inverse'
        pass

   # Complete the setup...
        config = dict() 	# local temporary dictionary
        config['torch_device'] = device
        config['flow'] = flow
        config['bc'] = bc 	# boundary conditions
        config['nu0'] = nu0  	# central frequency, Hz

        config['x_limits'] = x_limits
        config['nx'] = nx
        if dx==0. :
            dx = (x_limits[1] - x_limits[0]) / (nx-1)
        config['dx'] = dx

        config['x_s'] = x_s
        config['x_r'] = x_r

        # Load the model
        #C, C0 = basic_model(config, vBase=vBase) 
        C, C0 = basic_seismic_model(config) 
        self.C = torch.from_numpy(C).float().to(device)
        self.C0 = torch.from_numpy(C0).float().to(device)

        # Receiver geometry
        config['rcv'] = Receivers(config)

        dt = alpha * dx / self.C.max()
        nt = int( T / dt )
        dt = T / (nt-1)
        config['alpha'] = alpha   
        config['T'] = T  
        config['dt'] = dt
        config['nt'] = nt

        self.config = config

    def __str__(self):
        s = 'Wave1D_AGfunc config:'
        for key,val in self.config.items():
            s += '\n  %s \t: '%key + str(val)
        return s

    def run(self):
      config = self.config

      ts = np.linspace(0, config['T'], config['nt'], endpoint=False)

      if config['flow']=='forward' or config['flow']=='inverse':

        # Generate wavefields from true model
        propC = Wave1D_Propagator(self.C,config)
        print("Forward true model running...")
        time1 = time.time()
        us, d = propC.forward()
        time2 = time.time()
        print("Forward true model time: %g sec."%(time2 - time1))
        dn = d.squeeze(0).cpu().detach().numpy()
        #print( d.size(),d,'\n', dn )
        # Plot wavefields and trace data
        #plot_space_time(us, config, title=r'Wavefield u($x,t$)')

        x_r = config['x_r']
        for itr in range(len(x_r)):
          plt.figure()
          plt.plot(ts, dn[itr,:], \
            label=r'$x_r =\,{0}$'.format(x_r[itr]), linewidth=2)
          plt.xlabel(r'$t$', fontsize=16)
          plt.ylabel(r'$d(t)$', fontsize=16)
          plt.title('Trace at $x_r={0}$'.format(x_r[itr]), \
            fontsize=18)
          plt.legend()
      #plt.show()

      if config['flow']=='inverse':
        learning=0.5
        num_steps=10
        C = self.C
        C0 = self.C0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([C0],lr=learning)
        #optimizer = torch.optim.SGD([C0],lr=learning)
        #optimizer = torch.optim.RMSprop([C0],lr=learning)
        #lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: learning * (0.99 ** epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, \
                 lr_lambda=[lambda2])
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.989)
        criterion = torch.nn.SmoothL1Loss()

        for i in range(num_steps):

            print("\nForward trial model step %d running..."%i )
            start = time.time()
            C1 = C0.to(device)

            #torch.autograd.set_detect_anomaly(False)
            propC1 = Wave1D_Propagator(C1,config)
            print("Create propagator")
            us0, d0 = propC1.forward()
            end_forward = time.time()
            print("Forward trial model time: %g sec."%(end_forward - start))

            loss = criterion(d0,d)

            print("step %d: rate= %g  loss= %g "% \
                (i, learning*(0.95**i), loss.item()) )
            optimizer.zero_grad()
            print("Backward trial model step %d running..."%i )
            #loss.backward(retain_graph=True)
            loss.backward()
            end_backward = time.time()
            print("Backward timing: %g sec."%(end_backward-end_forward))
            optimizer.step()
            scheduler.step(i)

        # Apply smoothing
        C1 = gaussian_filter(C0.cpu().detach().numpy(), sigma=2)

        plt.figure()
        num = d.shape[0]
        plt.title('Recorded and modeled data traces')
        for i in range(num):
            plt.subplot(num, 1, i+1)
            plt.plot(ts, d[i].cpu().detach().numpy(),  linewidth=1)
            plt.plot(ts, d0[i].cpu().detach().numpy(), linewidth=1)

        xs = np.arange(config['nx'])*config['dx']
        plt.xlabel(r'$t$', fontsize=16)
        plt.ylabel(r'$d(t)$', fontsize=16)
        plt.savefig('Figures/trace%d.png'%num)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(xs, C.cpu().detach().numpy(), label=r'$C$: True model')
        plt.plot(xs, C0.cpu().detach().numpy(), label=r'$C0$: Inverted model')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(xs, C1, label=r'$C1$: Smoothed inverse')
        plt.legend()
        plt.savefig('Figures/inverted_model%d.png'%num)
        '''
    # Residual trace from proposal model C0
        u0s, d0 = forward_operator(self.C0, config)
        r = d - d0

        plt.figure()
        plt.plot(ts, r, \
            label=r'$x_r =\,{0}$'.format(config['x_r']), linewidth=2)
        plt.xlabel(r'$t$', fontsize=16)
        plt.ylabel(r'$d(t)$', fontsize=16)
        plt.title('Residual trace $x_r={0}$'.format(config['x_r']),\
            fontsize=18)
        plt.legend()

    # Compute the image from the residual data
        I_rtm = adjoint_operator(self.C0, r, config)

        # Plot the comparison of image against model
        xs = np.arange(config['nx'])*config['dx']
        dC = self.C - self.C0

        plt.figure()
        plt.subplot(2, 1, 1, title='Model perturbation and RTM image')
        plt.plot(xs, self.C, label=r'$C$')
        plt.plot(xs, self.C0, label=r'$C0$')
        plt.plot(xs, dC, label=r'$\delta C$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(xs, I_rtm, label=r'$I_{RTM}$')
        plt.legend()

    # Validate the adjoint condition 
        # config['bc'] = 'fixed'
        relerr = adjoint_condition(self.C0, config)
        print('relerr=',relerr)
        '''

      #plt.show()

#############################################################################

if __name__ == '__main__':

    demo = Wave1D_AGfunc( parseArgs() )

    print( demo )

    demo.run()

