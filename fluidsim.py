__author__ = 'Lei Huang'

import torch
import numpy as np
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.misc import imread

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
#@torch.jit.script
def roll(x: torch.Tensor, shift: int, axis: int = -1):

    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(axis, torch.arange(shift).long())
        return torch.cat([x.index_select(axis, torch.arange(shift, x.size(axis)).long()), gap], dim=axis)

    else:
        shift = x.size(axis) - shift
        gap = x.index_select(axis, torch.arange(shift, x.size(axis)).long())
        return torch.cat([gap, x.index_select(axis, torch.arange(shift).long())], dim=axis)
'''

#@torch.jit.script
def roll(x: torch.Tensor, shift: int, axis: int):
    if shift != 0:
        return x

    if axis == 0:
        return torch.cat((x[-shift:,:], x[:-shift,:]), dim=axis).to(device)
    else:
        return torch.cat((x[:,-shift:], x[:,:-shift]), dim=axis).to(device)

#@torch.jit.script
def project(vx, vy):
    """Project the velocity field to be approximately mass-conserving,
       using a few iterations of Gauss-Seidel."""
    p = torch.zeros(vx.shape).to(device)
    h = 1.0/torch.tensor(vx.shape[0]).float().to(device)
    div = -0.5 * h * (roll(vx, -1, axis=0) - roll(vx, 1, axis=0)
                    + roll(vy, -1, axis=1) - roll(vy, 1, axis=1))

    for k in range(10):
        p = (div + roll(p, 1, axis=0) + roll(p, -1, axis=0)
                 + roll(p, 1, axis=1) + roll(p, -1, axis=1))/4.0

    vx -= 0.5*(roll(p, -1, axis=0) - roll(p, 1, axis=0))/h
    vy -= 0.5*(roll(p, -1, axis=1) - roll(p, 1, axis=1))/h
    return vx, vy

#@torch.jit.script
def advect(f:torch.Tensor, vx:torch.Tensor, vy:torch.Tensor, cell_ys:torch.Tensor, cell_xs:torch.Tensor):
    """Move field f according to x and y velocities (u and v)
       using an implicit Euler integrator."""
    rows, cols = f.shape
    center_xs = (cell_xs - vx).view(-1).to(device)
    center_ys = (cell_ys - vy).view(-1).to(device)

    # Compute indices of source cells.
    left_ix = torch.floor(center_xs).to(device)
    top_ix  = torch.floor(center_ys).to(device)
    rw = (center_xs - left_ix).to(device)              # Relative weight of right-hand cells.
    bw = (center_ys - top_ix).to(device)               # Relative weight of bottom cells.
    left_ix  = torch.fmod(left_ix,     rows).long().to(device)  # Wrap around edges of simulation.
    right_ix = torch.fmod(left_ix + 1, rows).long().to(device)
    top_ix   = torch.fmod(top_ix,      cols).long().to(device)
    bot_ix   = torch.fmod(top_ix  + 1, cols).long().to(device)
    #print(left_ix,right_ix,top_ix,bot_ix)

    # A linearly-weighted sum of the 4 surrounding cells.
    flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
                 + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
    return flat_f.view(rows, cols)

#@torch.jit.script
def simulate(vx, vy, smoke, cell_ys, cell_xs, num_time_steps=200, ax=None, render=False):
    print("Running simulation...")
    for t in range(num_time_steps):
        if ax: plot_matrix(ax, smoke.cpu().detach().numpy(), t, render)
        vx_updated = advect(vx, vx, vy, cell_ys, cell_xs)
        vy_updated = advect(vy, vx, vy, cell_ys, cell_xs)
        vx, vy = project(vx_updated, vy_updated)
        smoke = advect(smoke, vx, vy, cell_ys, cell_xs)
    if ax: plot_matrix(ax, smoke.cpu().detach().numpy(), num_time_steps, render)
    return smoke

def plot_matrix(ax, mat, t, render=False):
    plt.cla()
    ax.matshow(mat)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    if render:
        matplotlib.image.imsave('step{0:03d}.png'.format(t), mat)
    plt.pause(0.001)

def distance_from_target_image(smoke, target):
    return torch.mean((target - smoke)**2)

def convert_param_vector_to_matrices(params, rows, cols):
    vx = params[:(rows*cols)].view(rows,cols).to(device) #
    #vx = np.reshape(params[:(rows*cols)], (rows, cols))
    vy = params[(rows*cols):].view(rows,cols).to(device) #
    #vy = np.reshape(params[(rows*cols):], (rows, cols))
    return vx, vy

def objective(params):
    init_vx, init_vy = convert_param_vector_to_matrices(params)
    final_smoke = simulate(init_vx, init_vy, init_smoke)
    return distance_from_target_image(final_smoke)

if __name__ == '__main__':

    simulation_timesteps = 200
    basepath = os.path.dirname(__file__)

    print("Loading initial and target states...")
    init_smoke = imread(os.path.join(basepath, 'init_smoke.png'))[:,:,0]
    #target = imread('peace.png')[::2,::2,3]
    target = imread(os.path.join(basepath, 'skull.png'))[::2,::2]
    rows, cols = target.shape

    init_dx_and_dy = np.zeros((2, rows, cols)).ravel()
    init_smoke = Variable(torch.from_numpy(init_smoke).float()).to(device)
    target = Variable(torch.from_numpy(target).float()).to(device)
    init_model = Variable(torch.from_numpy(init_dx_and_dy).float(), requires_grad=True)

    cell_ys, cell_xs = np.meshgrid(np.arange(rows), np.arange(cols))

    #print(cell_ys, cell_xs)
    cell_ys = torch.from_numpy(cell_ys).float().to(device)
    cell_xs = torch.from_numpy(cell_xs).float().to(device)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, frameon=False)

    #print("Rendering before optimization...")
    #init_vx, init_vy = convert_param_vector_to_matrices(init_model, rows, cols)
    #simulate(init_vx, init_vy, init_smoke, cell_ys, cell_xs, simulation_timesteps, ax, render=True)

    #print("Converting frames to an animated GIF...")
    #os.system("convert -delay 5 -loop 0 step*.png"
    #          " -delay 250 step100.png bef_opt.gif")  # Using imagemagick.
    #os.system("rm step*.png")

    print("Optimizing initial conditions...")
    optimizer = torch.optim.Adam([init_model],lr=0.01)


    for epoch in range(100):
        init_model1 = init_model.to(device=torch.device(device))
        init_vx, init_vy = convert_param_vector_to_matrices(init_model1, rows, cols)
        y_h = simulate(init_vx, init_vy, init_smoke, cell_ys, cell_xs, simulation_timesteps)
        loss = distance_from_target_image(y_h, target)
        optimizer.zero_grad()
        loss.backward()
        print(epoch, loss)
        optimizer.step()
        # print("grad: ", x_val, y_val, w.data, b.data, w.grad.data[0], b.grad.data[0])
        #init_model.data = init_model.data - 0.01 * init_model.grad.data
        #init_model.grad.data.zero_()

    print("Rendering optimized flow...")
    init_model1 = init_model.to(device=torch.device(device))
    init_vx, init_vy = convert_param_vector_to_matrices(init_model1, rows, cols)
    simulate(init_vx, init_vy, init_smoke, cell_ys, cell_xs, simulation_timesteps, ax, render=True)

    print("Converting frames to an animated GIF...")
    os.system("convert -delay 5 -loop 0 step*.png"
              " -delay 250 step100.png surprise.gif")  # Using imagemagick.
    os.system("rm step*.png")

