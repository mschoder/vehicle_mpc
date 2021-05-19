
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc, patches
import matplotlib.collections as clt
import environment
import scenarios
import trajectory_gen

# Set jshtml default mode for notebook use
rc('animation', html='jshtml')


def plot_one_step(env, x_ref, x_bar, x_opt, x_next=None, nominals=None):
    ''' Plots a single step of trajectory optimization in environment '''
    fig, ax = environment.plot_environment(env, figsize=(16, 10))
    ax.plot(x_ref[0, :], x_ref[1, :], '-o', alpha=0.8,
            color='blue', markersize=3, label='reference trajectory')
    if nominals is not None:
        for nom in nominals:
            ax.plot(nom[0, :], nom[1, :], 'r-', label='nominal trajectories')
    else:
        ax.plot(x_bar[0, :], x_bar[1, :], 'r-', label='nominal trajectory')
    ax.plot(x_opt[0, :], x_opt[1, :], '-o',
            color='lightseagreen', label='optimal trajectory')
    if x_next is not None:
        ax.plot(x_next[0], x_next[1], 'o', color='blueviolet', label='next x0')

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='best')
    return fig, ax


def plot_all_steps(env, x_ref_full, history):
    ''' Plots optimization paths in environment at each step over time
        course of the full MPC run
    '''
    fig, ax = environment.plot_environment(env, figsize=(16, 10))
    ax.plot(x_ref_full[0, :], x_ref_full[1, :],
            '-o', color='blue', label='reference trajectory', markersize=3)
    for i in range(len(history.keys())):
        xi = history[i]['x_opt']
        ax.plot(xi[0, :], xi[1, :], color='lightseagreen',
                linewidth=1, label='N-step optimal traj')
    x0x = [history[i]['x0'][0] for i in history.keys()]
    x0y = [history[i]['x0'][1] for i in history.keys()]
    ax.plot(x0x, x0y, '-o', color='blueviolet', label='actual trajectory')
    xf_bar = history[len(history.keys())-1]['x_bar']
    # ax.plot(xf_bar[0, :], xf_bar[1, :], 'r', label='xf_bar')
    ax.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='best')
    return fig, ax


def animate(env, ctrl_pts, bc_headings, v, dt, x_ref, history, rect=False):

    fig, ax = environment.plot_environment(env, figsize=(16, 10))
    xs = x_ref[0, :]
    ys = x_ref[1, :]
    x0s = [history[i]['x0'][0] for i in history.keys()]
    y0s = [history[i]['x0'][1] for i in history.keys()]

    # plot reference trajectory
    ax.plot(xs, ys, '-o', alpha=0.8, markersize=3,
            color='blue', label='reference trajectory')
    # optimized trajectory
    opt_line, = ax.plot([], [], '-o', lw=3, color='lightseagreen',
                        label='N-step optimal traj')
    nom_line, = ax.plot([], [], color='red', label='nominal trajectory')
    act_line, = ax.plot([], [], '-o', lw=3, markersize=6,
                        color='blueviolet', label='actual trajectory')

    if rect:
        ld, wd = 0.5, 0.2
        a2 = np.arctan2(wd, ld)
        diag = np.sqrt(ld**2 + wd**2)
        heading = np.rad2deg(np.arctan2((y0s[1]-y0s[0]), (x0s[1]-x0s[0])))
        car = patches.Rectangle(
            (x0s[0]-ld, y0s[0]-wd), 2*ld, 2*wd, angle=-heading, fc='none', lw=1, ec='k')

    def init():
        opt_line.set_data([], [])
        act_line.set_data([], [])
        nom_line.set_data([], [])
        if rect:
            ax.add_patch(car)
        return opt_line,

    def step(i):
        x = history[i]['x_opt'][0]
        y = history[i]['x_opt'][1]
        xbar = history[i]['x_bar'][0]
        ybar = history[i]['x_bar'][1]
        opt_line.set_data(x, y)
        act_line.set_data(x0s[:i], y0s[:i])
        nom_line.set_data(xbar, ybar)
        if rect:
            if (len(x) == 1):
                heading = bc_headings[1]
            else:
                heading = np.arctan2((y[1]-y[0]), (x[1]-x[0]))
            xoff = diag*np.cos(heading + a2)
            yoff = diag*np.sin(heading + a2)
            car.set_xy((x[0] - xoff, y[0] - yoff))
            car.angle = np.rad2deg(heading)

        return opt_line,

    anim = animation.FuncAnimation(fig, step, init_func=init,
                                   frames=len(history.keys()), interval=1000*dt*2, blit=True)

    ax.axis('equal')
    ax.legend()
    plt.close()
    return anim


if __name__ == "__main__":

    # TEST ANIMATION FUNCTION

    scene = scenarios.five_obstacle
    ctrl_pts = scene['control_pts']
    bch = scene['bc_headings']
    env = environment.Environment(scene['obs_list'],
                                  scene['start'], scene['goal'])
    env.add_control_points(ctrl_pts)
    v = 4
    dt = 0.1
    xs, ys, psi = trajectory_gen.sample_trajectory(ctrl_pts, bch, v, dt)
    nf = len(xs)
    x_ref = np.vstack((xs.reshape((1, nf)),
                       ys.reshape((1, nf)),
                       psi.reshape((1, nf)),
                       v*np.ones((1, nf)),
                       np.zeros((1, nf))))

    anim = animate(env, ctrl_pts, bch, v, dt, x_ref, None)
    plt.show()
