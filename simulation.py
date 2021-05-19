import numpy as np
import dynamics
import optimization


def simulate_lmpc(env, x_ref_full, noise, x0, N, dt, history={}):

    n_x = x_ref_full.shape[0]
    n_u = 2
    sigmas = noise
    nsteps = x_ref_full.shape[1]

    x_bar = x_ref_full[:, 0:N]
    u_bar = u_bar = np.zeros((n_u, N-1))

    n = N
    for s in range(nsteps):
        print('Step ', s, '/', nsteps)

        # Get new reference data, stepped by one
        end = N + s
        if end > x_ref_full.shape[1]:
            n = x_ref_full.shape[1] - s
            end = x_ref_full.shape[1]

        x_ref = x_ref_full[:, s:end]
        result, (x_opt, u_opt, z_opt, v_opt), x_bar, _ = optimization.iterative_linear_mpc(
            env, x_ref, x_bar, u_bar, x0, n, dt, max_iters=5)

        if len(x_opt.shape) == 1:  # handle 1D array output for last iteration
            x_opt = x_opt.reshape(x_opt.shape[0], -1)

        if u_opt.size > 0:

            if len(u_opt.shape) == 1:  # handle 1D array output for last iteration
                u_opt = u_opt.reshape(u_opt.shape[0], -1)

            # Evolve dynamics and get updated reference and nominal data
            u0 = u_opt[:, 0]
            x0 = dynamics.step_car_dynamics(x0, u0, sigmas, dt)

            if N + s >= x_ref_full.shape[1]:
                u_bar = u_opt[:, 1:]
            else:  # repeat last control as starting seed
                u_bar = np.hstack((u_opt[:, 1:], u_opt[:, -1:]))

            if len(u_opt.shape) == 1:  # handle 1D array output for last iteration
                u_bar = u_bar.reshape(u_bar.shape[0], -1)

            x_bar = dynamics.rollout(x0, u_bar, n_x, dt)

        # Log results
        history[s] = {'x_opt': x_opt, 'u_opt': u_opt, 'z_opt': z_opt,
                      'v_opt': v_opt, 'x_bar': x_bar, 'x0': x0, 'result': result}

    return history
