import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

from pydrake.all import (Variable,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, OsqpSolver, Solve, SnoptSolver, PiecewisePolynomial, eq)
from pydrake.solvers import branch_and_bound, gurobi
import pydrake.symbolic as sym
import gurobipy as gp
from gurobipy import GRB

# Local module files
import environment
import trajectory_gen
import scenarios


# Non-linear bicycle model
dt = 0.1
n_x = 5
n_u = 2


def car_continuous_dynamics(x, u):
    # x = [x position, y position, heading, speed, steering angle]
    # u = [acceleration, steering velocity]
    m = sym if x.dtype == object else np  # Check type for autodiff
    heading = x[2]
    v = x[3]
    steer = x[4]
    x_d = np.array([
        v*m.cos(heading),
        v*m.sin(heading),
        v*m.tan(steer),
        u[0],
        u[1]
    ])
    return x_d


def discrete_dynamics(x, u):
    # forward Euler
    x_next = x + dt * car_continuous_dynamics(x, u)
    return x_next


def rollout(x0, u_trj):
    ''' Returns state trajectory projected in time, x_trj: [N, number of states] '''
    x_trj = np.zeros((n_x, u_trj.shape[1]+1))
    x_trj[:, 0] = x0
    for n in range(1, u_trj.shape[1]+1):
        x_trj[:, n] = discrete_dynamics(x_trj[:, n-1], u_trj[:, n-1])
    return x_trj


class derivatives():
    # def __init__(self, discrete_dynamics, cost_stage, cost_final, n_x, n_u):
    def __init__(self, discrete_dynamics, n_x, n_u):
        self.x_sym = np.array([sym.Variable("x_{}".format(i))
                              for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i))
                              for i in range(n_u)])
        x = self.x_sym
        u = self.u_sym
        f = car_continuous_dynamics(x, u)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)

    def get_jacobians(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)
        return f_x, f_u


def get_linear_dynamics(derivs, x, u):
    ''' Linearize dynamics to take form dx = x[n+1] - x[n] = A*dx + B*du '''
    f_x, f_u = derivs.get_jacobians(x, u)
    A = f_x
    B = f_u
    return A, B


derivs = derivatives(discrete_dynamics, n_x, n_u)

# # Debug:
# x = np.array([1, 1, 0.5, 0.2, 0.05])
# u = np.array([.1, .1])
# fx, fu = derivs.get_jacobians(x, u)
# # print(derivs.get_jacobians(x, u))

# A,B = get_linear_dynamics(derivs, x, u)
# print(A,B)

# Linear MPC - Gurobi


def linear_optim(x_ref, x_bar, u_bar, x0, N, env):
    '''
    x_ref, u_ref -- reference trajectory (from external planner)
    x_bar, u_bar -- operating points for linearization
    x0, u0 -- initial conditions (current state)
    '''
    M = 1e6
    m = gp.Model()

    # Decision variables
    u = m.addMVar((n_u, N-1), vtype=GRB.CONTINUOUS)
    x = m.addMVar((n_x, N), vtype=GRB.CONTINUOUS)
    # slack variables z, etc defined below

    # Constraints
    for n in range(N-1):
        # Dynamics Constraints -- error state form
        # x_bar - nominal point; dx = (x - x_bar)
        # A, B, C = get_linear_dynamics(derivs, x_bar[:,n], u_bar[:,n])
        A, B = get_linear_dynamics(derivs, x_bar[:, n], u_bar[:, n])
#         dx = x[:,n] - x_bar[:,n]
#         du = u[:,n] - u_bar[:,n]

        Adx = A@x[:, n] - A@x_bar[:, n]
        Bdu = B@u[:, n] - B@u_bar[:, n]
        dxdt = Adx + Bdu

#         dxdt = A@dx + B@du
        fxu_bar = car_continuous_dynamics(x_bar[:, n], u_bar[:, n])
        m.addConstr(x[:, n+1] == x[:, n] + (fxu_bar + dxdt)*dt)
#         for i in range(n_x):
#             m.addConstr(x[i,n+1] == x[i,n] + (fxu_bar[i] + dxdt[i])*dt)

        # input bounds
        m.addConstr(u[0, n] >= -10.0)
        m.addConstr(u[0, n] <= 10.0)
        m.addConstr(u[1, n] >= -2.5)
        m.addConstr(u[1, n] <= 2.5)

    for n in range(N):
        # Velocity bounds on each timestep (velocity must be positive)
        m.addConstr(x[3, n] >= 0)
        m.addConstr(x[3, n] <= 20)

        # Steer angle
        max_steer = np.deg2rad(60.0)
        m.addConstr(x[4, n] >= -max_steer)
        m.addConstr(x[4, n] <= max_steer)

    # final conditions constraints on x, y
    xf = x_ref[:, N-1]
    dtol = 0.4
    htol = 0.05
    m.addConstr(x[0, N-1] >= xf[0]-dtol)
    m.addConstr(x[0, N-1] <= xf[0]+dtol)
    m.addConstr(x[1, N-1] >= xf[1]-dtol)
    m.addConstr(x[1, N-1] <= xf[1]+dtol)
    # prog.AddBoundingBoxConstraint(xf[2]-htol, xf[2]+htol, x[2,N-1])

    # Obstacles - polyhedrons
    obs = env.obstacles
    z = [[]]*len(obs)
    for j, ob in enumerate(obs):
        constraints = environment.linear_obstacle_constraints(ob)
        K = len(constraints)
        z[j] = m.addMVar((N, K), vtype=GRB.BINARY)
        for n in range(N):
            for k in range(K):
                a, b, c = constraints[k]
                # format: cy <= ax + b - buffer + Mz
                m.addConstr(c*x[1, n] <= a*x[0, n] + b + M*z[j][n, k])
            m.addConstr(z[j][n, :].sum() <= K-1)

    # COSTS
    # prog.AddQuadraticErrorCost(). # TODO - use this format
    cost = 0
    Q = np.diag([1, 1, .1, 0, 0])
    R = np.diag([0.01, 0])

    xe = m.addMVar((n_x, N), vtype=GRB.CONTINUOUS)
    ue = m.addMVar((n_u, N-1), vtype=GRB.CONTINUOUS)
    for n in range(N-1):

        #         print(type(gp.quicksum(xe[:,n]@Q@xe[:,n])))
        cost += xe[:, n]@Q@xe[:, n]
        cost += ue[:, n]@R@ue[:, n]

#         prog.AddCost(1.0*(x[0,n] - x_ref[0,n])**2 +
#                      1.0*(x[1,n] - x_ref[1,n])**2 +
#                      0.1*(x[2,n] - x_ref[2,n])**2 +
#                      0.1*u[0,n]**2 + 0.01*u[1,n]**2)

    m.setObjective(cost)
    m.optimize()

    return x.X, u.X, None, z.X


v = 1.5   # avg velocity
dt = 0.1
N = 40

scene = scenarios.two_obstacle
# control_pts = [(3, 0), (4.5, 1), (5, 2.8),
#                 (3.5, 3.4), (2, 5.1), (3.7, 6.7), (5.5, 6.3)]
control_pts = [(3, 0), (3.8, 1.3), (5, 2.8),
               (3.8, 4.1), (2, 5.1), (3.7, 6.7), (5.5, 6.3)]

bc_headings = (np.pi/6, -np.pi/6)
env = environment.Environment(scene['obs_list'],
                              scene['start'], scene['goal'])
env.add_control_points(control_pts)
ax = environment.plot_environment(env)

xc = [p[0] for p in control_pts]
yc = [p[1] for p in control_pts]

xs, ys, psi = trajectory_gen.sample_trajectory(xc, yc, bc_headings, v, dt)
ax.plot(xs, ys, 'ob', alpha=0.8, markersize=4)

# create full sampled reference trajectory
nf = len(xs)
x_ref_full = np.vstack((xs.reshape((1, nf)),
                        ys.reshape((1, nf)),
                        psi.reshape((1, nf)),
                        v*np.ones((1, nf)),
                        np.zeros((1, nf))))

x_ref = x_ref_full[:, 0:N]
x0 = x_ref[:, 0]


def iterative_mpc(model, env, x_ref, x_bar, x0, N, max_iters=10):

    start = time.perf_counter()
    eps = 0.05
    u_opt = np.zeros((2, N-1))

    if x_bar is None:
        x_bar = rollout(x0, u_opt)

    for i in range(max_iters):
        u_prev = u_opt
        x_opt, u_opt, result, z_opt = model(x_ref, x_bar, u_opt, x0, N, env)
        u_diff = np.sum(np.abs(u_opt - u_prev))
        if u_diff < eps:
            # print('u_diff for iter', i, ': ', u_diff, ' less than eps: ', eps)
            break
        # print('u_diff for iter ', i, ': ', u_diff)
        x_bar = rollout(x0, u_opt)
    end = time.perf_counter()
    print(i, ' iterations required, final eps is ', u_diff)
    print('Elapsed time for traj opt (s): ', end-start)
    return x_opt, u_opt, result, x_bar, z_opt


x_opt, u_opt, result, x_bar, z_opt = iterative_mpc(
    linear_optim, env, x_ref, None, x0, N)
