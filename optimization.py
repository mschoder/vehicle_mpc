import numpy as np
from pydrake.all import (Variable, MathematicalProgram,
                         OsqpSolver, Solve, SnoptSolver, eq)
from pydrake.solvers import branch_and_bound, gurobi
import pydrake.symbolic as sym
import time
import dynamics
import environment
import trajectory_gen


# GLOBALS
n_x = 5
n_u = 2
buffer = 0.3
Q = np.diag([1, 1, .05, .1, 0])
R = np.diag([.1, .01])
Q_slack = 10
bounds = {
    'accel_max': 10,
    'accel_min': -10,
    'steerv_max': 4,
    'steer_max_deg': 60,
    'v_max': 20,
}


##### Optimization for linearized dynamics #####
def linear_optimization(env, x_ref, x_bar, u_bar, x0, N, dt):
    ''' Builds and solves single step linear optimization problem
        with convex polygon obstacles and linearized dynamics
    '''
    prog, dvs = initialize_problem(n_x, n_u, N, x0)
    prog = linear_dynamics_constraints(prog, dvs, x_bar, u_bar, N, dt)
    prog = add_input_state_constraints(prog, dvs, bounds)
    prog, dvs = add_polyhedron_obstacle_constraints(
        prog, dvs, env.obstacles, buffer, N)
    prog = add_costs(prog, dvs, env, x_ref, Q, R, N)

    solver = gurobi.GurobiSolver()
    # solver.AcquireLicense()
    result = solver.Solve(prog)
    # assert(result.is_success), "Optimization Failed"
    status = result.get_solver_details().optimization_status
    assert(status == 2), "Optimization failed with code: " + str(status)

    x, u, z, v = dvs
    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    z_sol = [result.GetSolution(zj) for zj in z]
    v_sol = [result.GetSolution(vj) for vj in v]
    return result, (x_sol, u_sol, z_sol, v_sol)


def iterative_linear_mpc(env, x_ref, x_bar, u_bar, x0, N, dt, max_iters=5):
    '''
    Iteratively run optimization for linearized dynamics until input `u` converges
    with nominal input `u_bar`, which helps to reduce linearization error
    '''
    start = time.perf_counter()
    eps = 0.05

    if u_bar is None:
        u_opt = np.zeros((n_u, N-1))
    else:
        u_opt = u_bar

    if x_bar is None:
        x_bar = dynamics.rollout(x0, u_opt, n_x, dt)

    nominals = [x_bar]  # container for nominal traj at each iteration

    for i in range(max_iters):
        u_prev = u_opt
        result, (x_opt, u_opt, z_opt, v_opt) = linear_optimization(
            env, x_ref, x_bar, u_opt, x0, N, dt)
        u_diff = np.sum(np.abs(u_opt - u_prev))
        if u_diff < eps:
            break

        if len(u_opt.shape) == 1:
            # handle 1D array output for last iteration
            u_opt = u_opt.reshape(u_opt.shape[0], -1)
        x_bar = dynamics.rollout(x0, u_opt, n_x, dt)
        nominals.append(x_bar)

    end = time.perf_counter()
    print('\t', i+1, ' iterations, final control diff: ', np.round(u_diff, 6),
          ', Elapsed time (s): ', np.round(end-start, 4))
    return result, (x_opt, u_opt, z_opt, v_opt), x_bar, nominals


##### Nonlinear optimization problem #######
def nonlinear_optimization(env, x_ref, x_bar, u_bar, x0, N, dt):
    ''' Builds and solves single step linear optimization problem
    with convex polygon obstacles and linearized dynamics
    '''
    prog, dvs = initialize_problem(n_x, n_u, N, x0)
    prog = nonlinear_dynamics_constraints(prog, dvs, x_bar, u_bar, N, dt)
    prog = add_input_state_constraints(prog, dvs, bounds)
    prog, dvs = add_polyhedron_obstacle_constraints(
        prog, dvs, env.obstacles, buffer, N)
    prog = add_costs(prog, dvs, env, x_ref, Q, R, N)

    # Use SNOPT solver
    solver = SnoptSolver()
    result = solver.Solve(prog)
    assert(result.is_success), "Optimization Failed"

    x, u, z, v = dvs
    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    z_sol = [result.GetSolution(zj) for zj in z]
    v_sol = [result.GetSolution(vj) for vj in v]
    return result, (x_sol, u_sol, z_sol, v_sol), x_bar

##### Helper functions ######


def initialize_problem(n_x, n_u, N, x0):
    ''' returns base MathematicalProgram object and empty decision vars '''
    prog = MathematicalProgram()

    # Decision Variables
    x = prog.NewContinuousVariables(n_x, N, 'x')
    u = prog.NewContinuousVariables(n_u, N-1, 'u')
    z = []  # placeholder for obstacle binary variables
    q = []  # placeholder for obstacle slack buffer variables

    # initial conditions constraints
    prog.AddBoundingBoxConstraint(x0, x0, x[:, 0])

    decision_vars = [x, u, z, q]
    return prog, decision_vars


def linear_dynamics_constraints(prog, decision_vars, x_bar, u_bar, N, dt):
    '''
    Dynamics Constraints -- error state form
    x_bar - nominal point; dx = (x - x_bar)
    A, B, C = get_linear_dynamics(derivs, x_bar[:,n], u_bar[:,n])
    '''
    x, u, _, _ = decision_vars
    derivs = dynamics.derivatives(dynamics.discrete_dynamics, n_x, n_u)
    for n in range(N-1):
        A, B = dynamics.get_linear_dynamics(derivs, x_bar[:, n], u_bar[:, n])
        dx = x[:, n] - x_bar[:, n]
        du = u[:, n] - u_bar[:, n]
        dxdt = A@dx + B@du
        fxu_bar = dynamics.car_continuous_dynamics(x_bar[:, n], u_bar[:, n])
        for i in range(n_x):
            prog.AddConstraint(x[i, n+1] == x[i, n] +
                               (fxu_bar[i] + dxdt[i])*dt)
    return prog


def nonlinear_dynamics_constraints(prog, decision_vars, x_bar, u_bar, N, dt):
    ''' Add nonlinear dynamics constraints in original form '''
    x, u, _, _ = decision_vars
    for n in range(N-1):
        psi = x[2, n]
        v = x[3, n]
        steer = x[4, n]
        prog.AddConstraint(x[0, n+1] == x[0, n] + dt*v*sym.cos(psi))
        prog.AddConstraint(x[1, n+1] == x[1, n] + dt*v*sym.sin(psi))
        prog.AddConstraint(x[2, n+1] == x[2, n] + dt*v*sym.tan(steer))
        prog.AddConstraint(x[3, n+1] == x[3, n] + dt*u[0, n])
        prog.AddConstraint(x[4, n+1] == x[4, n] + dt*u[1, n])
    return prog


def add_input_state_constraints(prog, decision_vars, bounds):
    ''' Docstring '''
    x, u, _, _ = decision_vars
    b = bounds

    # input bounds on each timestep n
    prog.AddBoundingBoxConstraint(
        b['accel_min'], b['accel_max'], u[0, :])  # acceleration
    # steering velocity
    prog.AddBoundingBoxConstraint(-b['steerv_max'], b['steerv_max'], u[1, :])

    # Velocity bounds on each timestep (velocity must be positive)
    prog.AddBoundingBoxConstraint(0, b['v_max'], x[3, :])

    # Steer angle
    max_steer = np.deg2rad(b['steer_max_deg'])
    prog.AddBoundingBoxConstraint(-max_steer, max_steer, x[4, :])

    return prog


def add_polyhedron_obstacle_constraints(prog, decision_vars, obs_list, buffer, N):
    M = 1e6
    x, u, z, q = decision_vars
    z = [[]]*len(obs_list)
    q = [[]]*len(obs_list)
    for j, ob in enumerate(obs_list):
        constraints = environment.linear_obstacle_constraints(ob, buffer)
        K = len(constraints)
        z[j] = prog.NewBinaryVariables(rows=N, cols=K)
        q[j] = prog.NewContinuousVariables(rows=N, cols=K)
        prog.AddBoundingBoxConstraint(0, 0, q[j][0, :])  # first one is unused

        for n in range(1, N):  # Don't constrain initial position in case already inside obstacle
            for k in range(K):
                a, b, c = constraints[k]
                # format: cy <= ax + b - buffer + Mz
                prog.AddConstraint(
                    c*x[1, n] <= a*x[0, n] + b + M*z[j][n, k] + q[j][n, k])
                prog.AddConstraint(q[j][n, k] >= 0)
            prog.AddConstraint(z[j][n, :].sum() <= K-1)

    dvs = [x, u, z, q]
    return prog, dvs


def add_costs(prog, decision_vars, env, x_ref, Q, R, N):
    ''' Quadratic costs of form xTQx + uTRu '''
    x, u, z, q = decision_vars
    J = len(env.obstacles)
    for n in range(N):
        prog.AddQuadraticErrorCost(Q, x_ref[:, n], x[:, n])
        if (n < N - 1):
            b = np.zeros(2)
            prog.AddQuadraticCost(R, b, u[:, n])

    slack_cost = Q_slack*np.sum([q[j].sum() for j in range(J)])
    prog.AddCost(slack_cost)

    return prog


if __name__ == "__main__":

    # Test
    pass
