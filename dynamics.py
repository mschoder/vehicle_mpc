import numpy as np
import pydrake.symbolic as sym


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


def discrete_dynamics(x, u, dt):
    # forward Euler
    x_next = x + dt * car_continuous_dynamics(x, u)
    return x_next


def rollout(x0, u_trj, n_x, dt):
    ''' Returns state trajectory projected in time, x_trj: [N, number of states] '''
    x_trj = np.zeros((n_x, u_trj.shape[1]+1))
    x_trj[:, 0] = x0
    for n in range(1, u_trj.shape[1]+1):
        x_trj[:, n] = discrete_dynamics(x_trj[:, n-1], u_trj[:, n-1], dt)
    return x_trj


class derivatives():
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


def step_car_dynamics(x0, u0, sigmas, dt):
    ''' For simulation '''
    n_x = x0.shape[0]
    xn = discrete_dynamics(x0, u0, dt)
    w = np.random.normal(0, sigmas, n_x)
    return xn + w


if __name__ == "__main__":
    dt = 0.1
    n_x = 5
    n_u = 2
    derivs = derivatives(discrete_dynamics, n_x, n_u)
    dfx, dfu = derivs.f_x, derivs.f_u
    print(dfx, dfu)
