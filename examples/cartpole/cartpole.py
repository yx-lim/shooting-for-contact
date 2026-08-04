##
#
# Cartpole swing-up TO.
#
##

# directory imports
import sys
import os

ROOT = os.getenv("TRAJOPT_ROOT_DIR")
sys.path.append(ROOT)

# standard imports
import numpy as np

# local imports
from src.dynamics import DynamicsConfig
from src.multi_shooting import MultiShootingConfig, MultiShootingBase
from utils.file_utils import save_trajectory


########################################################################
# CARTPOLE SWINGUP
########################################################################

class CartpoleSwingUp(MultiShootingBase):
    """ Quadratic cost on a lifted output that uses sin/cos of the pole angle. """

    def __init__(self, dynamics_config: DynamicsConfig, config: MultiShootingConfig):
        super().__init__(dynamics_config, config)

        # Cost weights on the lifted output y = [cart, sin, cos, cart_v, pole_v].
        self.Q = np.diag([1.0, 1.0, 1.0, 0.1, 0.1])
        self.R = np.array([[0.01]])
        self.Qf = 50.0 * self.Q

        # goal state (upright) -> lifted goal output
        self.x_goal = np.array([0.0, 0.0, 0.0, 0.0])
        self.y_goal = self.output(self.x_goal)

    @staticmethod
    def output(x):
        """Lift the state: y = [cart, sin(theta), cos(theta), cart_vel, pole_vel]."""
        p, th, pdot, thdot = x
        return np.array([p, np.sin(th), np.cos(th), pdot, thdot])

    @staticmethod
    def output_jacobian(x):
        """dy/dx, shape (5, 4)."""
        _, th, _, _ = x
        J = np.zeros((5, 4))
        J[0, 0] = 1.0           # d cart / d cart
        J[1, 1] = np.cos(th)    # d sin(theta) / d theta
        J[2, 1] = -np.sin(th)   # d cos(theta) / d theta
        J[3, 2] = 1.0           # d cart_vel / d cart_vel
        J[4, 3] = 1.0           # d pole_vel / d pole_vel
        return J

    def stage_cost(self, x, u, k):
        dy = self.output(x) - self.y_goal
        return float(0.5 * (dy @ self.Q @ dy + u @ self.R @ u))

    def stage_cost_grad(self, x, u, k):
        dy = self.output(x) - self.y_goal
        J = self.output_jacobian(x)
        return J.T @ self.Q @ dy, self.R @ u

    def terminal_cost(self, x):
        dy = self.output(x) - self.y_goal
        return float(0.5 * dy @ self.Qf @ dy)

    def terminal_cost_grad(self, x):
        dy = self.output(x) - self.y_goal
        J = self.output_jacobian(x)
        return J.T @ self.Qf @ dy


def initial_guess(n_nodes, n_steps, nx, nu):
    """Linearly interpolate the pole angle pi -> 0 over the shooting nodes."""
    X = np.zeros((n_nodes + 1, nx))
    X[:, 1] = np.linspace(np.pi, 0.0, n_nodes + 1)  # pole_angle
    U = np.zeros((n_steps, nu))
    return X, U


########################################################################
#  MAIN
########################################################################

if __name__ == "__main__":
    
    # model 
    xml_path = os.path.join(ROOT, "models", "cartpole", "cartpole.xml")

    # MPC config
    N = 200           # traj opt horizon (shooting nodes)
    sim_dt = 0.02     # sim time step [s]
    node_dt = sim_dt  # node time step [s]

    # initial state
    x_init = np.array([0.0, np.pi, 0.0, 0.0])

    # dynamics config
    dyn_cfg = DynamicsConfig(model_path=xml_path,
                             sim_dt=sim_dt)

    # state box bounds (x = [cart, theta, cart_vel, pole_vel])
    cart_limit = 0.5
    x_lb = np.array([-cart_limit, -np.inf, -np.inf, -np.inf])
    x_ub = np.array([ cart_limit,  np.inf,  np.inf,  np.inf])

    # multi-shooting config
    ms_cfg = MultiShootingConfig(
        N=N,
        node_dt=node_dt,
        terminal_constraint=True,   # hard terminal constraint: x_N = x_goal
        u_lb=-150.0,
        u_ub= 150.0,
        x_lb=x_lb,
        x_ub=x_ub,
        ipopt_options={
            "print_level": 5,
            "max_iter": 1000,
            "tol": 1e-6,
            "hessian_approximation": "limited-memory",
        },
    )

    # initialize the problem and set the initial condition (x_0 = x_init)
    problem = CartpoleSwingUp(dyn_cfg, ms_cfg)
    problem.set_initial_state(x_init)
    X0, U0 = initial_guess(problem.N, problem.N_sim, problem.nx, problem.nu)

    # solve
    X, U, info = problem.solve(X0, U0)

    # save the trajectory for replay on the fine sim grid: time, state, input, model
    X_fine = problem.stitched_trajectory(X, U)
    t = sim_dt * np.arange(problem.N_sim + 1)
    save_path = os.path.join(ROOT, "examples", "cartpole", "cartpole.npz")
    save_trajectory(save_path, time=t, state=X_fine, input=U, model=xml_path,
                    spline_type=problem.spline.spline_type)
    print("saved:", save_path)
