##
#
# Cartpole swing-up MPC.
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
from src.mpc import ModelPredictiveControl, ModelPredictiveControlConfig
from src.spline import SplineConfig
from utils.file_utils import save_trajectory


########################################################################
# CARTPOLE MPC
########################################################################

class CartpoleMPC(ModelPredictiveControl):
    """ Quadratic cost on a lifted output that uses sin/cos of the pole angle.

    State:  x = [cart, pole_angle, cart_vel, pole_vel]
        hanging down -> [0, pi, 0, 0]
        upright      -> [0,  0, 0, 0]
    Output: y(x) = [cart, sin(theta), cos(theta), cart_vel, pole_vel]
    Using sin/cos instead of the raw angle makes the cost invariant to 2*pi wraps.
    """

    def __init__(self, dynamics_config: DynamicsConfig, config: ModelPredictiveControlConfig):
        super().__init__(dynamics_config, config)

        # cost weights on the lifted output y = [cart, sin, cos, cart_v, pole_v]
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


########################################################################
#  MAIN
########################################################################

if __name__ == "__main__":

    # model
    xml_path = os.path.join(ROOT, "models", "cartpole", "cartpole.xml")

    # MPC parameters
    H = 50           # prediction horizon (shooting nodes)
    T = 5.0          # total closed-loop sim time [s]
    sim_dt = 0.02    # sim time step [s]
    node_dt = 0.02   # node time step [s]
    mpc_dt = 0.02    # MPC time step [s]

    # initial state (hanging down)
    x_init = np.array([0.0, np.pi, 0.0, 0.0])

    # dynamics config
    dyn_cfg = DynamicsConfig(model_path=xml_path,
                             sim_dt=sim_dt)

    # spline config
    spline_cfg = SplineConfig(M=15,
                              spline_type="linear")

    # state box bounds (x = [cart, theta, cart_vel, pole_vel])
    cart_limit = 0.5
    x_lb = np.array([-cart_limit, -np.inf, -np.inf, -np.inf])
    x_ub = np.array([ cart_limit,  np.inf,  np.inf,  np.inf])

    # mpc config
    cfg = ModelPredictiveControlConfig(
        N=H,
        node_dt=node_dt,
        mpc_dt=mpc_dt,
        u_lb=-100.0,
        u_ub= 100.0,
        x_lb=x_lb,
        x_ub=x_ub,
        spline=spline_cfg,
        ipopt_options={
            "print_level": 0,
            "max_iter": 5,   
            "tol": 1e-2,
            "hessian_approximation": "limited-memory",
        },
        T=T,
        verbose=True,
    )

    # instantiate the MPC
    mpc = CartpoleMPC(dyn_cfg, cfg)

    # feasible warm start: sample an energy-pumping sinewave input over all
    # N_sim horizon sim steps (set_warm_start subsamples the states at the nodes)
    t_h = sim_dt * np.arange(mpc.N_sim)
    U_ws = 50.0 * np.sin(2.0 * np.pi * 0.5 * t_h).reshape(mpc.N_sim, mpc.nu)
    X_ws = mpc.dyn.rollout(x_init, U_ws)
    mpc.set_warm_start(X_ws, U_ws)

    # run the closed loop
    t, X, U = mpc.run(x_init)

    # save the closed-loop trajectory for replay: time, state, input, model
    save_path = os.path.join(ROOT, "examples", "cartpole_mpc", "cartpole_mpc.npz")
    save_trajectory(save_path, time=t, state=X, input=U, model=xml_path,
                    spline_type=mpc.spline.spline_type)
    print("saved:", save_path)
