##
#
# Triple pendulum cartpole swing-up MPC.
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
# TRIPLE CARTPOLE MPC
########################################################################

class TripleCartpoleMPC(ModelPredictiveControl):
    """ Quadratic cost on a lifted output that uses sin/cos of all three hinge angles.

    State:  x = [cart, hinge_1, hinge_2, hinge_3, cart_vel, hinge_1_vel, hinge_2_vel, hinge_3_vel]
        hanging down -> [0, 0, 0, 0, 0, 0, 0, 0]
        upright      -> [0, pi, 0, 0, 0, 0, 0, 0]
    Output: y(x) = [cart, sin(h1), cos(h1), sin(h2), cos(h2), sin(h3), cos(h3),
                    cart_vel, hinge_1_vel, hinge_2_vel, hinge_3_vel]
    Using sin/cos instead of the raw angles makes the cost invariant to 2*pi wraps.
    """

    def __init__(self, dynamics_config: DynamicsConfig, config: ModelPredictiveControlConfig):
        super().__init__(dynamics_config, config)

        # cost weights on the lifted output (11-dim):
        # [cart, sin h1, cos h1, sin h2, cos h2, sin h3, cos h3, cart_v, h1_v, h2_v, h3_v]
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
        self.R = np.array([[0.1]])
        self.Qf = 20.0 * self.Q

        # goal state (upright) -> lifted goal output
        self.x_goal = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.y_goal = self.output(self.x_goal)

    @staticmethod
    def output(x):
        """Lift the state with sin/cos of all three hinge angles; shape (11,)."""
        p, h1, h2, h3, pdot, h1dot, h2dot, h3dot = x
        return np.array([p,
                         np.sin(h1), np.cos(h1),
                         np.sin(h2), np.cos(h2),
                         np.sin(h3), np.cos(h3),
                         pdot, h1dot, h2dot, h3dot])

    @staticmethod
    def output_jacobian(x):
        """dy/dx, shape (11, 8)."""
        _, h1, h2, h3, _, _, _, _ = x
        J = np.zeros((11, 8))
        J[0, 0] = 1.0            # d cart / d cart
        J[1, 1] = np.cos(h1)     # d sin(h1) / d h1
        J[2, 1] = -np.sin(h1)    # d cos(h1) / d h1
        J[3, 2] = np.cos(h2)     # d sin(h2) / d h2
        J[4, 2] = -np.sin(h2)    # d cos(h2) / d h2
        J[5, 3] = np.cos(h3)     # d sin(h3) / d h3
        J[6, 3] = -np.sin(h3)    # d cos(h3) / d h3
        J[7, 4] = 1.0            # d cart_vel / d cart_vel
        J[8, 5] = 1.0            # d hinge_1_vel / d hinge_1_vel
        J[9, 6] = 1.0            # d hinge_2_vel / d hinge_2_vel
        J[10, 7] = 1.0           # d hinge_3_vel / d hinge_3_vel
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
    xml_path = os.path.join(ROOT, "models", "triple_cartpole", "triple_cartpole.xml")

    # MPC parameters
    H = 150         # prediction horizon (shooting nodes)
    T = 5.0         # total closed-loop sim time [s]
    sim_dt = 0.02   # sim time step [s]
    node_dt = 0.02  # node time step [s]
    mpc_dt = 0.02   # MPC time step [s]

    # initial state (hanging down); goal is declared in TripleCartpoleMPC
    x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # dynamics config
    dyn_cfg = DynamicsConfig(model_path=xml_path,
                             sim_dt=sim_dt)

    # spline config
    spline_cfg = SplineConfig(M=75,
                              spline_type="zero")

    # mpc config
    cfg = ModelPredictiveControlConfig(
        N=H,
        node_dt=node_dt,
        mpc_dt=mpc_dt,
        u_lb=-20.0,
        u_ub= 20.0,
        spline=spline_cfg,
        ipopt_options={
            "print_level": 0,
            "max_iter": 5,
            "tol": 1e-4,
            "hessian_approximation": "limited-memory",
        },
        T=T,
        max_iter_initial=500,
        verbose=True,
    )

    # instantiate the MPC
    mpc = TripleCartpoleMPC(dyn_cfg, cfg)

    # feasible warm start: sample a sinewave input over all N_sim horizon sim
    # steps (set_warm_start subsamples the states at the nodes)
    t_h = sim_dt * np.arange(mpc.N_sim)
    U_ws = 20.0 * np.sin(2.0 * np.pi * 0.5 * t_h).reshape(mpc.N_sim, mpc.nu)
    X_ws = mpc.dyn.rollout(x_init, U_ws)
    mpc.set_warm_start(X_ws, U_ws)

    # run the closed loop
    t, X, U = mpc.run(x_init)

    # save the closed-loop trajectory for replay: time, state, input, model
    save_path = os.path.join(ROOT, "examples", "triple_cartpole_mpc", "triple_cartpole_mpc.npz")
    save_trajectory(save_path, time=t, state=X, input=U, model=xml_path,
                    spline_type=mpc.spline.spline_type)
    print("saved:", save_path)
