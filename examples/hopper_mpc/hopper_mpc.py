##
#
# Hopper velocity tracking MPC.
#
##

# directory imports
import sys
import os

ROOT = os.getenv("TRAJOPT_ROOT_DIR")
sys.path.append(ROOT)

# standard imports
import numpy as np

# custom imports
from src.dynamics import DynamicsConfig
from src.mpc import ModelPredictiveControl, ModelPredictiveControlConfig
from src.spline import SplineConfig
from utils.file_utils import save_trajectory


########################################################################
# HOPPER MPC
########################################################################

class HopperMPC(ModelPredictiveControl):
    """ Quadratic cost on a lifted output that uses sin/cos of the torso pitch.

    State:  x = [px, pz, theta, leg, vx, vz, omega, leg_vel]
        px       : horizontal torso position
        pz       : vertical torso position
        theta    : torso pitch (upright = 0)
        leg      : spring-loaded leg extension (nominal 0)
        + their velocities
    Output: y(x) = [px, pz, sin(theta), cos(theta), leg, vx, vz, omega, leg_vel]
    Using sin/cos instead of the raw pitch makes the cost invariant to 2*pi wraps.
    """

    def __init__(self, dynamics_config: DynamicsConfig, config: ModelPredictiveControlConfig):
        super().__init__(dynamics_config, config)

        # weights on the lifted output (9-dim):
        # [px, pz, sin th, cos th, leg, vx, vz, omega, leg_vel]
        self.Q = np.diag([5.0, 10.0, 10.0, 10.0, 1.0, 0.1, 0.1, 1.0, 0.1])
        self.R = np.diag([1e-2, 1e-3])
        self.Qf = 10.0 * self.Q

        # velocity-tracking reference
        self.vx_goal = 2.0                 # desired forward speed [m/s]
        self.pz_ref = 2.0                  # target hop height [m]

    def y_ref(self, tau: float) -> np.ndarray:
        """Lifted reference output at horizon-relative time tau (= k*dt); shape (9,).

        y = [px, pz, sin th, cos th, leg, vx, vz, omega, leg_v].
        px ramps forward from the current position at vx_goal:
            px_ref(tau) = px_current + vx_goal * tau,   px_current = x_init[0]
        with vx held at vx_goal, pz at the ride height, everything else upright/zero.
        """
        px_ref = self.x_init[0] + self.vx_goal * tau       # relative to current px
        return np.array([px_ref, self.pz_ref,
                         0.0, 1.0,           # sin/cos of theta = 0 (upright)
                         0.0,                # leg
                         self.vx_goal, 0.0,  # vx, vz
                         0.0, 0.0])          # omega, leg_vel

    @staticmethod
    def output(x):
        """Lift the state with sin/cos of the torso pitch; shape (9,)."""
        px, pz, th, leg, vx, vz, omega, leg_v = x
        return np.array([px, pz,
                         np.sin(th), np.cos(th),
                         leg, vx, vz, omega, leg_v])

    @staticmethod
    def output_jacobian(x):
        """dy/dx, shape (9, 8)."""
        _, _, th, _, _, _, _, _ = x
        J = np.zeros((9, 8))
        J[0, 0] = 1.0           # d px / d px
        J[1, 1] = 1.0           # d pz / d pz
        J[2, 2] = np.cos(th)    # d sin(theta) / d theta
        J[3, 2] = -np.sin(th)   # d cos(theta) / d theta
        J[4, 3] = 1.0           # d leg / d leg
        J[5, 4] = 1.0           # d vx / d vx
        J[6, 5] = 1.0           # d vz / d vz
        J[7, 6] = 1.0           # d omega / d omega
        J[8, 7] = 1.0           # d leg_vel / d leg_vel
        return J

    def stage_cost(self, x, u, k):
        dy = self.output(x) - self.y_ref(k * self.dt)
        return float(0.5 * (dy @ self.Q @ dy + u @ self.R @ u))

    def stage_cost_grad(self, x, u, k):
        dy = self.output(x) - self.y_ref(k * self.dt)
        J = self.output_jacobian(x)
        return J.T @ self.Q @ dy, self.R @ u

    def terminal_cost(self, x):
        dy = self.output(x) - self.y_ref(self.N_sim * self.dt)
        return float(0.5 * dy @ self.Qf @ dy)

    def terminal_cost_grad(self, x):
        dy = self.output(x) - self.y_ref(self.N_sim * self.dt)
        J = self.output_jacobian(x)
        return J.T @ self.Qf @ dy


########################################################################
#  MAIN
########################################################################

if __name__ == "__main__":

    # model
    xml_path = os.path.join(ROOT, "models", "hopper", "hopper.xml")

    # MPC config
    H = 40           # prediction horizon (shooting nodes)
    T = 4.0          # total closed-loop sim time [s]
    sim_dt = 0.02    # sim time step [s]
    node_dt = 0.02   # node time step [s]
    mpc_dt = 0.02    # MPC time step [s]

    # initial state
    x_init = np.array([0.0, 2.0, 0.0, 0.0,  # px, pz, theta, leg
                       0.0, 0.0, 0.0, 0.0]) # vx, vz, omega, leg_vel

    # dynamics config
    dyn_cfg = DynamicsConfig(model_path=xml_path,
                             sim_dt=sim_dt,
                             friction_cone="pyramidal")

    # spline config
    spline_cfg = SplineConfig(M=20,
                              spline_type="zero")

    # state box bounds (x = [px, pz, theta, leg, vx, vz, omega, leg_vel])
    theta_angle = np.pi * 0.5
    pz_floor = 0.5
    x_lb = np.array([-np.inf, pz_floor, -theta_angle, -np.inf,
                     -np.inf, -np.inf,  -np.inf, -np.inf])
    x_ub = np.array([ np.inf,  np.inf,   theta_angle,  np.inf,
                      np.inf,  np.inf,   np.inf,  np.inf])

    # mpc config
    mpc_cfg = ModelPredictiveControlConfig(
        N=H,
        node_dt=node_dt,
        mpc_dt=mpc_dt,
        u_lb=np.array([-20.0, -300.0]),   # [body_motor, leg_motor]
        u_ub=np.array([ 20.0,  300.0]),
        x_lb=x_lb,
        x_ub=x_ub,
        spline=spline_cfg,
        ipopt_options={
            "print_level": 0,
            "max_iter": 10,
            "tol": 1e-4,
            "hessian_approximation": "limited-memory",
        },
        T=T,
        verbose=True,
    )

    # instantiate the MPC
    mpc = HopperMPC(dyn_cfg, mpc_cfg)

    # feasible warm start: roll a zero-input (passive) trajectory out from x_init
    # over all N_sim horizon sim steps (states are subsampled at the nodes)
    U_ws = np.zeros((mpc.N_sim, mpc.nu))
    X_ws = mpc.dyn.rollout(x_init, U_ws)
    mpc.set_warm_start(X_ws, U_ws)

    # run the closed loop
    t, X, U = mpc.run(x_init)

    # save the closed-loop trajectory for replay: time, state, input, model
    save_path = os.path.join(ROOT, "examples", "hopper_mpc", "hopper_mpc.npz")
    save_trajectory(save_path, time=t, state=X, input=U, model=xml_path,
                    spline_type=mpc.spline.spline_type)
    print("saved:", save_path)
