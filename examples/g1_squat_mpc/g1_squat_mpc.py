##
#
# Unitree G1 (23 dof) squatting + reaching MPC.
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
import mujoco
from src.dynamics import DynamicsConfig
from src.mpc import ModelPredictiveControl, ModelPredictiveControlConfig
from src.spline import SplineConfig
from utils.file_utils import save_trajectory


########################################################################
# G1 SQUAT MPC
########################################################################

class G1SquatMPC(ModelPredictiveControl):
    """Squat the floating-base G1 (23 dof) under MPC: one down-up height cycle over T.
        STATE       x = [qpos (nq=30) | qvel (nv=29)]; qpos carries the free-joint quaternion
        COST        0.5 e^T Qd e  +  0.5 du^T Rd du,  e = state_diff(x, x_ref(t)) (2*nv = 58)
        REFERENCE   x_nom with base z on a cosine dip z_ref(t): z_stand -> z_stand-squat_depth
                    at t=T/2 -> z_stand; arms swing forward/palms-down on the SAME envelope
    du = u - u_nom; x_nom / u_nom come from the model's "init_state" keyframe. The pelvis is
    held upright and the leg joints weighted loosely, so they are free to bend into the squat."""

    def __init__(self, dynamics_config: DynamicsConfig, config: ModelPredictiveControlConfig):
        super().__init__(dynamics_config, config)

        # nominal standing state and control from the model keyframe
        key_qpos = self.dyn.model.key_qpos[0].copy()                   # (nq,)
        self.x_nom = np.concatenate([key_qpos, np.zeros(self.dyn.nv)]) # (nx,)
        self.u_nom = self.dyn.model.key_ctrl[0].copy()                 # (nu,)

        # squat profile: one full down-up cycle spans the whole sim time T = n_steps*dt
        self.z_stand = float(self.x_nom[2])           # nominal pelvis height [m]
        self.squat_depth = 0.4                        # how far the pelvis drops [m]
        self.squat_period = self.n_steps * self.dt    # one cosine cycle over the full sim

        # arm "reach forward, palm down" targets on the SAME cosine envelope; qpos addresses are
        # fetched by name so this survives model edits. "Down" is approximate (no wrist pitch).
        m = self.dyn.model
        def _qadr(name):
            return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])
        def _dadr(name):
            return int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])
        self.arm_targets = {                          # qpos address -> target angle [rad]
            _qadr("robot/left_shoulder_pitch_joint"):  -1.4,   # swing the arms forward (nom +0.2)
            _qadr("robot/right_shoulder_pitch_joint"): -1.4,
            _qadr("robot/left_wrist_roll_joint"):      -0.4,   # roll the palms down; flip if up
            _qadr("robot/right_wrist_roll_joint"):     +0.4,
        }
        # dof (tangent) indices of those joints, used to firm up their tracking weight
        self._arm_dofs = [_dadr(n) for n in (
            "robot/left_shoulder_pitch_joint", "robot/right_shoulder_pitch_joint",
            "robot/left_wrist_roll_joint", "robot/right_wrist_roll_joint")]

        # tangent-error weights e = [dq(nv) | dv(nv)], nv = 6 base + 23 joints: track base z and
        # pelvis tilt hard, keep the leg joints loose so they can bend to reach the height.
        w_q = np.concatenate([[10.0, 10.0, 200.0],   # base position (z tracks the squat)
                              [150.0, 150.0, 150.0],  # base orientation (upright pelvis)
                              0.1 * np.ones(23)])     # joints: loose -> free to bend
        w_v = np.concatenate([[1.0, 1.0, 1.0],        # base linear velocity
                              [1.0, 1.0, 1.0],         # base angular velocity
                              0.05 * np.ones(23)])     # joint velocities
        # firm up tracking on the four reaching arm joints
        for dof in self._arm_dofs:
            w_q[dof] = 10.0
        self.Qd = np.concatenate([w_q, w_v])          # (2*nv,) diagonal state weight
        self.Qfd = 100.0 * self.Qd                     # terminal weight
        self.Rd = 2e-2 * np.ones(self.nu)             # control-deviation weight

    def _cycle(self, t):
        """Cosine envelope over the sim: 0 at t=0, 1 at t=T/2, back to 0 at t=T."""
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * t / self.squat_period))

    def z_ref(self, t):
        """Squat height: z_stand at t=0, dips by squat_depth at t=T/2, back at t=T."""
        return self.z_stand - self.squat_depth * self._cycle(t)

    def x_ref(self, t):
        """Nominal pose with the base height set to the squat profile and the arms swung
        forward/palms-down, both driven by the same _cycle(t) envelope."""
        xr = self.x_nom.copy()
        s = self._cycle(t)
        xr[2] = self.z_ref(t)
        # arms: interpolate each joint from its nominal angle to the reach target
        for adr, target in self.arm_targets.items():
            xr[adr] = (1.0 - s) * self.x_nom[adr] + s * target
        return xr

    def _time(self, k):
        """Absolute sim time of horizon stage k (t_now is set each MPC step)."""
        return self.t_now + k * self.dt

    def stage_cost(self, x, u, k):
        e = self.dyn.state_diff(x, self.x_ref(self._time(k)))
        du = u - self.u_nom
        return float(0.5 * (e @ (self.Qd * e) + du @ (self.Rd * du)))

    def stage_cost_grad(self, x, u, k):
        xr = self.x_ref(self._time(k))
        e = self.dyn.state_diff(x, xr)
        # matrix-free (J1 @ Q_inv)^T (Qd*e) -- no (2nv x nx) matrices built
        lx = self.dyn.state_diff_tangent_grad(x, xr, self.Qd * e)   # (nx,)
        lu = self.Rd * (u - self.u_nom)                   # (nu,)
        return lx, lu

    def terminal_cost(self, x):
        e = self.dyn.state_diff(x, self.x_ref(self._time(self.N_sim)))
        return float(0.5 * e @ (self.Qfd * e))

    def terminal_cost_grad(self, x):
        xr = self.x_ref(self._time(self.N_sim))
        e = self.dyn.state_diff(x, xr)
        return self.dyn.state_diff_tangent_grad(x, xr, self.Qfd * e)


########################################################################
#  MAIN
########################################################################

if __name__ == "__main__":

    # model
    xml_path = os.path.join(ROOT, "models", "unitree_g1", "g1_23dof_feet.xml")

    # MPC config
    H = 25           # prediction horizon (shooting nodes)
    T = 4.0          # total closed-loop sim time [s]
    sim_dt = 0.02    # sim time step [s]
    node_dt = 0.02   # node time step [s]
    mpc_dt = 0.02    # MPC time step [s]

    # dynamics config
    dyn_cfg = DynamicsConfig(model_path=xml_path,
                             sim_dt=sim_dt,
                             friction_cone="elliptic",
                             fd_eps=1e-6,
                             solref=[0.04, 1.0],
                             solimp=[0, 1.0, 0.01, 0.5, 2],
                             fd_centered=False,
                             actuator_mode="position",
                             n_threads=8)

    # spline config
    spline_cfg = SplineConfig(M=25,
                              spline_type="linear")

    # peek model for the state-bound joint lookups below; control bounds default to the
    # model's ctrlrange (dyn.u_lb / dyn.u_ub).
    _m = mujoco.MjModel.from_xml_path(xml_path)

    # state box bounds
    nx = _m.nq + _m.nv
    x_lb = np.full(nx, -np.inf)
    x_ub = np.full(nx,  np.inf)
    hip_roll_limit = 0.001
    for adr in (_m.jnt_qposadr[mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT, nm)]
                for nm in ("robot/left_hip_roll_joint", "robot/right_hip_roll_joint")):
        x_lb[adr] = -hip_roll_limit
        x_ub[adr] =  hip_roll_limit

    # mpc config
    mpc_cfg = ModelPredictiveControlConfig(
        N=H,
        node_dt=node_dt,
        mpc_dt=mpc_dt,
        spline=spline_cfg,
        terminal_constraint=False,
        x_lb=x_lb,
        x_ub=x_ub,
        ipopt_options={
            "print_level": 0,
            "max_iter": 3,
            "tol": 1e-3,
            "hessian_approximation": "limited-memory",
        },
        T=T,
        verbose=True,
    )

    # instantiate the MPC; start from the nominal standing pose
    mpc = G1SquatMPC(dyn_cfg, mpc_cfg)
    q_init = np.array([
        0,       0,     0.75,
        1,       0,     0,     0,
        -0.312,  0,     0,     0.669, -0.363,  0,
        -0.312,  0,     0,     0.669, -0.363,  0,
        0,
        0.2,    0.2,    0,     0.6,    0,
        0.2,   -0.2,    0,     0.6,    0])
    v_init = np.zeros(mpc.dyn.nv)
    x_init = np.concatenate([q_init, v_init])

    # warm start
    U_ws = np.tile(mpc.u_nom, (mpc.N_sim, 1))
    X_ws = mpc.dyn.rollout(x_init, U_ws)
    mpc.set_warm_start(X_ws, U_ws)

    # run the closed loop
    t, X, U = mpc.run(x_init)

    # save the closed-loop trajectory for replay: time, state, input, model
    save_path = os.path.join(ROOT, "examples", "g1_squat_mpc", "squat.npz")  # replay.py squat
    save_trajectory(save_path, time=t, state=X, input=U, model=xml_path,
                    spline_type=mpc.spline.spline_type)
    print("saved:", save_path)
