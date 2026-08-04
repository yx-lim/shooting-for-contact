##
#
# Per-motion configuration for the Go2 tracking-MPC template (go2_tracking_mpc.py).
#
##

# directory imports
import os

# standard imports
from dataclasses import dataclass
from typing import Optional

# local imports
from src.dynamics import DynamicsConfig
from src.spline import SplineConfig
from src.mpc import ModelPredictiveControlConfig


########################################################################
# STRUCTURAL EE BODY LIST
########################################################################

# Per row: (enable, body, w_pos, w_ori, w_vel, w_omega, relative_to_anchor); w_pos is one weight
# (x/y/z share it). rel=True -> anchor frame (drift-invariant); feet stay world (contacts).
DEFAULT_EE_ANCHOR = "robot/base_link"
DEFAULT_EE_TRACK = (
  # (enable, body,               w_pos, w_ori, w_vel, w_omega,  relative_to_anchor)
    (0,  "robot/base_link",      1.0,   1.0,   0.1,   0.1,   True),   # base == anchor (no-op)
    (1,  "robot/FL_calf",        1.0,   1.0,   0.1,   0.1,   False),  # front-left  foot (contact)
    (1,  "robot/FR_calf",        1.0,   1.0,   0.1,   0.1,   False),  # front-right foot (contact)
    (1,  "robot/RL_calf",        1.0,   1.0,   0.1,   0.1,   False),  # rear-left   foot (contact)
    (1,  "robot/RR_calf",        1.0,   1.0,   0.1,   0.1,   False),  # rear-right  foot (contact)
    (0,  "robot/FL_thigh",       1.0,   1.0,   0.1,   0.1,   True),
    (0,  "robot/FR_thigh",       1.0,   1.0,   0.1,   0.1,   True),
    (0,  "robot/RL_thigh",       1.0,   1.0,   0.1,   0.1,   True),
    (0,  "robot/RR_thigh",       1.0,   1.0,   0.1,   0.1,   True),
)

def _expand_ee_row(row):
    """Expand a 7-field config row (single w_pos) into the 8-field src/end_effector.py schema
    (w_pxy = w_pz). Rows already in 8-field form pass through unchanged."""
    if isinstance(row, (tuple, list)):
        row = tuple(row)
        if len(row) == 7:
            en, body, w_pos, w_ori, w_vel, w_om, rel = row
            return (en, body, w_pos, w_pos, w_ori, w_vel, w_om, rel)
    return row


########################################################################
# BASE CONFIG CLASS
########################################################################

@dataclass
class MPCTrackingConfig:
    """One tracked motion: all knobs for Go2TrackingMPC + driver.
    build_*_config() assemble the sub-configs (injecting ROOT, T, and the node qpos bounds)."""

    # identity / reference
    name: str                         # short key + output-file suffix
    traj_dir: tuple                   # path parts under trajectories/  e.g. ("go2", "wtw_pronking")

    # model / integration (rarely vary per motion)
    model_xml:   tuple = ("models", "unitree_go2", "go2.xml")   # robot + floor
    actuator_mode: str = "position"   # DEFAULT for this example: keep go2.xml's native <general>
                                      # PD servos (kp 20/40, kd 1/2 -- the real robot's gains), so
                                      # u is a joint-POSITION target and the reference joint angles
                                      # are themselves a valid warm start. "torque" converts them
                                      # to normalized direct-torque motors instead.
    warm_mode:   str   = "pd"         # TORQUE mode only: "pd" = PD-around-reference rollout seed;
                                      # ignored in position mode
    integrator:  str   = "implicitfast"
    sim_dt:      float = 0.01         # plant / integration step [s]
    fd_eps:      float = 1e-6         # finite-difference perturbation
    fd_centered: bool  = True         # symmetric (centered) finite differences
    n_threads:   int   = 8            # threads for the per-node FD Jacobian

    # cost-term selection + end-effector tracking
    track_state: bool = True          # state-space tangent tracking (0.5 e^T Qd e)
    track_body:  bool = True          # end-effector / body Cartesian tracking (ee_track below)
    exact_vel_grad: bool = False      # exact d/dq of the EE velocity terms (else cheap approximation)
    ee_anchor: str = DEFAULT_EE_ANCHOR
    ee_track:  tuple = DEFAULT_EE_TRACK

    # state-space tracking weights (assembled into Qd = [w_q(nv) | w_v(nv)])
    w_base_pos:    float = 10.0       # base x, y, z position
    w_base_ori:    float = 10.0       # base orientation
    w_base_linvel: float = 0.1        # base linear velocity
    w_base_angvel: float = 0.1        # base angular velocity
    w_joint_pos:   float = 0.1        # joint angles
    w_joint_vel:   float = 0.01       # joint velocities
    term_scale:    float = 100.0      # terminal-node weight multiplier (Qfd = term_scale * Qd)

    # actuator regularizers (0 -> off)
    Rtau:  float = 1e-6               # PD-servo torque penalty
    Rrate: float = 1e-2               # control-rate penalty on u_k - u_{k-1}

    # horizon / timing
    H: int = 30                       # shooting nodes in the prediction horizon
    node_dt: float = 0.02             # time per shooting interval [s] (K = node_dt/sim_dt plant steps)
    mpc_dt:  float = 0.02             # replan period [s] (must be a multiple of node_dt)
    T: Optional[float] = None         # closed-loop length [s]; None -> full reference duration

    # contact model (MuJoCo solref/solimp)
    friction_cone: str = "elliptic"           # "pyramidal" | "elliptic"
    friction: Optional[object] = None         # None -> keep go2.xml's per-geom friction (feet 0.6)
    condim:   Optional[int]    = None         # None -> keep go2.xml's per-geom condim (feet 3, body 1)
    solref: tuple = (0.02, 1.0)               # geom contact [timeconst, dampratio]; >= 2 x sim_dt
    solimp: tuple = (0, 0.95, 0.01, 0.5, 2)   # geom contact [dmin, dmax, width, midpoint, power]
    solreflimit: Optional[tuple] = None       # joint limit  [timeconst, dampratio]
    solimplimit: Optional[tuple] = None       # joint limit  [dmin, dmax, width, midpoint, power]

    # constraints
    joint_limits:    bool = True      # MuJoCo enforces joint ranges in the rollout / FD
    nlp_qpos_bounds: bool = False     # also box the shooting-node qpos in the NLP (IPOPT var bounds)

    # solver
    # NOTE: "mumps" ships with IPOPT and works out of the box, but it is much slower on a problem
    # this size. Installing HSL and switching to "ma57" is a significant speedup -- see the README.
    linear_solver:    str = "mumps"         # "mumps" | "ma27" | "ma57" | "ma97"  (ma* need HSL)
    hsllib:           str = "libcoinhsl.so" # HSL library to dlopen; only passed for an ma* solver
    M:                int = 15              # spline control knots
    spline:           str = "linear"        # spline basis: "zero" | "linear"
    max_iter_initial: int = 20              # IPOPT iters for the first (cold) solve
    max_iter:         int = 5               # IPOPT iters per MPC replan
    tol:            float = 1e-3            # IPOPT convergence tolerance

    # reference playback
    pz_offset:  float = 0.0           # constant lift of the reference base height [m]

    # assembly helpers (inject the runtime-only bits)

    def ee_track_expanded(self):
        """ee_track in src/end_effector.py schema (w_pos expanded to w_pxy = w_pz). Pass this to
        EETracker instead of the raw ee_track."""
        return tuple(_expand_ee_row(r) for r in self.ee_track)

    def traj_path(self, root):
        """Absolute path to this motion's reference-trajectory directory."""
        return os.path.join(root, "trajectories", *self.traj_dir)

    def build_dynamics_config(self, root):
        return DynamicsConfig(
            model_path=os.path.join(root, *self.model_xml),
            sim_dt=self.sim_dt,
            integrator=self.integrator,
            friction_cone=self.friction_cone,
            solref=list(self.solref),
            solimp=list(self.solimp),
            solreflimit=(list(self.solreflimit) if self.solreflimit is not None else None),
            solimplimit=(list(self.solimplimit) if self.solimplimit is not None else None),
            condim=self.condim,
            friction=self.friction,
            fd_eps=self.fd_eps,
            fd_centered=self.fd_centered,
            n_threads=self.n_threads,
            joint_limits=self.joint_limits,
            actuator_mode=self.actuator_mode,
        )

    def build_spline_config(self):
        return SplineConfig(M=self.M, spline_type=self.spline)

    def ipopt_linear_solver_options(self):
        """{"linear_solver": ...} plus "hsllib" only when an HSL (ma*) solver is selected, so a
        MUMPS run never carries a reference to a library it does not load."""
        opts = {"linear_solver": self.linear_solver}
        if self.linear_solver.startswith("ma"):
            opts["hsllib"] = self.hsllib
        return opts

    def build_mpc_config(self, T, x_lb=None, x_ub=None):
        return ModelPredictiveControlConfig(
            N=self.H,
            node_dt=self.node_dt,
            terminal_constraint=False,
            x_lb=x_lb, x_ub=x_ub,        # None unless nlp_qpos_bounds -> driver passes the bounds
            spline=self.build_spline_config(),
            ipopt_options={
                **self.ipopt_linear_solver_options(),
                "print_level": 0,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "hessian_approximation": "limited-memory",
            },
            mpc_dt=self.mpc_dt,
            T=T,
            max_iter_initial=self.max_iter_initial,
            verbose=True,
        )


########################################################################
# PER MOTION CONFIGS
########################################################################

PRONKING = MPCTrackingConfig(
    name="pronking",
    traj_dir=("go2", "wtw_pronking"),
    track_state=True,
    track_body=True,
    T=2.0,
    max_iter=7, 
    max_iter_initial=20,
)

HOPTURN = MPCTrackingConfig(
    name="hopturn",
    traj_dir=("go2", "stmr_hopturn"),
    track_state=True,
    track_body=True,
    pz_offset=0.018,
    T=3.0,                            
    H=20,                             
    max_iter=7,
    max_iter_initial=20,
)

CONFIGS = {
    "pronking": PRONKING,
    "hopturn": HOPTURN,
}
