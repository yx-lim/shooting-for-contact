##
#
# Per-motion configuration for the G1 tracking-MPC template (g1_tracking_mpc.py).
#
##

# directory imports
import os

# standard imports
import dataclasses
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
# (x/y/z share it). rel=True -> anchor frame (drift-invariant); feet/hands stay world (contacts).
DEFAULT_EE_ANCHOR = "robot/torso_link"
DEFAULT_EE_TRACK = (
  # (enable, body,                         w_pos, w_ori, w_vel, w_omega,  relative_to_anchor)
    (0,  "robot/torso_link",               1.0,   1.0,   0.1,   0.1,   True),   # torso
    (1,  "robot/pelvis",                   1.0,   1.0,   0.1,   0.1,   True),   # pelvis
    (0,  "robot/left_hip_roll_link",       1.0,   1.0,   0.1,   0.1,   True),
    (0,  "robot/left_knee_link",           1.0,   1.0,   0.1,   0.1,   True),
    (1,  "robot/left_ankle_roll_link",     1.0,   1.0,   0.1,   0.1,   False),  # left foot (contact)
    (0,  "robot/right_hip_roll_link",      1.0,   1.0,   0.1,   0.1,   True),
    (0,  "robot/right_knee_link",          1.0,   1.0,   0.1,   0.1,   True),
    (1,  "robot/right_ankle_roll_link",    1.0,   1.0,   0.1,   0.1,   False),  # right foot (contact)
    (0,  "robot/left_shoulder_roll_link",  1.0,   1.0,   0.1,   0.1,   True),
    (0,  "robot/left_elbow_link",          1.0,   1.0,   0.1,   0.1,   True),
    (1,  "robot/left_wrist_yaw_link",      1.0,   1.0,   0.1,   0.1,   False),  # left hand (contact)
    (0,  "robot/right_shoulder_roll_link", 1.0,   1.0,   0.1,   0.1,   True),
    (0,  "robot/right_elbow_link",         1.0,   1.0,   0.1,   0.1,   True),
    (1,  "robot/right_wrist_yaw_link",     1.0,   1.0,   0.1,   0.1,   False),  # right hand (contact)
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
    """One tracked motion: all knobs for G1TrackingMPC + driver.
    build_*_config() assemble the sub-configs (injecting ROOT, T, and the node qpos bounds)."""

    # identity / reference
    name: str                         # short key + output-file suffix
    traj_dir: tuple                   # path parts under trajectories/ e.g. ("g1", "bones_seed", "crawl_...")

    # model / integration (rarely vary per motion)
    model_xml:   tuple = ("models", "unitree_g1", "g1_29dof.xml")
    actuator_mode: str = "position"   # DEFAULT: keep the XML's PD servos, so u is a joint-POSITION
                                      # target. "torque" converts them to normalized <motor> actuators.
    warm_mode:   str   = "pd"         # TORQUE mode only: "pd" = PD-around-reference rollout seed;
                                      # ignored in position mode
    integrator:  str   = "implicitfast"
    sim_dt:      float = 0.01         # plant / integration step [s]
    fd_eps:      float = 1e-6         # finite-difference perturbation
    fd_centered: bool  = True         # symmetric (centered) finite differences
    n_threads:   int   = 8            # threads for the per-node FD Jacobian

    # cost-term selection + end-effector tracking
    track_state: bool = True          # state-space tangent tracking (0.5 e^T Qd e)
    track_body:  bool = False         # end-effector / body Cartesian tracking (ee_track below)
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
    Rrate: float = 1e-3               # control-rate penalty on u_k - u_{k-1}

    # horizon / timing
    H: int = 30                       # shooting nodes in the prediction horizon
    node_dt: float = 0.02             # time per shooting interval [s] (K = node_dt/sim_dt plant steps)
    mpc_dt:  float = 0.02             # replan period [s] (must be a multiple of node_dt)
    T: Optional[float] = None         # closed-loop length [s]; None -> full reference duration

    # contact model (MuJoCo solref/solimp)
    friction_cone: str = "elliptic"           # "pyramidal" | "elliptic"
    friction: Optional[object] = None         # geom friction on all geoms: scalar (slide) or [slide, spin, roll]
    condim:   Optional[int]    = None         # contact dim on all geoms (None=keep model; 3 -> +tangential friction)
    solref: tuple = (0.04, 1.0)               # geom contact [timeconst, dampratio]
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
    M:                int = 30              # spline control knots
    spline:           str = "zero"          # spline basis: "zero" | "linear"
    max_iter_initial: int = 200             # IPOPT iters for the first (cold) solve
    max_iter:         int = 5               # IPOPT iters per MPC replan
    tol:            float = 1e-3            # IPOPT convergence tolerance

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

JUMP = MPCTrackingConfig(
    name="jump",
    model_xml=("models", "unitree_g1", "g1_29dof.xml"),
    # traj_dir=("g1", "rom", "srb_ik_jump_fwd"),
    traj_dir=("g1", "rom", "kino_180_twist_jump"),
    track_state=True,
    track_body=True,
    spline="linear",
    Rrate=1e-2,
    max_iter_initial=20,
    max_iter=10,
)

CRAWL_FWD = MPCTrackingConfig(
    name="crawl_fwd",
    traj_dir=("g1", "bones_seed", "crawl_ff_loop_180_R_001__A229_periodic"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

CRAWL_BCK = dataclasses.replace(
    CRAWL_FWD,
    name="crawl_bck",
    traj_dir=("g1", "bones_seed", "crawl_ff_loop_360_R_001__A232_periodic"),
)

CRAWL_TURN = dataclasses.replace(
    CRAWL_FWD,
    name="crawl_turn",
    traj_dir=("g1", "bones_seed", "turn_crawl_360_004__A133_M_crop0-807_periodic"),
    max_iter=7,
)

STAND = MPCTrackingConfig(
    name="stand",
    traj_dir=("g1", "animation", "stand_50f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

STAND1 = MPCTrackingConfig(
    name="stand1",
    traj_dir=("g1", "animation", "stand1_10f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

SLIDE = MPCTrackingConfig(
    name="slide",
    traj_dir=("g1", "animation", "slide_3f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

SLIDEFLOAT = MPCTrackingConfig(
    name="slidefloat",
    traj_dir=("g1", "animation", "slidefloat_20f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

SLIDEKNEE = MPCTrackingConfig(
    name="slideknee",
    traj_dir=("g1", "animation", "slideknee_3f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

KNEEL = MPCTrackingConfig(
    name="kneel",
    traj_dir=("g1", "animation", "kneel_30f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

KNEEL1 = MPCTrackingConfig(
    name="kneel1",
    traj_dir=("g1", "animation", "kneel1_20f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

KNEEL2 = MPCTrackingConfig(
    name="kneel2",
    traj_dir=("g1", "animation", "kneel2_20f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

HOP = MPCTrackingConfig(
    name="hop",
    traj_dir=("g1", "animation", "hop_20f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

FRONT_FLIP = MPCTrackingConfig(
    name="frontflip",
    traj_dir=("g1", "animation", "frontflip_40f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

FRONT_FLIP1 = MPCTrackingConfig(
    name="frontflip1",
    traj_dir=("g1", "animation", "frontflip1"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

FRONT_FLIP2 = MPCTrackingConfig(
    name="frontflip2",
    traj_dir=("g1", "animation", "frontflip2"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)


MOONWALK = MPCTrackingConfig(
    name="moonwalk",
    traj_dir=("g1", "animation", "moonwalk_40f"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

LUNGES = MPCTrackingConfig(
    name="lunges",
    traj_dir=("g1", "animation", "lunges"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

LUNGES2 = MPCTrackingConfig(
    name="lunges2",
    traj_dir=("g1", "animation", "lunges2"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

CRAWLSTAND = MPCTrackingConfig(
    name="crawlstand",
    traj_dir=("g1", "animation", "crawlstand"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

CRAWLSTAND1 = MPCTrackingConfig(
    name="crawlstand1",
    traj_dir=("g1", "animation", "crawlstand1"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

WORM1 = MPCTrackingConfig(
    name="worm1",
    traj_dir=("g1", "animation", "worm1"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

WORM2 = MPCTrackingConfig(
    name="worm2",
    traj_dir=("g1", "animation", "worm2"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

THEWORM = MPCTrackingConfig(
    name="theworm",
    traj_dir=("g1", "animation", "theworm"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)

THEWORM2 = MPCTrackingConfig(
    name="theworm2",
    traj_dir=("g1", "animation", "theworm2"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)
PUSHUP = MPCTrackingConfig(
    name="pushup",
    traj_dir=("g1", "animation", "pushup"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)
PUSHUP2= MPCTrackingConfig(
    name="pushup2",
    traj_dir=("g1", "animation", "pushup2"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)
SQUAT = MPCTrackingConfig(
    name="squat",
    traj_dir=("g1", "animation", "squat"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)
TWERK = MPCTrackingConfig(
    name="twerk",
    traj_dir=("g1", "animation", "twerk"),
    track_state=True,
    track_body=True,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    Rrate=1e-2,
    H=30,
    M=15,
    sim_dt=0.01,
    node_dt=0.02,
    mpc_dt=0.02,
    max_iter_initial=20,
    max_iter=20,
)


CONFIGS = {
    "jump": JUMP,
    "crawl_fwd": CRAWL_FWD,
    "crawl_bck": CRAWL_BCK,
    "crawl_turn": CRAWL_TURN,
    "stand": STAND,
    "stand1": STAND1,
    "slide": SLIDE,
    "slidefloat": SLIDEFLOAT,
    "slideknee": SLIDEKNEE,
    "kneel": KNEEL,
    "kneel1": KNEEL1,
    "kneel2": KNEEL2,
    "hop": HOP,
    "frontflip": FRONT_FLIP,
    "frontflip1": FRONT_FLIP1,
    "frontflip2": FRONT_FLIP2,
    "moonwalk": MOONWALK,
    "lunges": LUNGES,
    "lunges2": LUNGES2,
    "crawlstand": CRAWLSTAND,
    "crawlstand1": CRAWLSTAND1,
    "worm1": WORM1,
    "worm2": WORM2,
    "theworm": THEWORM,
    "theworm2": THEWORM2,
    "pushup": PUSHUP,
    "pushup2": PUSHUP2,
    "squat": SQUAT,
    "twerk": TWERK,
}
