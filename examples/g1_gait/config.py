##
#
# Per-motion configuration for the G1 walking trajectory-optimizer (g1_gait.py).
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
from src.multi_shooting import MultiShootingConfig


########################################################################
# STRUCTURAL EE BODY LIST
########################################################################

# Per row: (enable, body, w_pxy, w_pz, w_ori, w_v, w_omega, rel). rel=True -> torso-anchor frame, else world.
# Walking loads only the feet: they track world z + orientation (w_pxy=0 -> footholds free),
# while the free-swinging hands and the pelvis track relative to the torso.
DEFAULT_EE_ANCHOR = "robot/torso_link"
DEFAULT_EE_TRACK = (
  # (enable, body,                        w_pxy, w_pz, w_ori, w_v,  w_omega, relative_to_anchor)
    (1, "robot/left_ankle_roll_link",     1.0,   1.0,  1.0,   0.01, 0.01,    False),  # left foot  (world z)
    (1, "robot/right_ankle_roll_link",    1.0,   1.0,  1.0,   0.01, 0.01,    False),  # right foot (world z)
    (1, "robot/left_wrist_yaw_link",      1.0,   1.0,  1.0,   0.01, 0.01,    True),   # left hand  vs torso
    (1, "robot/right_wrist_yaw_link",     1.0,   1.0,  1.0,   0.01, 0.01,    True),   # right hand vs torso
    (1, "robot/pelvis",                   1.0,   1.0,  1.0,   0.01, 0.01,    False),   # pelvis vs torso
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
class GaitTOConfig:
    """One walking motion: all knobs for G1GaitTO + driver. build_*_config() assemble the
    sub-configs (injecting ROOT and the reference-derived horizon). No twist / periodicity knobs --
    this optimizer only tracks the clip, starting from the reference's first frame."""

    # identity / reference
    name: str                         # short key + output-file suffix
    traj_dir: tuple                   # path parts under trajectories/ e.g. ("g1", "lafan", "walk...")
    model_xml: tuple = ("models", "unitree_g1", "g1_29dof_feet.xml")
    actuator_mode: str = "position"   # "position" | "torque" |
    pz_offset: float = 0.0            # add to the reference base z on load

    # model / integration
    integrator:  str   = "implicitfast"
    sim_dt:      float = 0.005        # plant / integration step [s]
    node_dt:     float = 0.02         # shooting-interval time [s] (K = node_dt/sim_dt steps)
    fd_eps:      float = 1e-6         # finite-difference perturbation
    fd_centered: bool  = True         # symmetric (centered) finite differences
    n_threads:   int   = 8            # threads for the per-node FD Jacobian

    # end-effector tracking (state tracking is always on; EE follows the ee_track weights)
    exact_vel_grad: bool = False      # exact d/dq of EE velocity terms (else cheap approx)
    ee_anchor: str = DEFAULT_EE_ANCHOR
    ee_track:  tuple = DEFAULT_EE_TRACK

    # state-space tracking weights (assembled into Qd = [w_q(nv) | w_v(nv)])
    w_base_pos:    float = 10.0       # base x, y, z position
    w_base_ori:    float = 10.0       # base orientation (full, yaw included -- there is no twist)
    w_base_linvel: float = 0.1        # base linear velocity
    w_base_angvel: float = 0.1        # base angular velocity
    w_joint_pos:   float = 0.1        # joint angles
    w_joint_vel:   float = 0.01       # joint velocities
    term_scale:    float = 10.0       # terminal-node weight multiplier (Qfd = term_scale * Qd)

    # actuator regularizers (0 -> off)
    Rtau:  float = 1e-6               # torque penalty 0.5 tau^T Rtau tau (physical N.m: gear*u)
    Rrate: float = 1e-2               # rate penalty on u_k-u_{k-1} (normalized command)

    # control parametrization
    ctrl_hz: float = 50.0             # control knots over the horizon -> M = round(N_sim*sim_dt*ctrl_hz)+1
    spline:  str   = "linear"         # spline basis: "zero" | "linear"

    # contact model (MuJoCo solref/solimp)
    solref: tuple = (0.04, 1.0)               # geom contact [timeconst, dampratio]
    solimp: tuple = (0, 0.95, 0.01, 0.5, 2)   # geom contact [dmin, dmax, width, midpoint, power]
    solreflimit: Optional[tuple] = None       # joint limit  [timeconst, dampratio]
    solimplimit: Optional[tuple] = None       # joint limit  [dmin, dmax, width, midpoint, power]
    condim:   Optional[int]    = None         # contact dim on all geoms (None=keep model)
    friction: Optional[object] = None         # geom friction on all geoms: scalar or [slide, spin, roll]
    friction_cone: str = "pyramidal"          # "pyramidal" | "elliptic"

    # constraints
    joint_limits: bool = True         # MuJoCo enforces joint ranges in the rollout / FD

    # solver
    # NOTE: "mumps" ships with IPOPT and works out of the box, but it is much slower on a problem
    # this size. Installing HSL and switching to "ma57" is a significant speedup -- see the README.
    linear_solver:    str = "mumps"         # "mumps" | "ma27" | "ma57" | "ma97"  (ma* need HSL)
    hsllib:           str = "libcoinhsl.so" # HSL library to dlopen; only passed for an ma* solver
    print_level:      int = 5
    max_iter:         int = 250             # IPOPT iters (full-horizon TO needs many)
    tol:            float = 1e-2
    acceptable_tol: float = 1e-2
    acceptable_constr_viol_tol: float = 1e-2
    acceptable_iter:  int = 5
    keep_best_sol_rho: float = 1e2          # keep-best merit weight obj + rho*inf_pr

    # assembly helpers (inject the runtime-only bits)

    def ee_track_expanded(self):
        """ee_track in the src/end_effector.py schema (single w_pos expanded to w_pxy = w_pz).
        Pass this to EETracker instead of the raw ee_track."""
        return tuple(_expand_ee_row(r) for r in self.ee_track)

    def traj_path(self, root):
        """Absolute path to this motion's reference-trajectory directory."""
        return os.path.join(root, "trajectories", *self.traj_dir)

    def ipopt_linear_solver_options(self):
        """{"linear_solver": ...} plus "hsllib" only when an HSL (ma*) solver is selected, so a
        MUMPS run never carries a reference to a library it does not load."""
        opts = {"linear_solver": self.linear_solver}
        if self.linear_solver.startswith("ma"):
            opts["hsllib"] = self.hsllib
        return opts

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

    def build_spline_config(self, n_sim):
        """ZOH/linear controls at ctrl_hz over the horizon: M = round(N_sim*sim_dt*ctrl_hz)+1 knots."""
        M = max(2, int(round(n_sim * self.sim_dt * self.ctrl_hz)) + 1)
        return SplineConfig(M=M, spline_type=self.spline)

    def build_ms_config(self, N, spline_cfg):
        """The full-horizon multiple-shooting NLP config. N is derived from the clip length."""
        return MultiShootingConfig(
            N=N,
            node_dt=self.node_dt,
            spline=spline_cfg,
            ipopt_options={
                **self.ipopt_linear_solver_options(),
                "print_level": self.print_level,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "acceptable_tol": self.acceptable_tol,
                "acceptable_constr_viol_tol": self.acceptable_constr_viol_tol,
                "acceptable_iter": self.acceptable_iter,
                "hessian_approximation": "limited-memory",
                "mu_strategy": "adaptive",
            },
            keep_best_sol_rho=self.keep_best_sol_rho,
        )


########################################################################
# PER MOTION CONFIGS
########################################################################

WALK_FWD = GaitTOConfig(
    name="walk_fwd",
    traj_dir=("g1", "lafan", "walk1_subject1_crop148-376_z-0.025_periodic"),
    sim_dt=0.01,             
    node_dt=0.02,
    ctrl_hz=50.0,   
    spline="linear",
)

RUN_FWD = dataclasses.replace(
    WALK_FWD, 
    name="run_fwd", 
    traj_dir=("g1", "bones_seed", "neutral_jog_ff_180_R_001__A534_M_crop75-500_z+0.025_periodic"),
    sim_dt=0.005,
)

RUN_BCK = dataclasses.replace(
    RUN_FWD,
    name="run_bck",
    traj_dir=("g1", "bones_seed", "neutral_jog_ff_360_R_001__A534_M_crop100-715_z+0.025_periodic")
)

CRAWL_FWD = GaitTOConfig(
    name="crawl_fwd",
    model_xml=("models", "unitree_g1", "g1_29dof.xml"),   # 69 geoms: knees/forearms collide too
    traj_dir=("g1", "bones_seed", "crawl_ff_loop_180_R_001__A229_periodic"),
    actuator_mode="torque",
    sim_dt=0.005,
    node_dt=0.02,
    ctrl_hz=50.0,
    spline="zero",
    condim=4,           # crawl contacts need tangential + torsional friction
    friction=[1.0, 0.005, 0.0001],
    friction_cone="elliptic",
    ee_track=(
      # (enable, body,                        w_pxy, w_pz, w_ori, w_v,  w_omega, relative_to_anchor)
        (1, "robot/left_ankle_roll_link",     1.0,   1.0,  1.0,   0.01, 0.01,    False),  # left knee/foot
        (1, "robot/right_ankle_roll_link",    1.0,   1.0,  1.0,   0.01, 0.01,    False),  # right knee/foot
        (1, "robot/left_wrist_yaw_link",      1.0,   1.0,  1.0,   0.01, 0.01,    False),  # left hand  (contact)
        (1, "robot/right_wrist_yaw_link",     1.0,   1.0,  1.0,   0.01, 0.01,    False),  # right hand (contact)
        (1, "robot/pelvis",                   1.0,   1.0,  1.0,   0.01, 0.01,    False),
    ),
)

CONFIGS = {
    "walk_fwd": WALK_FWD,
    "run_fwd": RUN_FWD,
    "run_bck": RUN_BCK,
    "crawl_fwd": CRAWL_FWD,
}
