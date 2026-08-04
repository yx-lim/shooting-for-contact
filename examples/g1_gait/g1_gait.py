##
#
# Unitree G1 (29 dof) walking trajectory optimization (single full-horizon solve).
#
##

# directory imports
import sys
import os

ROOT = os.getenv("TRAJOPT_ROOT_DIR")
sys.path.append(ROOT)

# standard imports
import argparse
import numpy as np
import mujoco

# local imports
from src.multi_shooting import MultiShootingBase
from utils.file_utils import load_reference, save_trajectory
from utils.math_utils import resample_state_trajectory
from src.end_effector import EETracker
from config import CONFIGS


########################################################################
# REFERENCE LOADING
########################################################################

def plant_grid_frames(n_ref, dt_ref, sim_dt):
    """Frame count after resampling onto the sim grid (matches resample_state_trajectory)."""
    return int(round((n_ref - 1) * dt_ref / sim_dt)) + 1


########################################################################
# G1 WALKING TO  (state + end-effector tracking)
########################################################################

class G1GaitTO(MultiShootingBase):
    """Full-horizon walking TO: track a reference clip with a quadratic tangent cost.
        COST         0.5 e^T Qd e  +  0.5 tau^T Rtau tau  +  EE terms  +  a control-rate penalty
                     e = state_diff(x, x_ref(k)) (2*nv); tau = gear * u on the <motor> model
        CONSTRAINTS  x_0 = x_ref[0], the dynamics defects, and the control box bounds"""

    def __init__(self, dynamics_config, ms_config, track_config, X_ref, dt_ref):
        super().__init__(dynamics_config, ms_config)
        c = track_config
        self.tcfg = c                            # per-motion GaitTOConfig; self.cfg stays the
                                                 # MultiShootingConfig (so keep_best_sol works).

        # reference resampled onto the sim grid; the horizon must span it to within one interval
        X_ref = np.asarray(X_ref, dtype=float).reshape(-1, self.nx)
        self.X_ref = resample_state_trajectory(self.dyn, X_ref, float(dt_ref), self.dt)
        n_fine = self.X_ref.shape[0] - 1
        if not (0 <= self.N_sim - n_fine < self.K):
            raise ValueError(
                f"horizon N_sim={self.N_sim} (N={self.N} x K={self.K}) does not span the "
                f"resampled reference ({self.X_ref.shape[0]} frames); set "
                f"N = ceil((plant_grid_frames(...) - 1) / K).")

        # qpos layout of the floating base: [pos(3) | quat(4, wxyz) | joints]
        self.qa = int(self.quat_addr)            # quaternion start (w)
        self.posz = self.qa - 1                  # base z

        # initial condition: start exactly at the reference's first frame
        self.set_initial_state(self.X_ref[0])

        # tracking weights Qd on the tangent error e = [dq(nv) | dv(nv)]: a soft pull
        nv = self.dyn.nv
        nvj = nv - 6
        w_q = np.concatenate([np.full(3, c.w_base_pos),        # base x, y, z
                              np.full(3, c.w_base_ori),        # base orientation (full, incl. yaw)
                              c.w_joint_pos * np.ones(nvj)])   # joints
        w_v = np.concatenate([np.full(3, c.w_base_linvel),     # base linear velocity
                              np.full(3, c.w_base_angvel),     # base angular velocity
                              c.w_joint_vel * np.ones(nvj)])   # joint velocities
        self.Qd = np.concatenate([w_q, w_v])              # (2*nv,)
        self.term_scale = c.term_scale                    # terminal-cost weight multiplier
        self.Qfd = self.term_scale * self.Qd

        # actuator model, auto-detected from the XML: tau = _tau_du*u + _tau_dq*q + _tau_dqd*qd
        # servo <general>: (kp,-kp,-kd), u = position target;  motor: (gear,0,0), u normalized [-1,1]
        self.act_qadr, self.act_vadr = self.dyn.act_qadr, self.dyn.act_vadr
        self.kp, self.kd = self.dyn.kp, self.dyn.kd     # servo gains (kp=1, kd=0 for a plain motor)
        self.gear = self.dyn.gear                       # transmission gear (1.0 for the servo models)
        self.is_torque = not bool(
            (self.dyn.model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).any())
        if self.is_torque:                              # <motor>: tau = gear * u
            self._tau_du, self._tau_dq, self._tau_dqd = self.gear, np.zeros(self.nu), np.zeros(self.nu)
        else:                                           # <general> PD servo: tau = kp*(u - q) - kd*qd
            self._tau_du, self._tau_dq, self._tau_dqd = self.kp, -self.kp, -self.kd
        self.Rtau = c.Rtau * np.ones(self.nu)           # torque-penalty weight (0 -> off)
        self.Rrate = c.Rrate * np.ones(self.nu)         # rate penalty on u_k - u_{k-1} (0 -> off)

        # per-body Cartesian tracking; track_ee follows the ee_track weights (all zero -> off)
        self.ee = EETracker(self.dyn, c.ee_track_expanded(), c.ee_anchor, self.X_ref,
                            exact_vel_grad=c.exact_vel_grad)
        self.track_ee = self.ee.active

    # reference (index lookup)

    def _ref(self, k):
        """Reference state at sim step k, clamped (the <K-step overhang holds the last frame)."""
        return self.X_ref[min(max(int(k), 0), self.X_ref.shape[0] - 1)]

    # warm start (initial guess)

    def warm_start(self, H):
        """Initial (X0,U0) over H sim steps, by actuator mode: servo -> the reference states and
        their joint angles as position targets; torque -> a PD-around-reference rollout.
        The servo seed is the reference ITSELF, not a rollout of it: the shooting nodes are free
        variables, so seeding them on the clip starts the solve where we want it to end up. The
        defects are nonzero at the seed, which is exactly what the solver is there to close."""
        if self.is_torque:
            return self.pd_warmstart(H)
        idx = np.minimum(np.arange(H), self.X_ref.shape[0] - 1)
        U0 = self.X_ref[idx][:, self.act_qadr]             # u is a joint-position target
        X0 = np.array([self._ref(k) for k in range(H + 1)])   # the reference states, un-rolled
        return X0, U0

    def pd_warmstart(self, H):
        """Roll the motor model out under PD tracking of the reference joints (servo gains captured
        pre-conversion), returning normalized commands u=tau/gear (H,nu) + states (H+1,nx)."""
        kp, kd = self.dyn.servo_kp, self.dyn.servo_kd     # servo PD gains captured before conversion
        x = self.x_init.copy()
        X = [x.copy()]
        U = []
        for k in range(H):
            q = x[self.act_qadr]                           # actuated-joint angles
            qd = x[self.dyn.nq + self.act_vadr]           # actuated-joint velocities
            tau = kp * (self._ref(k)[self.act_qadr] - q) - kd * qd    # physical PD torque [N.m]
            u = np.clip(tau / self.gear, self.dyn.u_lb, self.dyn.u_ub)  # -> normalized command
            x = self.dyn.dynamics(x, u)                    # one mj_step on the torque model
            X.append(x.copy())
            U.append(u)
        return np.array(X), np.array(U)

    def actuator_torque(self, x, u):
        """Physical actuator torque tau (N.m), affine in the control: tau = _tau_du*u + _tau_dq*q +
        _tau_dqd*qd. Servo -> kp*(u-q) - kd*qd; motor -> gear*u. See __init__ for the coefficients."""
        q = x[self.act_qadr]                              # actuated-joint angles
        qd = x[self.dyn.nq + self.act_vadr]               # actuated-joint velocities
        return self._tau_du * u + self._tau_dq * q + self._tau_dqd * qd

    # end-effector Cartesian tracking (delegated to src/end_effector.EETracker)

    def prepare_stage_cache(self, S):
        """MultiShootingBase hook: batch the EE-body FK over the stage states S, memoized once per
        eval point (objective and gradient are called at the same z -> same S)."""
        self.ee.prepare_cache(S)

    # state tracking cost + gradient (+ EE Cartesian tracking)

    def stage_cost(self, x, u, k):
        """Running cost at stage k: state tracking + torque regularizer + EE tracking."""
        # state-space tangent tracking against the reference (full orientation, yaw included)
        e = self.dyn.state_diff(x, self._ref(k))
        c = 0.5 * float(e @ (self.Qd * e))
        tau = self.actuator_torque(x, u)
        c += 0.5 * (tau @ (self.Rtau * tau))              # torque regularizer (always on; Rtau=0 off)
        if self.track_ee:
            c += self.ee.cost(x, k, 1.0, k=k)              # EE pos/ori/vel tracking (see ee_track)
        return float(c)

    def stage_cost_grad(self, x, u, k):
        """Gradient of stage_cost: (d/dx, d/du), same three terms in the same order."""
        # state tracking: matrix-free pull-back of the weighted tangent error
        xr = self._ref(k)
        e = self.dyn.state_diff(x, xr)
        lx = self.dyn.state_diff_tangent_grad(x, xr, self.Qd * e)
        # tau is affine in (u,q,qd); for a motor _tau_dq=_tau_dqd=0 so the state terms vanish
        g = self.Rtau * self.actuator_torque(x, u)
        lu = self._tau_du * g                             # d/du
        lx[self.act_qadr] += self._tau_dq * g             # d/dq_joint
        lx[self.dyn.nq + self.act_vadr] += self._tau_dqd * g   # d/dqd_joint
        if self.track_ee:
            lx += self.ee.grad_full(x, k, 1.0, k=k)        # EE pos/ori/vel tracking (see ee_track)
        return lx, lu

    # terminal cost

    def terminal_cost(self, x):
        """Terminal cost at x_N: the tracking terms only, scaled by term_scale."""
        e = self.dyn.state_diff(x, self._ref(self.N_sim))
        c = 0.5 * float(e @ (self.Qfd * e))
        if self.track_ee:
            c += self.ee.cost(x, self.N_sim, self.term_scale, k=self.N_sim)
        return float(c)

    def terminal_cost_grad(self, x):
        """Gradient of terminal_cost wrt x_N."""
        xr = self._ref(self.N_sim)
        e = self.dyn.state_diff(x, xr)
        lx = self.dyn.state_diff_tangent_grad(x, xr, self.Qfd * e)
        if self.track_ee:
            lx += self.ee.grad_full(x, self.N_sim, self.term_scale, k=self.N_sim)
        return lx

    # action-rate penalty (couples u_k, u_{k-1})
    # Cross-stage, so added at the objective/gradient level over U = spline(p), not in stage_cost.

    def objective(self, z):
        """Base objective + 0.5*dt*sum Rrate*(u_k - u_{k-1})^2; dt-scaled so Rrate is dt-invariant."""
        J = super().objective(z)
        _, U, _ = self.unpack(z)                          # U: (N_sim, nu) per-step controls
        dU = np.diff(U, axis=0)
        return float(J + 0.5 * self.dt * np.sum((self.Rrate * dU) * dU))

    def gradient(self, z):
        """Base gradient + the rate penalty's, pushed through the spline into the p-block."""
        g = super().gradient(z)
        _, U, _ = self.unpack(z)
        L_u = np.zeros_like(U)                            # d(rate cost)/dU, (N_sim, nu)
        d = self.Rrate * np.diff(U, axis=0)               # w*(u_k - u_{k-1}), k=1..N_sim-1
        L_u[1:] += d
        L_u[:-1] -= d
        # dt-scaled; jacobian_T_matvec applies H^T structurally (no dense H)
        g[self.p_slice] += self.dt * self.spline.jacobian_T_matvec(L_u)
        return g


########################################################################
#  MAIN
########################################################################

if __name__ == "__main__":

    # pick the motion config by name:  python g1_gait.py <motion>
    parser = argparse.ArgumentParser(
        description="Unitree G1 (29 dof) walking trajectory optimizer (single full-horizon solve).")
    parser.add_argument("motion", choices=sorted(CONFIGS),
                        help="which motion config to solve (see examples/g1_gait/config.py)")
    args = parser.parse_args()
    cfg = CONFIGS[args.motion]
    print(f"[g1_gait] motion config: {cfg.name}")

    # peek at the model to build the reference (+ its dt) before the TO
    xml_path = os.path.join(ROOT, *cfg.model_xml)
    _m = mujoco.MjModel.from_xml_path(xml_path)
    X_ref, dt_ref = load_reference(cfg.traj_path(ROOT), _m, pz_offset=cfg.pz_offset)

    # shooting grid spanning the whole clip (N derived from the clip length; K = node_dt/sim_dt)
    sim_dt, node_dt = cfg.sim_dt, cfg.node_dt
    n_fine = plant_grid_frames(X_ref.shape[0], dt_ref, sim_dt) - 1
    K = max(1, int(round(node_dt / sim_dt)))
    N = -(-n_fine // K)                                    # ceil division
    print(f"reference: {X_ref.shape[0]} frames @ dt_ref={dt_ref*1e3:.2f} ms "
          f"-> {n_fine} sim steps ({n_fine*sim_dt:.3f} s) at sim_dt={sim_dt*1e3:.0f} ms")
    print(f"shooting grid: N={N} x K={K} (node_dt={node_dt*1e3:.0f} ms)")

    # build the sub-configs from cfg (control bounds default to the model's ctrlrange)
    dyn_cfg = cfg.build_dynamics_config(ROOT)
    spline_cfg = cfg.build_spline_config(N * K)
    ms_cfg = cfg.build_ms_config(N, spline_cfg)
    print(f"control param: spline={cfg.spline} ctrl_hz={cfg.ctrl_hz:g} -> M={spline_cfg.M} knots")

    # build the problem, warm-start, solve (single shot)
    problem = G1GaitTO(dyn_cfg, ms_cfg, cfg, X_ref, dt_ref)
    mode = "torque(motor)" if problem.is_torque else "servo(PD)"
    print(f"initial state: pinned to the reference first frame  |  actuator: {mode}")

    # initial guess: PD-around-reference rollout on the torque model
    X0, U0 = problem.warm_start(problem.N_sim)

    print("\n=== walking TO (single solve) ===")
    X, U, info = problem.solve(X0, U0)
    status = info.get("status_msg", b"")
    status = status.decode() if isinstance(status, (bytes, bytearray)) else status

    # stitch nodes + controls onto the sim grid; report tracking error
    X_fine = problem.stitched_trajectory(X, U)
    err = np.array([problem.dyn.state_diff(X_fine[k], problem._ref(k))
                    for k in range(X_fine.shape[0])])
    rms_q = float(np.sqrt(np.mean(err[:, :problem.dyn.nv] ** 2)))
    rms_v = float(np.sqrt(np.mean(err[:, problem.dyn.nv:] ** 2)))
    zmin, zmax = float(X_fine[:, problem.posz].min()), float(X_fine[:, problem.posz].max())
    print(f"status: {status} | obj={info['obj_val']:.4f} | "
          f"track RMS: pos-tangent={rms_q:.4f} vel={rms_v:.4f} | z=[{zmin:.3f},{zmax:.3f}]")

    # control usage: raw normalized command |u| and the physical joint torque gear*u
    tau_phys = np.array([problem.actuator_torque(X_fine[k], U[min(k, U.shape[0]-1)])
                         for k in range(X_fine.shape[0])])
    print(f"control [{mode}]: |u|_max={np.abs(U).max():.3f} | "
          f"physical torque |tau|_max={np.abs(tau_phys).max():.1f} N.m "
          f"RMS={float(np.sqrt(np.mean(tau_phys ** 2))):.1f} N.m")

    # per-interval defects state_diff(x_{i+1}, Phi_K(x_i, p)), for replay
    ends = problem.dyn.rollout_batch(X[:N], U.reshape(N, K, problem.nu))[:, K]
    defects = np.array([problem.dyn.state_diff(X[i + 1], ends[i]) for i in range(N)])  # (N, ndx)

    # save for replay (replay.py shows the reference ghost and plots the defects)
    t = sim_dt * np.arange(X_fine.shape[0])
    stem = f"g1_gait_{cfg.name}"                        # replay via  replay.py g1_gait_<name>
    save_path = os.path.join(ROOT, "examples", "g1_gait", stem + ".npz")
    save_trajectory(save_path, time=t, state=X_fine, input=U, model=xml_path,
                    reference=problem.X_ref, spline_type=problem.spline.spline_type,
                    defects=defects, node_dt=np.asarray(K * problem.dt))
    print("saved:", save_path)
