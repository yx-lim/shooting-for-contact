##
#
# Unitree Go2 (12 dof) reference trajectory tracking with MPC.
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
from src.mpc import ModelPredictiveControl
from utils.file_utils import load_reference, save_trajectory
from utils.math_utils import resample_state_trajectory
from src.end_effector import EETracker
from config import CONFIGS


########################################################################
# GO2 TRACKING MPC
########################################################################

class Go2TrackingMPC(ModelPredictiveControl):
    """Receding-horizon tracking of a reference clip: state-space + end-effector Cartesian.
        COST        0.5 e^T Qd e  +  0.5 tau^T Rtau tau  +  EE terms  +  a control-rate penalty
                    e = state_diff(x, x_ref(t)) (2*nv); tau from the auto-detected actuator model
        CONSTRAINTS multiple-shooting dynamics defects + control box bounds
    Weights and which terms are active come from the per-motion MPCTrackingConfig."""

    def __init__(self, dynamics_config, mpc_config, track_config, X_ref, dt_ref):
        super().__init__(dynamics_config, mpc_config)
        self.cfg = track_config                            # per-motion MPCTrackingConfig
        c = self.cfg

        # upscale the reference onto the plant grid once, so x_ref(t) is an exact O(1) lookup.
        X_ref = np.asarray(X_ref, dtype=float).reshape(-1, self.nx)
        self.X_ref = resample_state_trajectory(self.dyn, X_ref, float(dt_ref), self.dt)

        # state-space tracking weights (tangent error e = [dq(nv) | dv(nv)]), from cfg
        # base position/orientation firmly, joints moderately, velocities lightly.
        nv = self.dyn.nv
        nvj = nv - 6
        w_q = np.concatenate([np.full(3, c.w_base_pos),        # base x, y, z
                              np.full(3, c.w_base_ori),        # base orientation
                              c.w_joint_pos * np.ones(nvj)])   # joints
        w_v = np.concatenate([np.full(3, c.w_base_linvel),     # base linear velocity
                              np.full(3, c.w_base_angvel),     # base angular velocity
                              c.w_joint_vel * np.ones(nvj)])    # joint velocities
        self.Qd = np.concatenate([w_q, w_v])              # (2*nv,)
        self.term_scale = c.term_scale                     # terminal-cost weight multiplier
        self.Qfd = self.term_scale * self.Qd

        # actuator model, auto-detected from the XML: tau = _tau_du*u + _tau_dq*q + _tau_dqd*qd
        # servo <general>: (kp,-kp,-kd), u = position target;  motor: (gear,0,0), u normalized [-1,1]
        self.act_qadr, self.act_vadr = self.dyn.act_qadr, self.dyn.act_vadr
        self.kp, self.kd = self.dyn.kp, self.dyn.kd     # servo gains (kp=1, kd=0 for a plain motor)
        self.gear = self.dyn.gear                       # transmission gear (1.0 for the servo models)
        self.is_torque = not bool(
            (self.dyn.model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).any())
        if self.is_torque:                              # <motor>: tau = gear * u (u normalized [-1,1])
            self._tau_du, self._tau_dq, self._tau_dqd = self.gear, np.zeros(self.nu), np.zeros(self.nu)
        else:                                           # <general> PD servo: tau = kp*(u - q) - kd*qd
            self._tau_du, self._tau_dq, self._tau_dqd = self.kp, -self.kp, -self.kd
        self.Rtau = c.Rtau * np.ones(self.nu)           # torque-penalty weight (0 -> off)
        self.Rrate = c.Rrate * np.ones(self.nu)         # rate penalty on u_k - u_{k-1} (0 -> off)

        # cost-term selection (from cfg): state-space tracking and/or EE/body tracking
        self.track_state = bool(c.track_state)
        if c.track_body:
            self.ee = EETracker(self.dyn, c.ee_track_expanded(), c.ee_anchor, self.X_ref,
                                exact_vel_grad=c.exact_vel_grad)
            self.track_ee = self.ee.active       # from the enabled ee_track rows' weights
        else:
            self.ee = None
            self.track_ee = False
        if not self.track_state and not self.track_ee:
            raise ValueError("no tracking terms enabled: set track_state and/or track_body (and, "
                             "for body tracking, give ee_track at least one nonzero weight)")

        # report which terms ended up active for this motion
        n_bodies = sum(1 for row in c.ee_track if row[0]) if self.track_ee else 0
        print(f"[Go2TrackingMPC] motion={c.name}  actuator={'torque' if self.is_torque else 'position'}"
              f"  tracking: state={self.track_state}  "
              f"body={self.track_ee} ({n_bodies}/{len(c.ee_track)} bodies enabled)")

    # time-varying reference

    def _time(self, k):
        """Absolute sim time of horizon stage k (t_now is set each MPC step)."""
        return self.t_now + k * self.dt

    def _ref(self, t):
        """Reference state at absolute time t: O(1) lookup into the upscaled (plant-grid)
        reference, clamped at the trajectory ends."""
        idx = int(round(t / self.dt))
        idx = min(max(idx, 0), self.X_ref.shape[0] - 1)
        return self.X_ref[idx]

    def _ref_idx(self, t):
        """Reference frame index at absolute time t (clamped). Used to look up the precomputed
        EE reference targets; decoupled from the horizon stage k (the FK-cache key)."""
        idx = int(round(t / self.dt))
        return min(max(idx, 0), self.X_ref.shape[0] - 1)

    def reference_controls(self, H):
        """Servo warm-start controls: reference actuated-joint angles for the first H steps, (H, nu).
        u is a position target, so commanding these drives the servos to the reference pose."""
        idx = np.minimum(np.arange(H), self.X_ref.shape[0] - 1)
        return self.X_ref[idx][:, self.act_qadr]

    def warm_start(self, H, x_init):
        """Initial (X_ws, U_ws) over H sim steps from x_init, by actuator mode: torque -> a
        PD-around-reference rollout; servo -> the reference joint angles as position targets,
        rolled out for a dynamically consistent seed. Returns (X (H+1,nx), U (H,nu))."""
        if self.is_torque and getattr(self.cfg, "warm_mode", "pd") == "pd":
            return self.pd_warmstart(H, x_init)
        U0 = self.reference_controls(H)
        X0 = self.dyn.rollout(np.asarray(x_init, dtype=float), U0)
        return X0, U0

    def pd_warmstart(self, H, x_init):
        """Torque-mode seed: roll the motor model out under PD tracking of the reference joints,
        using the servo gains the base model carried before the motor conversion. Returns the
        visited states (H+1,nx) and the normalized commands u = tau/gear (H,nu), clipped to bounds."""
        kp, kd = self.dyn.servo_kp, self.dyn.servo_kd     # servo PD gains captured before conversion
        x = np.asarray(x_init, dtype=float).copy()
        X = [x.copy()]
        U = []
        for k in range(H):
            q = x[self.act_qadr]                           # actuated-joint angles
            qd = x[self.dyn.nq + self.act_vadr]           # actuated-joint velocities
            tau = kp * (self._ref(k * self.dt)[self.act_qadr] - q) - kd * qd  # physical PD torque [N.m]
            u = np.clip(tau / self.gear, self.dyn.u_lb, self.dyn.u_ub)        # -> normalized command
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
        """Hook from MultiShootingBase: batch the EE-body FK over all stage states S (memoized on
        S, so the batched FK runs once per objective+gradient evaluation point)."""
        if self.ee is not None:
            self.ee.prepare_cache(S)

    # state-space tangent tracking cost + gradient (+ EE Cartesian tracking)

    def stage_cost(self, x, u, k):
        """Running cost at stage k: state tracking + torque regularizer + EE tracking."""
        c = 0.0
        if self.track_state:                              # state-space tangent tracking
            e = self.dyn.state_diff(x, self._ref(self._time(k)))
            c += 0.5 * e @ (self.Qd * e)
        tau = self.actuator_torque(x, u)
        c += 0.5 * (tau @ (self.Rtau * tau))              # torque regularizer (always on; Rtau=0 off)
        if self.track_ee:
            c += self.ee.cost(x, self._ref_idx(self._time(k)), 1.0, k)   # EE pos/ori/vel (see ee_track)
        return float(c)

    def stage_cost_grad(self, x, u, k):
        """Gradient of stage_cost: (d/dx, d/du), same three terms in the same order."""
        if self.track_state:                              # matrix-free pull-back of Qd*e
            xr = self._ref(self._time(k))
            e = self.dyn.state_diff(x, xr)
            lx = self.dyn.state_diff_tangent_grad(x, xr, self.Qd * e)
        else:
            lx = np.zeros(self.nx)
        # torque penalty: tau(x,u) is affine in (u, q, qd) with the per-mode coefficients; g = Rtau*tau.
        # For a motor _tau_dq=_tau_dqd=0, so the state-gradient terms vanish (torque is state-free).
        g = self.Rtau * self.actuator_torque(x, u)
        lu = self._tau_du * g                             # d/du
        lx[self.act_qadr] += self._tau_dq * g             # d/dq_joint
        lx[self.dyn.nq + self.act_vadr] += self._tau_dqd * g   # d/dqd_joint
        if self.track_ee:
            lx += self.ee.grad_full(x, self._ref_idx(self._time(k)), 1.0, k)  # EE (see ee_track)
        return lx, lu

    # terminal cost

    def terminal_cost(self, x):
        """Terminal cost at x_N: the tracking terms only, scaled by term_scale (no torque term)."""
        c = 0.0
        if self.track_state:
            e = self.dyn.state_diff(x, self._ref(self._time(self.N_sim)))
            c += 0.5 * e @ (self.Qfd * e)
        if self.track_ee:
            c += self.ee.cost(x, self._ref_idx(self._time(self.N_sim)), self.term_scale, self.N_sim)
        return float(c)

    def terminal_cost_grad(self, x):
        """Gradient of terminal_cost wrt x_N."""
        if self.track_state:
            xr = self._ref(self._time(self.N_sim))
            e = self.dyn.state_diff(x, xr)
            lx = self.dyn.state_diff_tangent_grad(x, xr, self.Qfd * e)
        else:
            lx = np.zeros(self.nx)
        if self.track_ee:
            lx += self.ee.grad_full(x, self._ref_idx(self._time(self.N_sim)), self.term_scale, self.N_sim)
        return lx

    # control-rate penalty (couples u_k, u_{k-1})
    # Cross-stage, so added at the objective/gradient level over U = spline(p), not in stage_cost.

    def objective(self, z):
        """Base objective + 0.5*dt*sum Rrate*(u_k - u_{k-1})^2; dt-scaled so Rrate is dt-invariant."""
        J = super().objective(z)
        _, U, _ = self.unpack(z)                          # U: (N_sim, nu) per-step controls
        dU = np.diff(U, axis=0)                           # u_k - u_{k-1}
        return float(J + 0.5 * self.dt * np.sum((self.Rrate * dU) * dU))

    def gradient(self, z):
        """Base gradient + the rate penalty's, pushed through the spline into the p-block."""
        g = super().gradient(z)
        _, U, _ = self.unpack(z)
        L_u = np.zeros_like(U)                            # d(rate cost)/dU, (N_sim, nu)
        d = self.Rrate * np.diff(U, axis=0)               # w*(u_k - u_{k-1}), k=1..N_sim-1
        L_u[1:] += d
        L_u[:-1] -= d
        # dt-scaled (matches objective); jacobian_T_matvec applies H^T structurally (no dense H)
        g[self.p_slice] += self.dt * self.spline.jacobian_T_matvec(L_u)
        return g


########################################################################
#  MAIN
########################################################################

if __name__ == "__main__":

    # pick the motion config by name:  python go2_tracking_mpc.py <motion>
    parser = argparse.ArgumentParser(
        description="Unitree Go2 (12 dof) reference-tracking MPC; pick a motion from config.CONFIGS.")
    parser.add_argument("motion", choices=sorted(CONFIGS),
                        help="which motion config to track (see examples/go2_tracking_mpc/config.py)")
    args = parser.parse_args()
    cfg = CONFIGS[args.motion]
    print(f"[go2_tracking_mpc] motion config: {cfg.name}")

    # peek at the model to build the reference (and its dt) before the MPC is created
    xml_path = os.path.join(ROOT, *cfg.model_xml)
    _m = mujoco.MjModel.from_xml_path(xml_path)
    X_ref, dt_ref = load_reference(cfg.traj_path(ROOT), _m,
                                   pz_offset=cfg.pz_offset, centered=True)
    print(f"reference: {X_ref.shape[0]} frames, "
          f"dt_ref={dt_ref*1e3:.2f} ms, {(X_ref.shape[0]-1)*dt_ref:.2f} s")

    # closed-loop length: cfg.T if set, else the full reference duration.
    # H (nodes) / node_dt / mpc_dt come from cfg; each interval integrates K = node_dt/sim_dt steps.
    T = cfg.T if cfg.T is not None else (X_ref.shape[0] - 1) * dt_ref

    # optional qpos box on the shooting nodes (NLP variable bounds), gated by cfg.nlp_qpos_bounds.
    # IPOPT holds these every iterate (so they survive early stops); only scalar joints get bounds.
    x_lb = x_ub = None
    if cfg.nlp_qpos_bounds:
        q_lb = np.full(_m.nq, -np.inf)
        q_ub = np.full(_m.nq,  np.inf)
        _scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        for j in range(_m.njnt):
            if _m.jnt_limited[j] and int(_m.jnt_type[j]) in _scalar:
                a = _m.jnt_qposadr[j]
                q_lb[a] = _m.jnt_range[j, 0]
                q_ub[a] = _m.jnt_range[j, 1]
        x_lb = np.concatenate([q_lb, np.full(_m.nv, -np.inf)])   # (nx,) -- bound joint qpos only
        x_ub = np.concatenate([q_ub, np.full(_m.nv,  np.inf)])

    # build the sub-configs from cfg (control bounds default to the model's ctrlrange).
    dyn_cfg = cfg.build_dynamics_config(ROOT)
    mpc_cfg = cfg.build_mpc_config(T, x_lb, x_ub)

    # instantiate the MPC and start from the first reference state
    mpc = Go2TrackingMPC(dyn_cfg, mpc_cfg, cfg, X_ref, dt_ref)
    x_init = X_ref[0].copy()

    # warm start (dispatched by actuator mode): servo -> reference joint-angle rollout; torque ->
    # PD-around-reference rollout. set_warm_start subsamples X to the nodes.
    X_ws, U_ws = mpc.warm_start(mpc.N_sim, x_init)
    mpc.set_warm_start(X_ws, U_ws)

    # run the closed loop
    t, X, U = mpc.run(x_init)

    # report how well the closed loop actually followed the clip (base position / height only --
    # the headline numbers; the full per-state comparison is in the replay plots).
    n = min(len(X), mpc.X_ref.shape[0])
    err_xy = np.linalg.norm(X[:n, :2] - mpc.X_ref[:n, :2], axis=1)
    err_z = np.abs(X[:n, 2] - mpc.X_ref[:n, 2])
    print(f"[tracking] base xy error: mean {err_xy.mean()*1e3:.1f} mm, max {err_xy.max()*1e3:.1f} mm"
          f" | base z error: mean {err_z.mean()*1e3:.1f} mm, max {err_z.max()*1e3:.1f} mm")

    # save the closed-loop trajectory for replay (same schema/tags as the TO).
    # replay via  replay.py go2_tracking_<name>   (the stem is unique, so no folder needed)
    stem = f"go2_tracking_{cfg.name}"
    save_path = os.path.join(ROOT, "examples", "go2_tracking_mpc", stem + ".npz")
    save_trajectory(save_path, time=t, state=X, input=U, model=xml_path, reference=mpc.X_ref,
                    spline_type=mpc.spline.spline_type, node_dt=np.asarray(mpc.K * mpc.dt))
    print("saved:", save_path)
