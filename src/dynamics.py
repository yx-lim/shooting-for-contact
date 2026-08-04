##
# 
# Mujoco Dynamics Class
# 
##

# standard imports
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# mujoco imports
import mujoco

# local imports
from utils.math_utils import quat_apply_inverse, quat_mul, quat_conjugate


########################################################################
# CONFIG
########################################################################

# Integrator name -> MuJoCo enum (overrides the model's <option integrator=...>)
INTEGRATORS = {
    "euler":        mujoco.mjtIntegrator.mjINT_EULER,
    "implicit":     mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}

# Friction cone name -> MuJoCo enum (overrides the model's <option cone=...>)
CONES = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic":  mujoco.mjtCone.mjCONE_ELLIPTIC,
}


@dataclass
class DynamicsConfig:

    # Path to the MuJoCo XML model
    model_path: str

    # Optional override of the model timestep
    sim_dt: Optional[float] = None

    # Optional override of the model integrator
    integrator: Optional[str] = "implicitfast"

    # Optional override of contact friction on all geoms (None -> keep the model's per-geom values).
    condim: Optional[int] = None                # 1 (+normal penetration), 3 (+tangetial fric),
                                                # 4 (+torsion), 6 (+rolling)
    friction: Optional[object] = None           # [slide, torsion, roll], 
                                                # [1.0, 0.005, 0.0001] default
    friction_cone: Optional[str] = "pyramidal"  # "pyramidal" or "elliptic"
                                                # friction cone (None -> keep the model's setting)

    # Optional override of the contact solver parameters on all geoms
    # https://mujoco.readthedocs.io/en/stable/modeling.html
    solref: Optional[object] = None  # [timeconst, dampratio], good: timeconst >= 2 x sim_dt
    solimp: Optional[object] = None  # [dmin, dmax, width, midpoint, power]

    # Optional override of the joint-limit solver parameters on all joints
    solreflimit: Optional[object] = None  # [timeconst, dampratio]
    solimplimit: Optional[object] = None  # [dmin, dmax, width, midpoint, power]

    # finite difference settings
    fd_eps: float = 1e-6       # epsilon perturbation for finite-diff Jacobian
    fd_centered: bool = False  # whether to use symmetric (centered) finite differences

    # number of worker threads for the per-node batch dynamics/Jacobian
    n_threads: int = 4

    # toggle MuJoCo sim enforcing the model's configuration limits
    joint_limits: bool = True

    # actuator mode, "position" servo or direct "torque" (default). "torque" converts a <general> PD
    # servo model to a normalized direct-torque <motor> at load time; "position" keeps native actuators.
    actuator_mode: str = "torque"


########################################################################
# DYNAMICS 
########################################################################

class Dynamics:
    """ Discrete-time MuJoCo dynamics: x_{k+1} = f(x_k, u_k) """

    def __init__(self, config: DynamicsConfig):

        # store config for use inside the class
        self.cfg = config

        # instantiate model and data
        self.model = mujoco.MjModel.from_xml_path(config.model_path)

        # override sim step dt
        if config.sim_dt is not None:
            self.model.opt.timestep = config.sim_dt

        # override integrator
        if config.integrator is not None:
            key = config.integrator.lower()
            if key not in INTEGRATORS:
                raise ValueError(
                    f"unknown integrator '{config.integrator}'; "
                    f"choose from {list(INTEGRATORS)}"
                )
            self.model.opt.integrator = INTEGRATORS[key]

        # override contact dimensionality / friction on all geoms (condim=3 -> tangential friction)
        if config.condim is not None:
            self.model.geom_condim[:] = int(config.condim)
        if config.friction is not None:
            fr = np.atleast_1d(np.asarray(config.friction, dtype=float))
            if fr.size == 1:
                self.model.geom_friction[:, 0] = fr[0]      # tangential (slide) only; keep spin/roll
            else:
                self.model.geom_friction[:] = fr            # full [slide, spin, roll]

        # override friction cone
        if config.friction_cone is not None:
            key = config.friction_cone.lower()
            if key not in CONES:
                raise ValueError(
                    f"unknown friction_cone '{config.friction_cone}'; choose from {list(CONES)}"
                )
            self.model.opt.cone = CONES[key]

        # override contact solver parameters on all geoms
        if config.solref is not None:
            self.model.geom_solref[:] = np.asarray(config.solref, dtype=float)  # broadcast to all geoms
        if config.solimp is not None:
            self.model.geom_solimp[:] = np.asarray(config.solimp, dtype=float)  # broadcast to all geoms

        # override joint-limit solver parameters on all joints
        if config.solreflimit is not None:
            self.model.jnt_solref[:] = np.asarray(config.solreflimit, dtype=float)
        if config.solimplimit is not None:
            self.model.jnt_solimp[:] = np.asarray(config.solimplimit, dtype=float)

        # detect a 3D floating base (i.e. the model has a free joint, humanoid, quadrupeds, etc.)
        self.has_3d_floating_base = bool(np.any(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))

        # quaternion address of the floating base
        self.quat_addr: Optional[int] = None
        if self.has_3d_floating_base:
            free_jnt = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
            self.quat_addr = int(self.model.jnt_qposadr[free_jnt]) + 3

        # finalize model and create data
        self.data = mujoco.MjData(self.model)

        # get model dimensions
        self.nq = self.model.nq       # generalized configuration (qpos)
        self.nv = self.model.nv       # generalized velocities (qvel)
        self.nu = self.model.nu       # control inputs (ctrl)
        self.nx = self.nq + self.nv   # full state [qpos, qvel]
        self.ndx = 2 * self.nv        # tangent state [dq, dv]

        # load PD gains set in the model
        self.servo_kp = self.model.actuator_gainprm[:, 0].copy()
        self.servo_kd = -self.model.actuator_biasprm[:, 2].copy()
        if str(config.actuator_mode).lower() == "torque":
            self._convert_to_motor()

        # actuator (position-servo) model + control / effort limits
        self._load_actuator_and_limits()

        # optionally stop the sim from enforcing the model's joint ranges
        if config.joint_limits == False:
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT      # no joint-limit constraints

        # thread pool for the per-node batch dynamics/Jacobian
        self.n_threads = max(1, int(config.n_threads))
        self._pool_data: Optional[list] = None
        self._kin_data: Optional[list] = None
        self._executor: Optional[ThreadPoolExecutor] = None

        # per-node qacc_warmstart caches so each node reseeds the contact solver
        self._ws_cache: dict = {}

        # index maps for the matrix-free gradient transforms (see _build_grad_maps)
        self._build_grad_maps()


    ##############################################################
    # MODEL INTROSPECTION
    ##############################################################

    def _convert_to_motor(self) -> None:
        """Rewrite the model's <general> PD-servo actuators into normalized direct-torque <motor>."""
        m = self.model

        # nothing to do if the model has no PD servos
        if not bool((m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).any()):
            print("[dynamics] actuator_mode='torque': model has no affine PD-servo actuators to "
                  "convert; leaving actuators as authored.")
            return

        # the servo's force cap becomes the motor gear, so u = 1 commands the joint torque limit
        gear = m.actuator_forcerange[:, 1].copy()  # servo force cap == joint torque limit [N.m]
        if not np.all(gear > 0):
            raise ValueError("actuator_mode='torque' needs a positive forcerange upper bound on every "
                             "actuator to use as the motor gear (joint torque limit).")

        # rewrite in place as a normalized motor: tau_joint = gear * u, with u in [-1, 1]
        ones = np.ones(self.nu)
        m.actuator_gaintype[:]     = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[:]     = mujoco.mjtBias.mjBIAS_NONE    # drop PD feedback -> open-loop torque
        m.actuator_dyntype[:]      = mujoco.mjtDyn.mjDYN_NONE      # no activation dynamics
        m.actuator_gainprm[:]      = 0.0
        m.actuator_gainprm[:, 0]   = 1.0                           # f = u
        m.actuator_biasprm[:]      = 0.0                           # no bias term
        m.actuator_gear[:, 0]      = gear                          # tau_joint = gear * f
        m.actuator_ctrlrange[:]    = np.column_stack([-ones, ones])
        m.actuator_ctrllimited[:]  = 1
        m.actuator_forcerange[:]   = np.column_stack([-ones, ones])
        m.actuator_forcelimited[:] = 1

    def _load_actuator_and_limits(self) -> None:
        """Cache the actuator model + limits as attributes, so controllers/costs share one source.

        Actuator PD model (servo) tau = kp*(u - q_joint) - kd*qd_joint; kp/kd read 1/0 for a motor:
          kp, kd              (nu,)  gains = actuator_gainprm[:,0], -actuator_biasprm[:,2]
          act_qadr, act_vadr  (nu,)  qpos / qvel address of each actuator's driven joint (JOINT trans.)

        Limits, taken from the model as authored (MuJoCo enforces both during the sim):
          u_lb/u_ub     control, from ctrlrange (raw, not gated on ctrllimited)
          tau_lb/tau_ub effort, from forcerange (honors forcelimited)
        """
        m = self.model

        # position-servo gains
        self.kp = m.actuator_gainprm[:, 0].copy()                 # (nu,)
        self.kd = -m.actuator_biasprm[:, 2].copy()                # (nu,)

        # actuator -> driven-joint qpos / qvel address (joint transmission)
        self.act_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]]
                                  for a in range(self.nu)], dtype=int)
        self.act_vadr = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]]
                                  for a in range(self.nu)], dtype=int)

        # transmission gear: joint torque = gear * actuator_force, so a <motor> maps ctrl -> gear*ctrl
        # (normalized model: ctrlrange=[-1,1], gear=tau_max). Defaults to 1 for the position servos.
        self.gear = m.actuator_gear[:, 0].copy()                  # (nu,)

        # control limits
        self.u_lb = m.actuator_ctrlrange[:, 0].copy()
        self.u_ub = m.actuator_ctrlrange[:, 1].copy()

        # actuator effort
        flim = m.actuator_forcelimited.astype(bool)
        self.tau_lb = np.where(flim, m.actuator_forcerange[:, 0], -np.inf)
        self.tau_ub = np.where(flim, m.actuator_forcerange[:, 1],  np.inf)


    ##############################################################
    # SET AND GET STATE
    ##############################################################

    def set_state(self, x: np.ndarray, u: np.ndarray) -> None:
        d = self.data
        d.qpos[:] = x[:self.nq]                  # positions
        d.qvel[:] = x[self.nq:self.nq + self.nv] # velocities
        d.ctrl[:] = u                            # control

    def get_state(self) -> np.ndarray:
        d = self.data
        return np.concatenate([d.qpos, d.qvel])


    ##############################################################
    # MANIFOLD OPERATORS
    ##############################################################

    def state_diff(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        """ Tangent difference dx = xa ⊖ xb, shape (ndx,). """
        xa = np.asarray(xa, dtype=float)
        xb = np.asarray(xb, dtype=float)

        # position difference on the manifold: dq points from xb to xa (dt = 1)
        # contains the quaternion-aware difference
        dq = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.model, dq, 1.0, xb[:self.nq], xa[:self.nq])

        # velocity difference is a plain vector difference
        dv = xa[self.nq:] - xb[self.nq:]

        return np.concatenate([dq, dv])

    def state_integrate(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """ Box-plus x_new = x ⊕ dx, with dx in the tangent space (ndx,). """
        x = np.asarray(x, dtype=float)
        dx = np.asarray(dx, dtype=float)

        # integrate the position on the manifold: q <- q ⊕ dq (dt = 1)
        # contains the quaternion-aware integration
        q = x[:self.nq].copy()
        mujoco.mj_integratePos(self.model, q, dx[:self.nv], 1.0)

        # velocity update is a plain vector addition
        v = x[self.nq:] + dx[self.nv:]

        return np.concatenate([q, v])

    @staticmethod
    def _sub_quat_jacobian(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        """ 3x3 Jacobian d(qa ⊖ qb)/d(qa), where qa ⊖ qb is the rotation-vector
        difference (mju_subQuat). The Jacobian wrt qb is -jac^T. This is the SO(3)
        log-map right-Jacobian; it tends to the identity as the angle -> 0.

        Ported from mujoco_mpc DifferentiateSubQuat (mjpc/utilities.cc).
        """
        # rotation vector axis = qa ⊖ qb, then split into angle * unit-axis
        axis = np.zeros(3)
        mujoco.mju_subQuat(axis, qa, qb)
        angle = float(np.linalg.norm(axis))

        # small angle, Jacobian is the identity
        if angle < 1e-10:
            return np.eye(3)
        axis = axis / angle

        # coefficients (th2 = angle / 2)
        th2 = 0.5 * angle
        c0 = th2
        c1 = 1.0 - (1.0 if abs(th2) < 6e-8 else th2 / np.tan(th2))

        # skew-symmetric + outer-product structure
        a0, a1, a2 = axis
        jac = np.array([
            [1.0 + c1 * (-a1*a1 - a2*a2), c1*a0*a1 - c0*a2,            c0*a1 + c1*a0*a2          ],
            [c0*a2 + c1*a0*a1,            1.0 + c1 * (-a0*a0 - a2*a2), c1*a1*a2 - c0*a0          ],
            [c1*a0*a2 - c0*a1,            c0*a0 + c1*a1*a2,            1.0 + c1 * (-a0*a0 - a1*a1)],
        ])
        return jac

    @staticmethod
    def _quat_integrate_jacobian(q: np.ndarray) -> np.ndarray:
        """ 4x3 Jacobian d(q ⊕ dtheta)/d(dtheta) at dtheta=0, for q ⊕ dtheta = q (x) exp(dtheta).
        Right-multiply convention matching mju_quatIntegrate. 
        """
        w, x, y, z = q
        return 0.5 * np.array([
            [-x, -y, -z],
            [ w, -z,  y],
            [ z,  w, -x],
            [-y,  x,  w],
        ])

    @staticmethod
    def _quat_integrate_jacobian_pinv(q: np.ndarray) -> np.ndarray:
        """ Left-inverse (3x4) of _quat_integrate_jacobian G: pinv(G) = (4/||q||^2) G^T.

        The columns of G are orthogonal with G^T G = (||q||^2 / 4) I_3, so the
        pseudo-inverse is available in closed form -- no SVD. Reduces to 4 G^T for a
        unit quaternion, and stays correct for the non-unit quaternions that appear
        mid-iteration (the unit-norm constraint is only satisfied at convergence).
        """
        w, x, y, z = q
        n2 = w * w + x * x + y * y + z * z
        return (4.0 / n2) * Dynamics._quat_integrate_jacobian(q).T

    def tangent_jacobian(self, x: np.ndarray) -> np.ndarray:
        """ Full-state -> tangent map Q_inv = pinv(Q), shape (ndx, nx). Converts a full-state
        Jacobian column into the tangent space; identity with no floating base.

        Q = d(state_integrate(x, dx))/d(dx) at dx = 0 (nx, ndx) is the tangent -> full map:
        identity except a 4x3 _quat_integrate_jacobian block per free/ball quaternion, and
        Q_inv is its left inverse (Q_inv @ Q = I). Built analytically instead of forming Q and
        calling np.linalg.pinv -- Q is block-structured over disjoint qpos-row / dof-col sets,
        so its pinv is the block-wise pinv transposed into place, every block identity but the
        quaternion one (closed form in _quat_integrate_jacobian_pinv). Avoids an SVD per node,
        per IPOPT eval. """
        # fixed base: qpos IS the tangent state, so the map is just the identity
        if not self.has_3d_floating_base:
            return np.eye(self.nx)
        x = np.asarray(x, dtype=float)
        Q_inv = np.zeros((self.ndx, self.nx))

        # position block: one identity entry per scalar dof, the closed-form pinv per quaternion
        m = self.model
        for j in range(m.njnt):
            padr = m.jnt_qposadr[j]   # address in qpos
            vadr = m.jnt_dofadr[j]    # address in qvel / tangent

            jtype = m.jnt_type[j]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                for i in range(3):                       # translation
                    Q_inv[vadr + i, padr + i] = 1.0
                Q_inv[vadr + 3:vadr + 6, padr + 3:padr + 7] = \
                    self._quat_integrate_jacobian_pinv(x[padr + 3:padr + 7])
            elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                Q_inv[vadr:vadr + 3, padr:padr + 4] = \
                    self._quat_integrate_jacobian_pinv(x[padr:padr + 4])
            else:                                        # hinge / slide
                Q_inv[vadr, padr] = 1.0

        # velocity block: qvel maps to itself
        Q_inv[self.nv:, self.nq:] = np.eye(self.nv)
        return Q_inv

    def state_diff_jacobian(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Jacobians of state_diff(x1, x2) wrt tangent perturbations of x1 and x2.

        Returns (J1, J2), each (ndx, ndx), such that for small tangent
        increments d1, d2:
            state_diff(state_integrate(x1, d1), state_integrate(x2, d2))
                ~= state_diff(x1, x2) + J1 @ d1 + J2 @ d2.

        Reduces to (I, -I) when there is no floating base. The position block uses the
        quaternion-aware sub-quat Jacobian on free/ball joints; the velocity block is +/-I.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        nv = self.nv
        J1 = np.zeros((self.ndx, self.ndx))
        J2 = np.zeros((self.ndx, self.ndx))

        # position block (nv x nv): loop joints, fill per joint type
        m = self.model
        for j in range(m.njnt):
            padr = m.jnt_qposadr[j]   # address in qpos
            vadr = m.jnt_dofadr[j]    # address in qvel / tangent

            jtype = m.jnt_type[j]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                # translation: plain difference d(x1 - x2)
                for i in range(3):
                    J1[vadr + i, vadr + i] = 1.0
                    J2[vadr + i, vadr + i] = -1.0
                # rotation: sub-quat block (quaternion lives at padr+3, dofs at vadr+3)
                qa = x1[padr + 3: padr + 7]
                qb = x2[padr + 3: padr + 7]
                jaca = self._sub_quat_jacobian(qa, qb)
                r = vadr + 3
                J1[r:r + 3, r:r + 3] = jaca
                J2[r:r + 3, r:r + 3] = -jaca.T

            elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                # rotation only: the same sub-quat block, with no translation rows
                qa = x1[padr: padr + 4]
                qb = x2[padr: padr + 4]
                jaca = self._sub_quat_jacobian(qa, qb)
                J1[vadr:vadr + 3, vadr:vadr + 3] = jaca
                J2[vadr:vadr + 3, vadr:vadr + 3] = -jaca.T

            else:  # hinge or slide: plain difference
                J1[vadr, vadr] = 1.0
                J2[vadr, vadr] = -1.0

        # velocity block (nv x nv): dv = v1 - v2
        J1[nv:, nv:] = np.eye(nv)
        J2[nv:, nv:] = -np.eye(nv)

        return J1, J2


    ##############################################################
    # MATRIX-FREE GRADIENT TRANSFORMS
    ##############################################################

    def _build_grad_maps(self) -> None:
        """Index maps for the matrix-free gradient transforms below.

        The tangent -> full map Q (nx x ndx), tangent_jacobian (Q_inv, ndx x nx) and
        state_diff_jacobian's J1 (ndx x ndx) are identity everywhere except one
        small block per free/ball-joint quaternion. The gradient pull-backs
        Q^T lx, Q_inv^T g and (J1 Q_inv)^T w therefore reduce to an index copy
        plus a tiny matvec per quaternion -- no large matrices are built. For
        fixed-base models they are pure index copies.
        """
        # walk the joints, splitting them into 1-1 (full_idx <-> tan_idx) and quaternion blocks
        m = self.model
        full_idx, tan_idx, qblocks = [], [], []
        for j in range(m.njnt):
            padr = int(m.jnt_qposadr[j])
            vadr = int(m.jnt_dofadr[j])
            jtype = m.jnt_type[j]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                full_idx += [padr + i for i in range(3)]
                tan_idx += [vadr + i for i in range(3)]
                qblocks.append((padr + 3, vadr + 3))     # quaternion / rotation tangent
            elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                qblocks.append((padr, vadr))
            else:                                        # hinge / slide: 1-1
                full_idx.append(padr)
                tan_idx.append(vadr)
        full_idx += list(range(self.nq, self.nx))        # velocity block, 1-1
        tan_idx += list(range(self.nv, self.ndx))
        self._g_full_idx = np.array(full_idx, dtype=int)
        self._g_tan_idx = np.array(tan_idx, dtype=int)
        self._g_qblocks = qblocks

    def tangent_grad_to_full(self, x: np.ndarray, g_tan: np.ndarray) -> np.ndarray:
        """lx = tangent_jacobian(x).T @ g_tan, computed matrix-free; shape (nx,).

        Maps a TANGENT-space cost gradient to full-state coordinates (the form
        decision variables use). Equivalent to building Q_inv and multiplying.
        """
        x = np.asarray(x, dtype=float)
        g_tan = np.asarray(g_tan, dtype=float)

        # scatter the 1-1 entries, then one 3x4 matvec per quaternion block
        lx = np.zeros(self.nx)
        lx[self._g_full_idx] = g_tan[self._g_tan_idx]
        for (padr, vadr) in self._g_qblocks:
            P = self._quat_integrate_jacobian_pinv(x[padr:padr + 4])   # (3, 4)
            lx[padr:padr + 4] = P.T @ g_tan[vadr:vadr + 3]
        return lx

    def full_grad_to_tangent(self, x: np.ndarray, lx: np.ndarray) -> np.ndarray:
        """g_tan = Q(x).T @ lx, computed matrix-free; shape (ndx,). Q is the tangent -> full
        map d(state_integrate(x, dx))/d(dx) (see tangent_jacobian).

        Maps a FULL-state cost gradient to the tangent space (used to chain
        internal-stage gradients through the K-step sensitivities).
        """
        x = np.asarray(x, dtype=float)
        lx = np.asarray(lx, dtype=float)

        # gather the 1-1 entries, then one 4x3 matvec per quaternion block (the reverse map)
        g = np.zeros(self.ndx)
        g[self._g_tan_idx] = lx[self._g_full_idx]
        for (padr, vadr) in self._g_qblocks:
            G = self._quat_integrate_jacobian(x[padr:padr + 4])        # (4, 3)
            g[vadr:vadr + 3] = G.T @ lx[padr:padr + 4]
        return g

    def state_diff_tangent_grad(self, x1: np.ndarray, x2: np.ndarray,
                                w: np.ndarray) -> np.ndarray:
        """lx = (J1(x1, x2) @ tangent_jacobian(x1)).T @ w, matrix-free; shape (nx,).

        The gradient pull-back of a tangent-space tracking cost: for
        l = f(e) with e = state_diff(x1, x2), dl/dx1 = (J1 Q_inv)^T (df/de).
        J1 is identity except the sub-quat rotation block, so this is one 3x3
        matvec per quaternion followed by tangent_grad_to_full.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        w = np.asarray(w, dtype=float)

        # apply J1's sub-quat block to w, then reuse the tangent -> full pull-back for the rest
        w2 = w.copy()
        for (padr, vadr) in self._g_qblocks:
            jaca = self._sub_quat_jacobian(x1[padr:padr + 4], x2[padr:padr + 4])
            w2[vadr:vadr + 3] = jaca.T @ w[vadr:vadr + 3]
        return self.tangent_grad_to_full(x1, w2)

    ##############################################################
    # DYNAMICS and JACOBIANS
    ##############################################################

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """ One discrete step: x_{k+1} = f(x_k, u_k). """
        # set state
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        self.set_state(x, u)

        # forward step
        mujoco.mj_step(self.model, self.data)

        return self.get_state()


    ##############################################################
    # BATCHED (THREADED) DYNAMICS and JACOBIANS
    ##############################################################

    def _ensure_pool(self) -> None:
        """Lazily build the per-worker MjData pool and the thread pool."""
        if self._executor is None:
            self._pool_data = [mujoco.MjData(self.model) for _ in range(self.n_threads)]
            self._executor = ThreadPoolExecutor(max_workers=self.n_threads)

    def _ensure_kin_pool(self) -> None:
        """Lazily build the kinematics MjData pool (separate from the dynamics
        pool so batched FK never perturbs the dynamics pool's warm-start state)."""
        self._ensure_pool()
        if self._kin_data is None:
            self._kin_data = [mujoco.MjData(self.model) for _ in range(self.n_threads)]

    def _warmstart(self, key: str, shape: tuple) -> np.ndarray:
        """Persistent per-node qacc_warmstart cache for `key`, (re)zeroed when `shape` changes.
        Seeding each node's solve from its own cached row makes the batch independent of the
        thread partition (a node never inherits another node's warm start)."""
        cache = self._ws_cache.get(key)
        if cache is None or cache.shape != shape:
            cache = np.zeros(shape)
            self._ws_cache[key] = cache
        return cache

    def _run_batched(self, K: int, work, pool: Optional[list] = None) -> None:
        """Run `work(data, idxs)` over K items, threaded across an MjData pool
        (the dynamics pool by default).

        Each worker owns one MjData and processes a disjoint, strided set of indices
        (worker w handles w, w+nt, w+2*nt, ...), so no MjData is shared between threads.
        Falls back to a plain serial loop on self.data when threading would not pay off.
        """
        # serial fallback: with one thread or one item, threading cannot pay off
        nt = self.n_threads
        if nt <= 1 or K <= 1:
            work(self.data, range(K))
            return

        # fan out: worker w takes the strided slice w, w+nt, ... so no MjData is ever shared
        self._ensure_pool()
        data = pool if pool is not None else self._pool_data
        assert data is not None and self._executor is not None
        futures = [self._executor.submit(work, data[w], range(w, K, nt))
                   for w in range(nt)]
        for f in futures:
            f.result()   # re-raise any worker exception

    def dynamics_batch(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """ f_k = f(x_k, u_k) for every node, threaded. states (K, nx), controls (K, nu)
        -> F (K, nx). Matches calling dynamics() per node, but parallel over nodes.

        NOTE: rollout_batch with K = 1 computes the same values, so this looks redundant. It is
        kept deliberately: MultiShootingBase's K == 1 path calls this UNCACHED, which gives each
        MjData the same mj_step history -- and therefore the same qacc_warmstart evolution -- as
        the pre-K-step implementation. Routing K = 1 through the cached rollout instead is correct
        but shifts those solves in the last bits, so it is a deliberate, separate decision. Same
        applies to dynamics_jacobian_batch below. """
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float)
        K = states.shape[0]
        F = np.zeros((K, self.nx))
        nq, nv = self.nq, self.nv

        # per node: load the state, seed the contact solver, one mj_step, store the next state
        ws = self._warmstart("node", (K, nv))
        def work(d, idxs):
            for k in idxs:
                x = states[k]
                d.qpos[:] = x[:nq]
                d.qvel[:] = x[nq:nq + nv]
                d.ctrl[:] = controls[k]
                d.qacc_warmstart[:] = ws[k]          # seed from this node's own last solve
                mujoco.mj_step(self.model, d)
                ws[k] = d.qacc_warmstart             # carry it forward (thread-partition-invariant)
                F[k, :nq] = d.qpos
                F[k, nq:] = d.qvel

        self._run_batched(K, work)
        return F

    def dynamics_jacobian_batch(self, states: np.ndarray, controls: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Per-node forward step and FD Jacobians, threaded over nodes.

        states (K, nx), controls (K, nu) -> (F, A, B) with
            F (K, nx)         next states f(x_k, u_k)
            A (K, ndx, ndx)   df/dx in tangent space
            B (K, ndx, nu)    df/du
        One worker pass computes both the nonlinear step (mj_step) and the FD Jacobian
        (mjd_transitionFD) per node, reusing the same MjData."""
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float)
        K = states.shape[0]
        nq, nv = self.nq, self.nv
        F = np.zeros((K, self.nx))
        A = np.zeros((K, self.ndx, self.ndx))
        B = np.zeros((K, self.ndx, self.nu))
        eps, centered = self.cfg.fd_eps, self.cfg.fd_centered

        # per node: one mj_step for f_k, then re-load the same state and FD the Jacobians about it
        ws = self._warmstart("node", (K, nv))
        def work(d, idxs):
            for k in idxs:
                x = states[k]
                u = controls[k]
                # nonlinear next state f_k, seeded from this node's own last solve
                d.qpos[:] = x[:nq]
                d.qvel[:] = x[nq:nq + nv]
                d.ctrl[:] = u
                d.qacc_warmstart[:] = ws[k]
                mujoco.mj_step(self.model, d)
                ws[k] = d.qacc_warmstart             # carry it (before FD perturbs the state)
                F[k, :nq] = d.qpos
                F[k, nq:] = d.qvel
                d.qpos[:] = x[:nq]
                d.qvel[:] = x[nq:nq + nv]
                d.ctrl[:] = u
                d.qacc_warmstart[:] = ws[k]          # same seed for the FD center
                mujoco.mjd_transitionFD(self.model, d, eps, centered,
                                        A[k], B[k], None, None)

        self._run_batched(K, work)
        return F, A, B

    def rollout_batch(self, X0s: np.ndarray, U: np.ndarray) -> np.ndarray:
        """ K-step interval rollouts, threaded over intervals.

        X0s (N, nx) interval start states, U (N, K, nu) the controls of the K
        internal steps of each interval -> Z (N, K+1, nx) with Z[i, 0] = X0s[i]
        and Z[i, j+1] = f(Z[i, j], U[i, j]). The intervals are independent (each
        starts from its own X0s[i]), so they are distributed across the MjData
        pool; the K steps inside an interval are sequential. With K = 1 this
        matches dynamics_batch (plus the leading copy of the start states). """
        X0s = np.asarray(X0s, dtype=float)
        U = np.asarray(U, dtype=float)
        N, K = U.shape[0], U.shape[1]
        Z = np.zeros((N, K + 1, self.nx))
        nq, nv = self.nq, self.nv

        # per interval: K sequential mj_steps, chaining each step's output into the next
        ws = self._warmstart("rollout", (N, K, nv))
        def work(d, idxs):
            for i in idxs:
                Z[i, 0] = X0s[i]
                for j in range(K):
                    z = Z[i, j]
                    d.qpos[:] = z[:nq]
                    d.qvel[:] = z[nq:nq + nv]
                    d.ctrl[:] = U[i, j]
                    d.qacc_warmstart[:] = ws[i, j]   # seed from this step's own last solve
                    mujoco.mj_step(self.model, d)
                    ws[i, j] = d.qacc_warmstart      # carry it (thread-partition-invariant)
                    Z[i, j + 1, :nq] = d.qpos
                    Z[i, j + 1, nq:] = d.qvel

        self._run_batched(N, work)
        return Z

    def rollout_jacobian_batch(self, X0s: np.ndarray, U: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ K-step interval rollouts plus per-internal-step FD Jacobians, threaded over
        intervals. X0s (N, nx), U (N, K, nu) -> (Z, A, B) with
            Z (N, K+1, nx)      internal states, Z[i, 0] = X0s[i]
            A (N, K, ndx, ndx)  df/dx at (Z[i, j], U[i, j]), tangent space
            B (N, K, ndx, nu)   df/du at (Z[i, j], U[i, j])
        Per internal step this follows dynamics_jacobian_batch: one mj_step for the nonlinear
        next state, then reset + mjd_transitionFD. (A_j, B_j) are the per-step terms of the
        K-step sensitivity recursion S^x_{j+1} = A_j S^x_j, S^p_{j+1} = A_j S^p_j + B_j H_j
        (see multi_shooting). """
        X0s = np.asarray(X0s, dtype=float)
        U = np.asarray(U, dtype=float)
        N, K = U.shape[0], U.shape[1]
        nq, nv = self.nq, self.nv
        Z = np.zeros((N, K + 1, self.nx))
        A = np.zeros((N, K, self.ndx, self.ndx))
        B = np.zeros((N, K, self.ndx, self.nu))
        eps, centered = self.cfg.fd_eps, self.cfg.fd_centered

        # per interval: for each of the K steps, mj_step for the next state then FD about it
        ws = self._warmstart("rollout", (N, K, nv))
        def work(d, idxs):
            for i in idxs:
                Z[i, 0] = X0s[i]
                for j in range(K):
                    z = Z[i, j]
                    u = U[i, j]
                    # nonlinear next state, seeded from this step's own last solve
                    d.qpos[:] = z[:nq]
                    d.qvel[:] = z[nq:nq + nv]
                    d.ctrl[:] = u
                    d.qacc_warmstart[:] = ws[i, j]
                    mujoco.mj_step(self.model, d)
                    ws[i, j] = d.qacc_warmstart       # carry it (before FD perturbs the state)
                    Z[i, j + 1, :nq] = d.qpos
                    Z[i, j + 1, nq:] = d.qvel
                    d.qpos[:] = z[:nq]
                    d.qvel[:] = z[nq:nq + nv]
                    d.ctrl[:] = u
                    d.qacc_warmstart[:] = ws[i, j]    # same seed for the FD center
                    mujoco.mjd_transitionFD(self.model, d, eps, centered,
                                            A[i, j], B[i, j], None, None)

        self._run_batched(N, work)
        return Z, A, B

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """ Roll out the dynamics from x0 under an input sequence.
        
        Args:
            x0 : (nx,)    initial state
            U  : (N, nu)  control sequence
        Returns:
            X : (N+1, nx) with X[0] = x0 and X[k+1] = f(X[k], U[k]).
        """
        x0 = np.asarray(x0, dtype=float)
        U = np.asarray(U, dtype=float).reshape(-1, self.nu)

        # sequential chain on self.data -- each step feeds the next (not parallelizable)
        N = U.shape[0]
        X = np.zeros((N + 1, self.nx))
        X[0] = x0
        for k in range(N):
            X[k + 1] = self.dynamics(X[k], U[k])

        return X


    ##############################################################
    # BODY KINEMATICS (world- or anchor-frame pose / twist + Jacobians)
    ##############################################################

    def body_ids(self, names) -> np.ndarray:
        """ Resolve body names to model body ids; raises if any name is missing. """

        # look each name up, failing loudly on the first miss (a typo would silently mistrack)
        ids = []
        for nm in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid < 0:
                raise ValueError(f"body '{nm}' not found in model '{self.cfg.model_path}'")
            ids.append(int(bid))
        return np.array(ids, dtype=int)

    @staticmethod
    def _to_anchor_frame(pos, quat, anchor_pos, anchor_quat):
        """Express world-frame body poses in the anchor body's frame (batched).

        Local pose, matching the tracking example / mjlab structure:
            pos_local  = R_a^T (p_i - p_a)                (quat_apply_inverse)
            quat_local = conj(q_a) (x) q_i
        `pos`/`quat` are (..., nb, 3)/(..., nb, 4); `anchor_pos`/`anchor_quat` are the anchor's
        world pose with the SAME leading dims and a singleton body axis ((..., 1, 3)/(..., 1, 4))
        so they broadcast over the nb bodies. The anchor's own row maps to (0, identity).
        Velocities are intentionally left in the WORLD frame by the callers (mjlab convention),
        so only pose is transformed here."""
        pos_local = quat_apply_inverse(anchor_quat, pos - anchor_pos)
        quat_local = quat_mul(quat_conjugate(anchor_quat), quat)
        return pos_local, quat_local

    def body_state(self, x: np.ndarray, bids: np.ndarray, anchor=None):
        """ Pose, spatial velocity, and body Jacobians of `bids` body IDs at state x. Returns
            pos    (nb, 3)      body frame-origin position    (world, or anchor-relative)
            quat   (nb, 4)      orientation, MuJoCo wxyz      (world, or anchor-relative)
            linvel (nb, 3)      world linear  velocity = Jp @ qvel
            angvel (nb, 3)      world angular velocity = Jr @ qvel
            Jp, Jr (nb, 3, nv)  translational / rotational body Jacobian, world
        anchor=None -> all world. A body id instead expresses each POSE in that body's frame
        (see _to_anchor_frame); velocities and Jacobians stay world. Velocities come from the
        Jacobians, not mj_objectVelocity, so a cost and its analytic gradient use the same
        quantities. Costs one mj_forward plus one mj_jacBody per body. """
        # FK at x: populates xpos / xquat and everything mj_jacBody needs
        x = np.asarray(x, dtype=float)
        d = self.data
        d.qpos[:] = x[:self.nq]
        d.qvel[:] = x[self.nq:self.nq + self.nv]
        mujoco.mj_forward(self.model, d)

        # read each body's world pose, then its translational / rotational Jacobian
        bids = np.asarray(bids, dtype=int)
        nb = bids.shape[0]
        pos = d.xpos[bids].copy()
        quat = d.xquat[bids].copy()
        Jp = np.zeros((nb, 3, self.nv))
        Jr = np.zeros((nb, 3, self.nv))
        jp = np.zeros((3, self.nv))
        jr = np.zeros((3, self.nv))
        for i in range(nb):
            mujoco.mj_jacBody(self.model, d, jp, jr, int(bids[i]))
            Jp[i] = jp
            Jr[i] = jr

        # twists straight from the Jacobians, so cost and gradient see identical quantities
        qvel = d.qvel
        linvel = Jp @ qvel     # (nb,3,nv) @ (nv,) -> (nb,3)
        angvel = Jr @ qvel     # (nb,3,nv) @ (nv,) -> (nb,3)

        # optionally re-express the poses in the anchor's frame (velocities stay world)
        if anchor is not None:
            ap = d.xpos[int(anchor)].copy()[None, :]    # (1,3) broadcast over bodies
            aq = d.xquat[int(anchor)].copy()[None, :]   # (1,4)
            pos, quat = self._to_anchor_frame(pos, quat, ap, aq)
        return pos, quat, linvel, angvel, Jp, Jr

    def body_state_batch(self, X: np.ndarray, bids: np.ndarray, anchor=None):
        """body_state over a batch of states, row-wise identical to it: X (n, nx), bids (nb,)
        -> pos (n,nb,3), quat (n,nb,4), linvel (n,nb,3), angvel (n,nb,3), Jp/Jr (n,nb,3,nv).
        `anchor` means what it does in body_state.

        Threaded across a SEPARATE MjData pool: FK is pure kinematics (independent of MjData
        history), and a separate pool leaves the dynamics pool's warm starts untouched. Runs
        only mj_kinematics + mj_comPos per state -- all mj_jacBody needs -- not full
        mj_forward. """

        # convert to arrays and allocate outputs
        X = np.asarray(X, dtype=float)
        bids = np.asarray(bids, dtype=int)
        n, nb = X.shape[0], bids.shape[0]
        nq, nv = self.nq, self.nv
        pos = np.zeros((n, nb, 3))
        quat = np.zeros((n, nb, 4))
        Jp = np.zeros((n, nb, 3, nv))
        Jr = np.zeros((n, nb, 3, nv))

        # anchor world pose per state (only when an anchor frame is requested)
        anchor_id = None if anchor is None else int(anchor)
        apos = np.zeros((n, 3)) if anchor_id is not None else None
        aquat = np.zeros((n, 4)) if anchor_id is not None else None

        # per state: kinematics only (no dynamics), then pose + Jacobian for every body
        def work(d, idxs):
            jp = np.zeros((3, nv))
            jr = np.zeros((3, nv))
            for k in idxs:
                d.qpos[:] = X[k, :nq]
                mujoco.mj_kinematics(self.model, d)
                mujoco.mj_comPos(self.model, d)
                pos[k] = d.xpos[bids]
                quat[k] = d.xquat[bids]
                if anchor_id is not None:
                    apos[k] = d.xpos[anchor_id]
                    aquat[k] = d.xquat[anchor_id]
                for i in range(nb):
                    mujoco.mj_jacBody(self.model, d, jp, jr, int(bids[i]))
                    Jp[k, i] = jp
                    Jr[k, i] = jr

        # run the batch, threaded if worthwhile (n > 1)
        if self.n_threads > 1 and n > 1:
            self._ensure_kin_pool()          # _run_batched falls back to serial without a pool
        self._run_batched(n, work, pool=self._kin_data)

        # twists for every state at once: contract each body Jacobian with that row's qvel
        V = X[:, nq:nq + nv]
        linvel = np.einsum("kbjv,kv->kbj", Jp, V)
        angvel = np.einsum("kbjv,kv->kbj", Jr, V)

        # optionally re-express the poses in the anchor's frame (velocities stay world)
        if anchor_id is not None:
            pos, quat = self._to_anchor_frame(pos, quat, apos[:, None, :], aquat[:, None, :])
        return pos, quat, linvel, angvel, Jp, Jr
