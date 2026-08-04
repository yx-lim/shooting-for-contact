##
#
# End-effector / Body-error Tracking Class
#
##

# standard imports
from dataclasses import dataclass
import numpy as np

# local imports
from utils.math_utils import (quat_apply_inverse, quat_mul, quat_conjugate,
                              quat_to_mat, quat_error_to_rotvec)


########################################################################
# END EFFECTOR BODY CONFIG
########################################################################

@dataclass
class EEBody:
    """One tracked body: five weights plus a frame flag.
        w_pxy, w_pz    POSITION, horizontal (world x,y share one weight) and vertical
        w_ori          ORIENTATION, full rotation vector, one weight
        w_v, w_omega   VELOCITY, linear and angular, world-frame 3-vectors
    relative_to_anchor picks the frame for ALL of a body's terms (False -> world-absolute,
    True -> anchor-relative). Weight 0 drops that term; enable=False drops the whole body."""
    body: str
    w_pxy: float
    w_pz: float
    w_ori: float
    w_v: float
    w_omega: float
    relative_to_anchor: bool
    enable: bool = True


def _as_ee_body(row):
    """Parse one config row into an EEBody. Accepts every shape the configs' _expand_ee_row can
    emit, plus an already-built EEBody:
        8-tuple  (enable, body, w_pxy, w_pz, w_ori, w_v, w_omega, rel)
        7-tuple  (        body, w_pxy, w_pz, w_ori, w_v, w_omega, rel)   enable defaults True
    _expand_ee_row emits the 8-tuple for a 7-field config row and the 7-tuple for a 6-field one
    (the variant that omits `enable`), so both must be handled here."""
    if isinstance(row, EEBody):
        return row
    row = tuple(row)
    if len(row) == 8:
        en, body, w_pxy, w_pz, w_ori, w_v, w_om, rel = row
    elif len(row) == 7:
        body, w_pxy, w_pz, w_ori, w_v, w_om, rel = row
        en = True
    else:
        raise ValueError(f"EE_TRACK row must have 7 or 8 fields, got {len(row)}: {row!r}")
    return EEBody(body, w_pxy, w_pz, w_ori, w_v, w_om, bool(rel), bool(en))


########################################################################
# END EFFECTOR TRACKER
########################################################################

class EETracker:
    """Precomputes the reference targets and evaluates the EE tracking cost + gradient.
        once        EETracker(dyn, bodies, anchor, X_ref) -- FKs the whole reference up front
        per eval    prepare_cache(S), one batched FK over all stage states
        per stage   cost(x, idx, wscale, k) / grad_full(x, idx, wscale, k)
    `idx` indexes the (possibly sliding) reference, `k` is the FK-cache key (the horizon stage);
    for a one-shot TO the two coincide, so pass idx == k. `active` is False when nothing is
    tracked (all weights 0 / all bodies disabled) -- the caller should then skip the EE cost."""

    def __init__(self, dyn, bodies, anchor, X_ref, exact_vel_grad=False):
        # exact d/dq of the velocity terms costs 2*nv FK passes PER stage, so it is off by default;
        # the cheap g_dv term (exact wrt qvel) is always on. See _ee_vel_qgrad.
        self.dyn = dyn
        self.nx = dyn.nx
        self.ee_exact_vel_grad = bool(exact_vel_grad)
        rows = [_as_ee_body(b) for b in bodies]
        self._setup(rows, anchor, np.asarray(X_ref, dtype=float).reshape(-1, self.nx))

    def _setup(self, rows, anchor, X_ref):
        """Parse the body spec into vectorized arrays and precompute the reference targets. Per
        term: FK-local body indices `_b`, weights `_w` (1/n_bodies folded in), a `_world` mask
        (True = world, False = anchor-relative), and the reference values.
            POSITION      p_b,  p_axw (per-axis [w_pxy, w_pxy, w_pz]),  p_world,  pos_ref
            ORIENTATION   o_b,  o_w,   o_world,  o_ref_quat + o_ref_relquat
            VELOCITY      v_b,  v_w,   v_world,  v_ref     (linear)
                          a_b,  a_w,   a_world,  a_ref     (angular)
        Also ee_bids (enabled bodies + the anchor, FK'd every eval) and ee_anchor_idx, the
        anchor's row in that output."""
        # the set of bodies to FK each eval: enabled tracked bodies + the anchor, deduped.
        # loc maps a body name to its row in that FK output, which is what p_b/o_b/v_b/a_b index.
        names = list(dict.fromkeys([r.body for r in rows if r.enable] + [anchor]))
        self.ee_bids = self.dyn.body_ids(names)
        loc = {n: i for i, n in enumerate(names)}
        ai = self.ee_anchor_idx = loc[anchor]
        # position bodies (per-axis weights [w_pxy, w_pxy, w_pz]; world or anchor frame per `rel`)
        pb, paxw, pworld = [], [], []
        for r in rows:
            if r.enable and (r.w_pxy != 0.0 or r.w_pz != 0.0):
                pb.append(loc[r.body]); paxw.append([r.w_pxy, r.w_pxy, r.w_pz])
                pworld.append(not r.relative_to_anchor)
        self.p_b = np.asarray(pb, dtype=int)
        self.p_axw = np.asarray(paxw, dtype=float).reshape(-1, 3)
        self.p_world = np.asarray(pworld, dtype=bool)
        # orientation bodies
        ob, ow, oworld = [], [], []
        for r in rows:
            if r.enable and r.w_ori != 0.0:
                ob.append(loc[r.body]); ow.append(float(r.w_ori))
                oworld.append(not r.relative_to_anchor)
        self.o_b = np.asarray(ob, dtype=int); self.o_w = np.asarray(ow, dtype=float)
        self.o_world = np.asarray(oworld, dtype=bool)
        # velocity bodies: linear (w_v) and angular (w_omega), each independent, each carrying the
        # body's frame flag (world vs anchor-relative), like orientation
        vb, vw, vworld, ab, aw, aworld = [], [], [], [], [], []
        for r in rows:
            if r.enable and r.w_v != 0.0:
                vb.append(loc[r.body]); vw.append(float(r.w_v)); vworld.append(not r.relative_to_anchor)
            if r.enable and r.w_omega != 0.0:
                ab.append(loc[r.body]); aw.append(float(r.w_omega)); aworld.append(not r.relative_to_anchor)
        self.v_b = np.asarray(vb, dtype=int); self.v_w = np.asarray(vw, dtype=float)
        self.a_b = np.asarray(ab, dtype=int); self.a_w = np.asarray(aw, dtype=float)
        self.v_world = np.asarray(vworld, dtype=bool)   # True -> world vel, False -> anchor-frame
        self.a_world = np.asarray(aworld, dtype=bool)
        # fold 1/n_bodies into each weight array so every term is the MEAN per-body squared error
        # (mjlab convention) -- weights then stay independent of how many bodies are tracked
        self.p_axw /= max(len(pb), 1)
        self.o_w /= max(len(ob), 1)
        self.v_w /= max(len(vb), 1)
        self.a_w /= max(len(ab), 1)
        # per-term enable flags; with nothing tracked at all there are no references to build
        self.track_ee_pos = len(pb) > 0
        self.track_ee_ori = len(ob) > 0
        self.track_ee_lv = len(vb) > 0
        self.track_ee_av = len(ab) > 0
        self.track_ee = (self.track_ee_pos or self.track_ee_ori
                         or self.track_ee_lv or self.track_ee_av)
        self._ee_S = None; self._ee_fk = None
        if not self.track_ee:
            return
        # reference world pose + twist of all tracked bodies, by FK, once
        ref_pos, ref_quat, ref_lv, ref_av = \
            self.dyn.body_state_batch(X_ref, self.ee_bids)[:4]        # (nf,nb,{3,4,3,3})
        self.ee_nf = ref_pos.shape[0]                                 # reference frame count
        if self.track_ee_pos:
            self.pos_ref = self._ee_body_pos(ref_pos, ref_quat)       # (nf, n_pos, 3)
        if self.track_ee_ori:
            self.o_ref_quat = ref_quat[:, self.o_b, :]                     # (nf, n_ori, 4) world
            qa_ref = ref_quat[:, ai:ai + 1, :]                            # (nf, 1, 4) anchor quat
            self.o_ref_relquat = quat_mul(quat_conjugate(qa_ref),         # (nf, n_ori, 4) anchor-rel
                                          self.o_ref_quat)
        if self.track_ee_lv or self.track_ee_av:
            # frame-appropriate reference velocities (world, or anchor-frame exact derivative)
            lin_ref, ang_ref = self._ee_body_vel(ref_pos, ref_quat, ref_lv, ref_av)  # batched over nf
            if self.track_ee_lv:
                self.v_ref = lin_ref                                      # (nf, n_lv, 3)
            if self.track_ee_av:
                self.a_ref = ang_ref                                      # (nf, n_av, 3)

    @property
    def active(self):
        """True iff at least one tracking term is enabled."""
        return self.track_ee

    def prepare_cache(self, S):
        """Batch the EE-body FK for ALL stage states S in one threaded pass per evaluation point,
        memoized on S (the optimizer calls objective() and gradient() at the same z -> same S, so
        the batched FK runs once per point)."""
        if not self.track_ee:
            return

        # skip the FK entirely when S is unchanged (objective + gradient hit the same point)
        S = np.asarray(S, dtype=float)
        if self._ee_S is not None and self._ee_S.shape == S.shape and np.array_equal(self._ee_S, S):
            return
        self._ee_S = S.copy()
        self._ee_fk = self.dyn.body_state_batch(S, self.ee_bids)

    def _ee_at(self, x, k):
        """(pos (nb,3), quat (nb,4), linvel (nb,3), angvel (nb,3), Jp (nb,3,nv), Jr (nb,3,nv))
        for x: served from the stage cache when x matches stage k, else a direct FK (keeps the
        cost valid for arbitrary states, e.g. FD checks). quat / linvel / angvel / Jr are all
        world-frame (the anchor frame is handled in the cost)."""

        # cache hit only when this x really IS stage k's state; otherwise pay for a one-off FK
        if (self._ee_fk is not None and self._ee_S is not None and k is not None
                and 0 <= k < self._ee_S.shape[0] and np.array_equal(self._ee_S[k], x)):
            f = self._ee_fk
            return f[0][k], f[1][k], f[2][k], f[3][k], f[4][k], f[5][k]
        return self.dyn.body_state(x, self.ee_bids)

    def _ee_body_pos(self, pos, quat):
        """Frame-appropriate positions of the tracked position bodies (..., n_pos, 3). A WORLD
        body keeps its world position p_i; an ANCHOR-relative body uses R_a^T(p_i - p_a) (the
        body position expressed in the anchor frame). quat_apply_inverse(q_a, .) is R_a^T(.);
        leading batch dims (reference frames / FD perturbations) broadcast."""

        # build BOTH framings, then select per body with the world/anchor mask
        a = self.ee_anchor_idx
        pw = pos[..., self.p_b, :]                                    # (...,n_pos,3) world
        rel = quat_apply_inverse(quat[..., a, :][..., None, :],
                                 pos[..., self.p_b, :] - pos[..., a, :][..., None, :])
        return np.where(self.p_world[:, None], pw, rel)

    def _ee_pos_resid(self, pos, quat, idx):
        """Per-body position residual (n_pos, 3) at reference index idx: frame-appropriate position
        (world p_i, or anchor-relative R_a^T(p_i-p_a)) minus the reference's (see _ee_body_pos)."""
        kk = min(max(int(idx), 0), self.ee_nf - 1)
        return self._ee_body_pos(pos, quat) - self.pos_ref[kk]

    def _ee_ori_resid(self, quat, idx):
        """Per-ori-body orientation error as a rotation vector (n_ori, 3) at reference index idx.
        Each body is compared in WORLD frame (o_world) or expressed RELATIVE to the anchor
        (conj(q_a)(x)q_b vs the reference's anchor-relative orientation)."""

        # again both framings, then select: world error vs anchor-relative error
        kk = min(max(int(idx), 0), self.o_ref_quat.shape[0] - 1)
        qb = quat[self.o_b]                                          # (n_ori, 4) world body quats
        e_w = quat_error_to_rotvec(qb, self.o_ref_quat[kk])          # world-frame error
        qa = quat[self.ee_anchor_idx]                                # (4,) anchor quat
        q_rel = quat_mul(quat_conjugate(qa)[None], qb)               # conj(q_a)(x)q_b, (n_ori,4)
        e_a = quat_error_to_rotvec(q_rel, self.o_ref_relquat[kk])    # anchor-relative error
        return np.where(self.o_world[:, None], e_w, e_a)             # (n_ori, 3)

    def _ee_body_vel(self, pos, quat, linvel, angvel):
        """Frame-appropriate velocities of the tracked velocity bodies: lin (...,n_lv,3), ang
        (...,n_av,3). A WORLD body keeps its world velocity; an ANCHOR-relative body gets the
        exact time-derivative of the relative pose, expressed in the anchor frame:
            lin = R_a^T[(v_i - v_a) - omega_a x (p_i - p_a)]     ( = J_relpos_i @ qvel )
            ang = R_a^T(omega_i - omega_a)                       ( = J_relori_i @ qvel )
        the same relative-pose Jacobians the position/orientation terms use. Leading batch dims
        broadcast; either output is an empty (..., 0, 3) array when that term is off."""

        # the anchor's own pose and twist: the frame every relative term is measured against
        a = self.ee_anchor_idx
        qa = quat[..., a, :]                                          # (...,4) anchor quat
        lva = linvel[..., a, :]; ava = angvel[..., a, :]; pa = pos[..., a, :]   # (...,3)

        # linear: world velocity, or the exact d/dt of the relative position (incl. the omega x r term)
        lin = ang = None
        if self.track_ee_lv:
            lw = linvel[..., self.v_b, :]                             # (...,n_lv,3) world linvel
            rel = ((lw - lva[..., None, :])
                   - np.cross(ava[..., None, :], pos[..., self.v_b, :] - pa[..., None, :]))
            rel = quat_apply_inverse(qa[..., None, :], rel)           # R_a^T(...) into anchor frame
            lin = np.where(self.v_world[:, None], lw, rel)
        # angular: world rate, or the relative rate rotated into the anchor frame
        if self.track_ee_av:
            aw = angvel[..., self.a_b, :]                             # (...,n_av,3) world angvel
            rel = quat_apply_inverse(qa[..., None, :], aw - ava[..., None, :])
            ang = np.where(self.a_world[:, None], aw, rel)
        return lin, ang

    def _ee_vel_resid(self, pos, quat, linvel, angvel, idx):
        """Body velocity residuals at reference index idx: (dv (n_lv,3), dw (n_av,3)), each the
        frame-appropriate (world or anchor-frame) velocity minus the reference's (see
        _ee_body_vel). Either is None when that term is off."""

        # frame-appropriate velocity minus the reference's, per enabled term
        kk = min(max(int(idx), 0), self.ee_nf - 1)
        lin, ang = self._ee_body_vel(pos, quat, linvel, angvel)
        dv = (lin - self.v_ref[kk]) if self.track_ee_lv else None
        dw = (ang - self.a_ref[kk]) if self.track_ee_av else None
        return dv, dw

    def _ee_ori_grad_dq(self, quat, Jr, idx, wscale):
        """d/dq of the orientation cost (tangent config part, (nv,)). The orientation log-map
        Jacobian is approximated by identity (small-error regime), so d(rotvec)/dq = Jr_b for
        world bodies and R_a^T(Jr_b - Jr_a) for anchor-relative bodies; the anchor coupling
        (-Jr_a^T sum_b R_a e_b) is summed onto the anchor's rotational Jacobian."""

        # weighted residual, the tracked bodies' rotational Jacobians, and the world/anchor mask
        e = self._ee_ori_resid(quat, idx)                            # (n_ori, 3)
        cw = wscale * self.o_w                                       # (n_ori,)
        Jrb = Jr[self.o_b]                                           # (n_ori, 3, nv)
        mw = self.o_world                                            # (n_ori,) world mask
        # world-frame bodies: direct map sum_b cw_b Jr_b^T e_b (relative bodies zeroed)
        g_dq = np.einsum("bjn,bj->n", Jrb, (cw * mw)[:, None] * e)
        # anchor-relative bodies: W_b = cw_b R_a e_b; sum_b Jr_b^T W_b - Jr_a^T sum_b W_b
        if not np.all(mw):
            Ra = quat_to_mat(quat[self.ee_anchor_idx])               # world <- anchor
            W = ((cw * ~mw)[:, None] * e) @ Ra.T                     # (n_ori, 3)
            g_dq = (g_dq + np.einsum("bjn,bj->n", Jrb, W)
                    - Jr[self.ee_anchor_idx].T @ W.sum(0))
        return g_dq

    def cost(self, x, idx, wscale, k=None):
        """EE tracking cost at state x: wscale * 0.5 * the sum of the enabled terms.
            POSITION      sum_b (w_pxy, w_pxy, w_pz)_b . dp_b^2
            ORIENTATION   sum_b w_ori_b ||rotvec_b||^2
            VELOCITY      sum_b w_v_b ||dv_b||^2  +  sum_b w_omega_b ||dw_b||^2
        dp_b / dv_b / dw_b are the frame-appropriate residuals (see _ee_body_pos, _ee_body_vel).
        Weights carry a 1/n_bodies factor from setup, so each term is the MEAN per-body squared
        error, not a raw sum. `idx` indexes the sliding reference, `k` is the FK-cache key."""

        # sum the enabled terms, each 0.5 * weight * ||residual||^2 over its tracked bodies
        pos, quat, linvel, angvel, _, _ = self._ee_at(x, k)
        c = 0.0
        if self.track_ee_pos:
            d = self._ee_pos_resid(pos, quat, idx)                     # (n_pos, 3)
            c += 0.5 * np.sum(self.p_axw * d * d)
        if self.track_ee_ori:
            eo = self._ee_ori_resid(quat, idx)
            c += 0.5 * np.sum(self.o_w[:, None] * eo * eo)
        if self.track_ee_lv or self.track_ee_av:
            dv, dw = self._ee_vel_resid(pos, quat, linvel, angvel, idx)
            if dv is not None:
                c += 0.5 * np.sum(self.v_w[:, None] * dv * dv)
            if dw is not None:
                c += 0.5 * np.sum(self.a_w[:, None] * dw * dw)
        return float(wscale * c)

    def grad_full(self, x, idx, wscale, k=None):
        """Gradient of cost() in FULL state coordinates, (nx,). Each term maps through the body
        Jacobian for a world body, or the relative-pose Jacobian for an anchor-relative one.
            POSITION      Jp_b            world
                          R_a^T(Jp_b - Jp_a + [p_b - p_a]x Jr_a)    anchor-relative
            ORIENTATION   see _ee_ori_grad_dq
            VELOCITY      d/dqvel = J_vel_b^T residual (the dominant term)
        The velocity term's d/dq configuration coupling is added only when ee_exact_vel_grad is
        set. `idx` indexes the sliding reference, `k` is the FK-cache key."""

        # accumulate into the two tangent halves: g_dq (configuration) and g_dv (velocity)
        pos, quat, linvel, angvel, Jp, Jr = self._ee_at(x, k)
        g_dq = np.zeros(self.dyn.nv)
        g_dv = np.zeros(self.dyn.nv)
        a = self.ee_anchor_idx
        Ra = quat_to_mat(quat[a])                                          # world <- anchor
        if self.track_ee_pos:
            cp = (wscale * self.p_axw) * self._ee_pos_resid(pos, quat, idx)   # (n_pos, 3) weighted resid
            pw = self.p_world[:, None]
            # WORLD bodies: direct map Jp_b^T cp_b
            g_dq += np.einsum("bjn,bj->n", Jp[self.p_b], cp * pw)
            # ANCHOR bodies: relative-position Jacobian transpose, with U_b = R_a cp_b
            if not np.all(self.p_world):
                U = (cp * ~pw) @ Ra.T                                       # (n_pos, 3)
                d = pos[self.p_b] - pos[a]                                  # (n_pos, 3) world offsets
                g_dq += (np.einsum("bjn,bj->n", Jp[self.p_b], U)
                         - Jp[a].T @ U.sum(0)
                         + Jr[a].T @ np.cross(U, d).sum(0))
        if self.track_ee_ori:
            g_dq += self._ee_ori_grad_dq(quat, Jr, idx, wscale)
        if self.track_ee_lv or self.track_ee_av:
            dv, dw = self._ee_vel_resid(pos, quat, linvel, angvel, idx)
            if dv is not None:                                             # d/dqvel of linear term
                cv = (wscale * self.v_w)[:, None] * dv                      # (n_lv, 3) weighted resid
                vw = self.v_world[:, None]
                # WORLD bodies: direct map J_p,b^T cv_b
                g_dv += np.einsum("bjn,bj->n", Jp[self.v_b], cv * vw)
                # ANCHOR bodies: relative-position Jacobian transpose (same coupling as the
                # relative position gradient), with U_b = R_a cv_b
                if not np.all(self.v_world):
                    U = (cv * ~vw) @ Ra.T                                   # (n_lv, 3)
                    d = pos[self.v_b] - pos[a]                              # (n_lv, 3) world offsets
                    g_dv += (np.einsum("bjn,bj->n", Jp[self.v_b], U)
                             - Jp[a].T @ U.sum(0)
                             + Jr[a].T @ np.cross(U, d).sum(0))
            if dw is not None:                                             # d/dqvel of angular term
                cw = (wscale * self.a_w)[:, None] * dw                      # (n_av, 3)
                aw = self.a_world[:, None]
                g_dv += np.einsum("bjn,bj->n", Jr[self.a_b], cw * aw)
                # ANCHOR bodies: relative-orientation Jacobian transpose
                if not np.all(self.a_world):
                    W = (cw * ~aw) @ Ra.T                                   # (n_av, 3)
                    g_dv += np.einsum("bjn,bj->n", Jr[self.a_b], W) - Jr[a].T @ W.sum(0)
            if self.ee_exact_vel_grad:
                g_dq += self._ee_vel_qgrad(x, dv, dw, wscale)

        # stack the halves into a tangent gradient, then pull back to full-state coordinates
        g_tan = np.concatenate([g_dq, g_dv])
        return self.dyn.tangent_grad_to_full(x, g_tan)

    def _ee_vel_qgrad(self, x, dv, dw, wscale, eps=1e-6):
        """Exact d/dq of the velocity terms -- the contribution the d/dqvel map omits because the
        frame velocity vel_b = J_vel_b(q) qvel also depends on the configuration (through the body
        Jacobians, and for anchor bodies through R_a, p_i-p_a, omega_a). Central-differences the
        frame-appropriate body velocities wrt the configuration tangent (velocity held fixed) and
        contracts with the residuals; returns the (nv,) g_dq correction. Cost: 2*nv batched FK
        passes -- gated by ee_exact_vel_grad (off by default)."""

        # build the 2*nv perturbed configurations (+eps / -eps along each config tangent axis)
        nv, ndx = self.dyn.nv, self.dyn.ndx
        Xp = np.empty((2 * nv, self.nx))
        d = np.zeros(ndx)
        for i in range(nv):
            d[:] = 0.0; d[i] = eps;  Xp[2 * i] = self.dyn.state_integrate(x, d)
            d[i] = -eps;             Xp[2 * i + 1] = self.dyn.state_integrate(x, d)

        # one batched FK over all perturbations -> frame velocities at each
        pos_p, quat_p, lvp, avp, _, _ = self.dyn.body_state_batch(Xp, self.ee_bids)  # (2nv,nb,*)
        lin_p, ang_p = self._ee_body_vel(pos_p, quat_p, lvp, avp)   # (2nv, n_lv/n_av, 3) frame vels

        # central difference per axis, contracted with the weighted residual
        g = np.zeros(nv)
        if dv is not None:
            dlin = (lin_p[0::2] - lin_p[1::2]) / (2.0 * eps)        # (nv, n_lv, 3)
            g += np.einsum("qbj,bj->q", dlin, (wscale * self.v_w)[:, None] * dv)
        if dw is not None:
            dang = (ang_p[0::2] - ang_p[1::2]) / (2.0 * eps)        # (nv, n_av, 3)
            g += np.einsum("qbj,bj->q", dang, (wscale * self.a_w)[:, None] * dw)
        return g
