##
#
# Math Utilities
#
##

import numpy as np
import mujoco


########################################################################
# FINITE DIFFERENCING
########################################################################

def differentiate_qpos(model, Q, dt, centered=False):
    """Quaternion-aware velocity from a qpos trajectory.

    Given Q (T, nq) sampled at fixed dt, return V (T, nv) where each row v_k satisfies
    mj_integratePos(q_k, v_k * dt) ~= q_{k+1}. Free/ball-joint quaternions are differenced
    on the manifold via mujoco.mj_differentiatePos, so V is in MuJoCo's qvel convention and
    is directly usable as the velocity half of a full state [qpos, qvel].

    Forward difference by default; centered uses symmetric interior differences. The final
    velocity repeats the previous one (forward) so len(V) == len(Q).
    """
    # a single frame carries no velocity information -> all zeros
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    T = Q.shape[0]
    V = np.zeros((T, model.nv))
    if T < 2:
        return V

    dq = np.zeros(model.nv)
    if centered:
        # symmetric difference over 2*dt on the interior, one-sided at the two endpoints
        for k in range(1, T - 1):
            mujoco.mj_differentiatePos(model, dq, 2.0 * dt, Q[k - 1], Q[k + 1])
            V[k] = dq
        mujoco.mj_differentiatePos(model, dq, dt, Q[0], Q[1]);   V[0] = dq
        mujoco.mj_differentiatePos(model, dq, dt, Q[-2], Q[-1]); V[-1] = dq
    else:
        # forward difference; the last row repeats the previous so len(V) == len(Q)
        for k in range(T - 1):
            mujoco.mj_differentiatePos(model, dq, dt, Q[k], Q[k + 1])
            V[k] = dq
        V[-1] = V[-2]
    return V


def interpolate_state(dyn, X_ref, dt_ref, t):
    """Geodesic interpolation of a full-state reference sampled uniformly at dt_ref,
    evaluated at absolute time t.

    X_ref is (N, nx) = [qpos, qvel] per row. Between the two bracketing samples we move
    along the manifold geodesic: dx = state_diff(x_{i+1}, x_i) is the tangent from x_i to
    x_{i+1}, and the interpolant is state_integrate(x_i, alpha * dx). This is SLERP on the
    free-joint quaternion and linear interpolation on positions and velocities. Clamped to
    the trajectory ends for t outside [0, (N-1)*dt_ref].
    """
    # locate the sample bracketing t
    X_ref = np.asarray(X_ref, dtype=float)
    N = X_ref.shape[0]
    s = t / dt_ref
    i0 = int(np.floor(s))

    # clamp to the endpoints for t outside the trajectory
    if i0 < 0:
        return X_ref[0].copy()
    if i0 >= N - 1:
        return X_ref[N - 1].copy()

    # walk a fraction alpha along the geodesic from x_i0 toward x_i0+1
    alpha = s - i0
    dx = dyn.state_diff(X_ref[i0 + 1], X_ref[i0])      # tangent x_i0 -> x_i0+1
    return dyn.state_integrate(X_ref[i0], alpha * dx)


def resample_state_trajectory(dyn, X_ref, dt_ref, dt_new):
    """Resample a full-state reference (sampled at dt_ref) onto a uniform grid at dt_new,
    spanning the same total duration. Each output sample is geodesically interpolated via
    interpolate_state (SLERP on the base quaternion, linear on the rest), so upsampling to
    a finer dt_new keeps the quaternion on the manifold. Returns X_new (M, nx) with
    M = round((N-1) * dt_ref / dt_new) + 1; the endpoints coincide with X_ref's.
    """
    # output grid: same total duration, new step -> M samples
    X_ref = np.asarray(X_ref, dtype=float)
    N = X_ref.shape[0]
    total = (N - 1) * dt_ref
    M = int(round(total / dt_new)) + 1

    # geodesically interpolate the reference at each new sample time
    X_new = np.zeros((M, X_ref.shape[1]))
    for k in range(M):
        X_new[k] = interpolate_state(dyn, X_ref, dt_ref, k * dt_new)
    return X_new


########################################################################
# ROTATIONS / QUATERNIONS  (MuJoCo convention: wxyz, scalar-first)
########################################################################

def normalize_quat(q):
    """Return q / ||q|| (wxyz), guarding against a zero quaternion."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n > 0.0, n, 1.0)


def quat_mul(a, b):
    """Hamilton product a (x) b for wxyz quaternions, broadcasting over leading dims; (...,4)."""
    # split both operands into components, then assemble the four Hamilton product terms
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw], axis=-1)


def quat_conjugate(q):
    """Conjugate (inverse for a unit quaternion) of wxyz quaternions, batched; (...,4)."""
    return np.asarray(q, dtype=float) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_rotvec(q):
    """Log map of wxyz quaternions to rotation vectors (axis * angle), batched; (...,4)->(...,3).
    Returns the SHORTEST rotation (angle wrapped to (-pi, pi]); matches mju_quat2Vel."""
    # recover the rotation angle from the scalar/vector split, wrapped to the shortest arc
    q = normalize_quat(q)
    w = q[..., 0]
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)                     # sin(theta/2)
    theta = 2.0 * np.arctan2(s, w)
    theta = np.where(theta > np.pi, theta - 2.0 * np.pi, theta)

    # rescale the vector part from sin(theta/2) to theta (-> 0 at identity, where s vanishes)
    scale = np.where(s > 1e-12, theta / np.where(s > 1e-12, s, 1.0), 0.0)
    return v * scale[..., None]


def quat_to_mat(q):
    """3x3 rotation matrix R(q) for a single wxyz quaternion."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def quat_error_to_rotvec(q, q_ref):
    """Orientation error as a rotation vector, log(q (x) conj(q_ref)), batched; (...,4)->(...,3).
    This is the geodesic "box-minus" of two wxyz quaternion fields."""
    return quat_to_rotvec(quat_mul(q, quat_conjugate(q_ref)))


def quat_apply_inverse(quat, vec):
    """Rotate `vec` from the world frame into the body frame of `quat` (wxyz): R(quat)^T @ vec.

    Batched and singularity-free (no Euler angles). `quat` is (..., 4) wxyz and `vec` is
    (..., 3) with broadcastable leading dims; returns (..., 3). Ported from mjlab's
    quat_apply_inverse (Isaac-Lab convention)."""
    # Rodrigues-style double-cross form: applies R^T without ever building the 3x3 matrix
    quat = np.asarray(quat, dtype=float)
    vec = np.asarray(vec, dtype=float)
    xyz = quat[..., 1:]
    t = 2.0 * np.cross(xyz, vec)
    return vec - quat[..., 0:1] * t + np.cross(xyz, t)


def projected_gravity(quat, gravity_dir=(0.0, 0.0, -1.0)):
    """Gravity direction expressed in the body frame of `quat` (wxyz).

    This is mjlab's projected_gravity_b: quat_apply_inverse(quat, gravity_dir). A unit
    3-vector that is smooth everywhere on SO(3) -- no gimbal lock, unlike roll/pitch Euler --
    and yaw-invariant: upright -> ~[0,0,-1], inverted (handstand) -> ~[0,0,+1]. `quat` is
    (..., 4); returns (..., 3)."""
    # broadcast the world gravity direction over the batch, then rotate it into the body frame
    quat = np.asarray(quat, dtype=float)
    g = np.broadcast_to(np.asarray(gravity_dir, dtype=float), quat.shape[:-1] + (3,))
    return quat_apply_inverse(quat, g)
