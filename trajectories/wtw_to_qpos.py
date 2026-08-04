##
#
# Convert a flat quadruped motion log (one wide CSV) into a repo clip folder.
#
# Handles the closely-related "walk-these-ways"-style RL rollout logs and STMR retargeting
# outputs: one CSV with a header row and one row per frame, holding the base twist, the 12 leg
# joint angles, and the foot positions. The optimizers here instead read a CLIP FOLDER
#
#     <name>/qpos_<dof>dof.csv    one MuJoCo qpos row per frame
#     <name>/time.csv             per-frame timestamps (these set the frame rate)
#
# (see utils/file_utils.load_reference and trajectories/README.md). This script does that
# conversion: it rebuilds qpos from the logged columns, trims episode resets, and -- because a
# silent frame/sign/order mistake here would quietly poison every downstream solve -- verifies
# the result by forward kinematics against the log's own foot positions.
#
# LOG SCHEMA (columns looked up BY NAME, so column order in the file does not matter)
#   x_pos, y_pos  OR  com_x, com_y    base origin in the world x-y plane [m]. The two producers
#                           name it differently; STMR's "com_" prefix is a misnomer -- the values
#                           are the base/root origin, which is what the e<i> offsets measure from.
#   height                  base height  = world base z [m]
#   quat_x/y/z/w            base orientation, SCALAR-LAST (MuJoCo wants scalar-first wxyz)
#   base<i>/shoulder<i>/elbow<i>, i = 1..4     leg i's hip / thigh / calf angle [rad]
#   e<i>x/y/z               foot i offset from the base origin [m]              (check fallback).
#                           Frame varies by producer -- STMR uses the FULL base frame, wtw the
#                           YAW-ALIGNED heading frame -- so check_fk fits both and reports which.
#   e<i>{x,y,z}_wf          foot i ABSOLUTE position in the world [m]  (preferred check; absent
#                           from STMR logs)
#   vx..wz, com_v*          base twist / velocity command                       (unused: the
#                           velocities are re-derived from qpos by load_reference, so the clip
#                           stays self-consistent with the model's qvel convention)
#
# The leg index i = 1..4 is FL, FR, RL, RR -- the Go2 XML's joint order -- so the 12 joint
# columns drop straight into qpos[7:19]. That IS the assumption the FK check validates.
#
# NOTE ON FOOT HEIGHT: the e<i> foot reference point is the MJCF foot SITE, which sits at the
# centre of the 22 mm collision sphere. A producer that retargets onto a POINT foot therefore
# leaves the sphere buried by up to its radius; --pz-offset lifts the clip to compensate (the
# script measures and suggests the value).
#
# Usage
#   python trajectories/wtw_to_qpos.py go2/wtw_pronking.csv          # -> go2/wtw_pronking/
#   python trajectories/wtw_to_qpos.py go2/hopturn_go2_STMR_resampled_50Hz.csv --pz-offset 0.0193
#
##

# directory imports
import sys
import os

ROOT = os.getenv("TRAJOPT_ROOT_DIR")
sys.path.append(ROOT)

# standard imports
import argparse
import csv

import numpy as np

# custom imports
import mujoco


########################################################################
# LOG SCHEMA
########################################################################

# leg i (1-based, as named in the log) -> the model's leg prefix, in MuJoCo joint order
LEGS = ("FL", "FR", "RL", "RR")

# per-leg joint columns, in the model's within-leg order (hip, thigh, calf)
LEG_JOINT_COLS = ("base", "shoulder", "elbow")

DEFAULT_MODEL = ("models", "unitree_go2", "go2.xml")
DEFAULT_DT = 0.02                     # both producers here run at 50 Hz

# The base x-y column pair, in preference order -- the producers name it differently but mean the
# same thing (the base/root origin in the world plane). See the header note on "com_".
BASE_XY_ALIASES = (("x_pos", "y_pos"), ("com_x", "com_y"))


def joint_columns():
    """The 12 leg-angle column names in MuJoCo joint order: leg 1..4 x (hip, thigh, calf)."""
    return [f"{j}{i}" for i in range(1, len(LEGS) + 1) for j in LEG_JOINT_COLS]


def resolve_base_xy(col):
    """Return the (x, y) column names holding the base position, from BASE_XY_ALIASES."""
    for pair in BASE_XY_ALIASES:
        if all(n in col for n in pair):
            return pair
    raise ValueError("no base x-y columns found; expected one of "
                     + " or ".join("/".join(p) for p in BASE_XY_ALIASES))


def has_world_feet(col):
    """True if the log carries the absolute world foot positions (e<i>{x,y,z}_wf)."""
    return all(f"e{i}{a}_wf" in col for i in range(1, len(LEGS) + 1) for a in "xyz")


########################################################################
# LOAD
########################################################################

def read_log(csv_path):
    """Read the log into (D (N, ncols), col) where col maps a column name -> its index."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{csv_path}: need a header row plus at least one data row")
    header = [h.strip() for h in rows[0]]
    D = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=float)
    col = {name: i for i, name in enumerate(header)}
    resolve_base_xy(col)                                   # raises with a clear message if absent
    missing = [n for n in ["height", "quat_w", "quat_x", "quat_y", "quat_z"]
               + joint_columns() if n not in col]
    if missing:
        raise ValueError(f"{csv_path}: missing expected columns: {', '.join(missing)}")
    return D, col


def longest_run(D, col, jump_tol=10.0):
    """Trim episode resets: return the slice of the longest contiguous segment of D.

    An RL log often ends (or is stitched) at an environment reset, where the base position
    teleports back to the origin. Left in, that one-frame jump becomes a huge bogus velocity
    when load_reference finite-differences the clip. Frames are split wherever the base-position
    step exceeds `jump_tol` x the median step, and the longest run is kept.
    """
    xc, yc = resolve_base_xy(col)
    P = np.stack([D[:, col[xc]], D[:, col[yc]], D[:, col["height"]]], axis=1)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)                  # (N-1,)
    if len(step) == 0:
        return slice(0, len(D))
    med = float(np.median(step))
    breaks = np.flatnonzero(step > jump_tol * max(med, 1e-9)) + 1       # first frame after a jump
    edges = np.concatenate([[0], breaks, [len(D)]])
    lengths = np.diff(edges)
    b = int(np.argmax(lengths))
    return slice(int(edges[b]), int(edges[b + 1]))


########################################################################
# CONVERT
########################################################################

def build_qpos(D, col, nq):
    """Assemble the MuJoCo qpos rows (N, nq) from the log's columns.

    qpos = [x, y, z | quat wxyz | 12 joint angles]: the base position comes from
    (x_pos, y_pos, height); the quaternion is reordered from the log's scalar-LAST xyzw to
    MuJoCo's scalar-first wxyz and renormalized; the joint block is the 12 leg columns in
    MuJoCo joint order (see joint_columns()).
    """
    xc, yc = resolve_base_xy(col)
    N = len(D)
    Q = np.zeros((N, nq))
    Q[:, 0] = D[:, col[xc]]
    Q[:, 1] = D[:, col[yc]]
    Q[:, 2] = D[:, col["height"]]
    quat = np.stack([D[:, col["quat_w"]], D[:, col["quat_x"]],
                     D[:, col["quat_y"]], D[:, col["quat_z"]]], axis=1)     # -> wxyz
    n = np.linalg.norm(quat, axis=1, keepdims=True)
    Q[:, 3:7] = quat / np.where(n > 0.0, n, 1.0)
    Q[:, 7:] = np.stack([D[:, col[c]] for c in joint_columns()], axis=1)
    return Q


def yaw_of(quat):
    """Heading (yaw) angle [rad] of a wxyz quaternion."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_mat(yaw):
    """Rotation about the world z-axis by `yaw` (world <- heading frame)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_mat(quat):
    """3x3 rotation matrix of a wxyz quaternion (world <- body frame)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def rezero(Q):
    """Put the clip's first frame at the world origin with zero yaw, in place.

    Rotates the whole clip about the world z-axis by -yaw_0 and shifts x-y so frame 0 sits at
    (0, 0) -- the pose relative to the ground is untouched, so contacts and joint angles are
    unchanged. Useful when a log starts mid-episode at an arbitrary place and heading.
    """
    q0 = Q[0, 3:7]
    yaw0 = np.arctan2(2.0 * (q0[0] * q0[3] + q0[1] * q0[2]),
                      1.0 - 2.0 * (q0[2] ** 2 + q0[3] ** 2))
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    xy = Q[:, :2] - Q[0, :2]
    Q[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    Q[:, 1] = s * xy[:, 0] + c * xy[:, 1]
    qz = np.array([np.cos(-yaw0 / 2.0), 0.0, 0.0, np.sin(-yaw0 / 2.0)])     # wxyz, about world z
    for k in range(len(Q)):
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qz, Q[k, 3:7])
        Q[k, 3:7] = out
    return Q


########################################################################
# VALIDATION
########################################################################

def check_fk(model, Q, D, col, stride=1):
    """FK the rebuilt qpos every stride-th frame and compare each foot site to the log's own foot
    channels -- quaternion convention, leg order and joint signs all feed foot position, so a
    mapping bug shows up as metres of error. Channels, in order of preference:
        e<i>{x,y,z}_wf   ABSOLUTE world foot position; also pins base x-y. wtw logs only.
        e<i>{x,y,z}      offset FROM THE BASE ORIGIN; blind to base x-y. Frame differs by
                         producer (STMR full base, wtw yaw-only), so both are fitted.
    The residual is taken in a body-fixed frame and split: `offset` is the per-leg MEAN (a
    constant per-leg shift is benign -- just different fixed link offsets in the producer's
    model), `scatter` is the spread about it (THE mapping-error signal; sub-mm means exact).
    Returns (mode, offset (4,3), scatter (n,4)) in metres, or (None,)*3 if nothing to check."""
    data = mujoco.MjData(model)
    sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{leg}") for leg in LEGS]
    if any(s < 0 for s in sids):
        return None, None, None                            # model has no foot sites -> skip
    have_rel = all(f"e{i}{a}" in col for i in range(1, len(LEGS) + 1) for a in "xyz")
    if not (has_world_feet(col) or have_rel):
        return None, None, None                            # nothing to check against

    ks = list(range(0, len(Q), max(1, int(stride))))
    P = np.zeros((len(ks), len(LEGS), 3))                  # FK foot positions, world
    for n, k in enumerate(ks):
        data.qpos[:] = Q[k]
        mujoco.mj_kinematics(model, data)
        P[n] = [data.site_xpos[s] for s in sids]

    def split(res):
        """(offset, scatter) from a per-frame per-leg residual (n, 4, 3)."""
        offset = res.mean(axis=0)
        return offset, np.linalg.norm(res - offset, axis=2)

    if has_world_feet(col):
        E = np.stack([np.stack([D[ks, col[f"e{i+1}{a}_wf"]] for a in "xyz"], axis=1)
                      for i in range(len(LEGS))], axis=1)          # (n, 4, 3) world
        res = np.stack([yaw_mat(yaw_of(Q[k, 3:7])).T @ (P[n] - E[n]).T
                        for n, k in enumerate(ks)]).transpose(0, 2, 1)
        offset, scatter = split(res)
        return "world foot position (e<i>_wf), residual in the heading frame", offset, scatter

    # relative channel: fit both candidate frames and keep the better one
    E = np.stack([np.stack([D[ks, col[f"e{i+1}{a}"]] for a in "xyz"], axis=1)
                  for i in range(len(LEGS))], axis=1)              # (n, 4, 3) base-relative
    best = None
    for label, rot in (("full base frame", lambda q: quat_mat(q)),
                       ("heading frame", lambda q: yaw_mat(yaw_of(q)))):
        res = np.stack([rot(Q[k, 3:7]).T @ (P[n] - Q[k, :3]).T
                        for n, k in enumerate(ks)]).transpose(0, 2, 1) - E
        offset, scatter = split(res)
        if best is None or scatter.mean() < best[2].mean():
            best = (f"foot offset from the base (e<i>), {label}", offset, scatter)
    return best


def check_ground(model, Q, stride=1):
    """Measure how the clip sits on the floor, and what --pz-offset would put it there. The foot
    reference point is the MJCF foot SITE, at the centre of the collision SPHERE, so a grounded
    foot sits at z = +radius rather than 0. A producer that retargeted onto a POINT foot leaves
    the site at z ~ 0, burying the sphere and handing the optimizer a reference it can only track
    by penetrating the floor. Stance height is the MEDIAN over grounded frames, which is robust
    to flight phases and to the deepest landing frames.
    Returns (stance_site_z, radius, suggested_pz_offset), or None with no foot spheres/sites."""
    data = mujoco.MjData(model)
    sids, rads = [], []
    for leg in LEGS:
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{leg}")
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{leg}_foot_collision")
        if s < 0 or g < 0:
            return None
        sids.append(s)
        rads.append(float(model.geom_size[g, 0]))
    radius = float(np.mean(rads))
    lo = []
    for k in range(0, len(Q), max(1, int(stride))):
        data.qpos[:] = Q[k]
        mujoco.mj_kinematics(model, data)
        lo.append(min(data.site_xpos[s][2] for s in sids))
    lo = np.asarray(lo)
    grounded = lo[lo < radius + 0.01]                  # frames with a foot at/near the floor
    stance_z = float(np.median(grounded)) if len(grounded) else float(lo.min())
    return stance_z, radius, radius - stance_z


def check_joint_limits(model, Q):
    """Report any joint angle in the clip that falls outside the model's joint range.

    The tracking MPC can run in position-servo mode, where the reference angles are commanded
    straight through as control targets and get clipped to ctrlrange -- so a reference that
    leaves the joint range would be silently unreachable. Returns a list of message strings.
    """
    msgs = []
    for j in range(model.njnt):
        if not model.jnt_limited[j] or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        a = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        q = Q[:, a]
        if q.min() < lo or q.max() > hi:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            msgs.append(f"  {name}: clip [{q.min():+.4f}, {q.max():+.4f}] "
                        f"outside range [{lo:+.4f}, {hi:+.4f}]")
    return msgs


def report_dt(D, col, dt):
    """Cross-check the assumed frame period against the log's own velocity channels.

    The log carries no timestamps, so `dt` is an input. But it also logs the base twist, and the
    planar path length must equal the mean planar speed x duration -- so the ratio below is ~1
    only when `dt` is right (it scales as 1/dt, e.g. 2.0 means the true rate is twice `dt`).
    """
    xc, yc = resolve_base_xy(col)
    P = np.stack([D[:, col[xc]], D[:, col[yc]]], axis=1)
    path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if not {"vx", "vy"} <= set(col):
        return None
    speed = float(np.linalg.norm(np.stack([D[:, col["vx"]], D[:, col["vy"]]], 1), axis=1).mean())
    if speed <= 0.0 or path < 0.25 * (len(D) - 1) * dt * speed:
        # Degenerate for a near-in-place motion (a hop-turn barely translates), where the ratio
        # measures noise rather than the frame rate. report_vz below is the check that works there.
        return None
    return path / (speed * (len(D) - 1) * dt)


def report_vz(D, col, dt):
    """Cross-check `dt` against the logged VERTICAL velocity: fit vz to d(height)/dt.

    Works where report_dt cannot -- a hop or an in-place turn hardly translates, so the planar
    path-length test degenerates, but the base bobs strongly, so vz vs the finite-differenced
    height is a sharp test. The returned slope scales as 1/dt and is 1.0 when `dt` is right.
    """
    if "vz" not in col:
        return None
    vz = D[:, col["vz"]]
    if len(D) < 3 or np.allclose(vz, 0.0):
        return None
    fd = np.gradient(D[:, col["height"]], dt)
    if not np.any(np.abs(vz) > 1e-9):
        return None
    return float(np.polyfit(vz, fd, 1)[0])


########################################################################
#  MAIN
########################################################################

def rel(path):
    """Path relative to ROOT for printing, or the absolute path when it lies outside the repo
    (so an out-of-tree --out does not print as a wall of '../')."""
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(os.path.abspath(ROOT) + os.sep) else path


def resolve_csv(name):
    """Resolve a log path: as given, else under trajectories/, adding .csv if needed."""
    for cand in (name, name + ".csv",
                 os.path.join(ROOT, "trajectories", name),
                 os.path.join(ROOT, "trajectories", name + ".csv")):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(f"log not found: {name}")


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Convert a walk-these-ways-style quadruped log into a repo clip folder "
                    "(qpos_<dof>dof.csv + time.csv).")
    p.add_argument("log", help="the log CSV (a path, or a name under trajectories/, "
                               "e.g. go2/wtw_pronking)")
    p.add_argument("-o", "--out", default=None,
                   help="output clip folder (default: <log dir>/<log stem>/)")
    p.add_argument("--dt", type=float, default=DEFAULT_DT,
                   help=f"frame period [s] -- the log has no timestamps (default {DEFAULT_DT}, "
                        f"i.e. 50 Hz); cross-checked against the logged base velocity")
    p.add_argument("--model", default=None,
                   help="MuJoCo XML defining the joint order / foot sites for the FK check "
                        "(default models/unitree_go2/go2.xml)")
    p.add_argument("--rezero", action="store_true",
                   help="move the first frame to the origin with zero yaw (pose vs the ground "
                        "is unchanged)")
    p.add_argument("--pz-offset", type=float, default=0.0,
                   help="constant shift of the base height [m] (raises the whole clip)")
    p.add_argument("--crop", default=None, metavar="START:STOP",
                   help="keep only frames [START:STOP) of the trimmed clip")
    p.add_argument("--no-check", action="store_true", help="skip the FK validation")
    p.add_argument("--check-stride", type=int, default=1,
                   help="FK-check every Nth frame (default 1 = all)")
    args = p.parse_args()

    if args.dt <= 0.0:
        raise ValueError(f"--dt must be positive, got {args.dt}")

    # load the log and trim episode resets
    csv_path = resolve_csv(args.log)
    D_all, col = read_log(csv_path)
    keep = longest_run(D_all, col)
    D = D_all[keep]
    print(f"log     : {rel(csv_path)}  ({len(D_all)} frames)")
    if len(D) != len(D_all):
        print(f"          trimmed to frames [{keep.start}:{keep.stop}] ({len(D)}), "
              f"dropping {len(D_all) - len(D)} across an episode reset")

    # optional crop, applied after the reset trim
    if args.crop:
        a, _, b = args.crop.partition(":")
        sl = slice(int(a) if a else None, int(b) if b else None)
        D = D[sl]
        print(f"          cropped to {len(D)} frames ({args.crop})")
    if len(D) < 2:
        raise ValueError("need at least 2 frames after trimming/cropping")

    # frame rate: two cross-checks, each reading 1.0 when dt is right --
    # path-length vs speed (needs travel), else vz vs d(height)/dt (works for hops / turns)
    ratio = report_dt(D, col, args.dt)
    label, ratio = ("path/(speed*T)", ratio) if ratio is not None \
        else ("slope(dz/dt vs vz)", report_vz(D, col, args.dt))
    head = f"dt      : {args.dt * 1e3:.2f} ms ({1.0 / args.dt:.1f} Hz), {(len(D) - 1) * args.dt:.2f} s"
    if ratio is None:
        print(f"{head} (not cross-checked)")
    else:
        note = "ok" if abs(ratio - 1.0) < 0.05 else f"SUSPECT -- try --dt {args.dt * ratio:.4f}"
        print(f"{head} | {label} = {ratio:.3f} [{note}]")

    # rebuild qpos
    model_parts = tuple(args.model.split(os.sep)) if args.model else DEFAULT_MODEL
    model_path = args.model if (args.model and os.path.isfile(args.model)) \
        else os.path.join(ROOT, *model_parts)
    model = mujoco.MjModel.from_xml_path(model_path)
    n_joints = model.nq - 7
    if n_joints != 3 * len(LEGS):
        raise ValueError(f"{model_path}: expected {3 * len(LEGS)} joints (nq={7 + 3 * len(LEGS)}), "
                         f"got {n_joints} (nq={model.nq})")
    Q = build_qpos(D, col, model.nq)
    print(f"model   : {rel(model_path)}  (nq={model.nq}, {n_joints} joints)")

    # validate BEFORE repositioning: check_fk needs the LOG's frame, since --rezero and
    # --pz-offset move the whole robot and would read as a bogus offset --
    check_lines = []
    if not args.no_check:
        mode, offset, scatter = check_fk(model, Q, D, col, stride=args.check_stride)
        if offset is None:
            print("FK check: skipped (no foot sites on the model, or no foot columns in the log)")
        else:
            check_lines.append(f"FK check: {mode}")
            check_lines.append(f"  scatter (mapping error)  mean {scatter.mean()*1e3:6.2f} mm, "
                               f"max {scatter.max()*1e3:6.2f} mm")
            for i, leg in enumerate(LEGS):
                check_lines.append(
                    f"  {leg} constant offset  "
                    f"[{offset[i,0]*1e3:+7.2f} {offset[i,1]*1e3:+7.2f} {offset[i,2]*1e3:+7.2f}] mm"
                    f"  (|.| = {np.linalg.norm(offset[i])*1e3:5.2f} mm)")
            print("\n".join(check_lines))
            if scatter.max() > 0.01:
                print("          WARNING: scatter > 10 mm -- the column mapping (leg order, joint "
                      "signs, quaternion convention) is probably wrong")
            elif np.linalg.norm(offset, axis=1).max() > 0.03:
                print("          NOTE: the constant offsets are large; the log's kinematic model "
                      "differs from this MJCF's fixed link offsets (benign, but the reference "
                      "foot placement is shifted by that much)")
    lim = check_joint_limits(model, Q)
    if lim:
        print("limits  : WARNING, reference joint angles leave the model's joint ranges:")
        print("\n".join(lim))

    # ground clearance: is the clip standing ON the floor, or sunk into it?
    ground = check_ground(model, Q, stride=args.check_stride)
    if ground is not None:
        stance_z, radius, suggest = ground
        print(f"ground  : foot sphere radius {radius*1e3:.1f} mm; stance foot centre sits at "
              f"{stance_z*1e3:+.1f} mm")
        if abs(suggest - args.pz_offset) > 0.002:
            print(f"          the sphere rests on the floor when the centre is at "
                  f"+{radius*1e3:.1f} mm -> consider --pz-offset {suggest:.4f}"
                  f"{f' (currently {args.pz_offset})' if args.pz_offset else ''}")
        else:
            print(f"          --pz-offset {args.pz_offset} puts the stance sphere on the floor")

    # reposition the validated clip (rigid moves; joint angles untouched)
    if args.rezero:
        Q = rezero(Q)
    Q[:, 2] += args.pz_offset

    # write the clip folder
    out_dir = args.out or os.path.join(os.path.dirname(csv_path),
                                       os.path.splitext(os.path.basename(csv_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    qpos_path = os.path.join(out_dir, f"qpos_{n_joints}dof.csv")
    time_path = os.path.join(out_dir, "time.csv")
    np.savetxt(qpos_path, Q, delimiter=",")
    np.savetxt(time_path, args.dt * np.arange(len(Q)), delimiter=",")
    with open(os.path.join(out_dir, "description.txt"), "w") as f:
        f.write(f"converted from {rel(csv_path)} by trajectories/wtw_to_qpos.py\n"
                f"frames  : {len(Q)} at dt = {args.dt} s ({(len(Q) - 1) * args.dt:.3f} s)\n"
                f"model   : {rel(model_path)} (nq = {model.nq})\n"
                f"joints  : {', '.join(joint_columns())}\n"
                f"          -> {', '.join(f'{l}_{j}' for l in LEGS for j in ('hip', 'thigh', 'calf'))}\n"
                f"rezero  : {args.rezero}\npz_offset: {args.pz_offset}\n"
                + ("\n".join(check_lines) + "\n" if check_lines else ""))
    print(f"saved   : {rel(qpos_path)}  ({Q.shape[0]} x {Q.shape[1]})")
    print(f"          {rel(time_path)}")
