##
#
# Manipulate G1 reference trajectories: in-place center/yaw/retime (keeping _og backups), or
# derive a sagittal MIRROR into a new folder (reverses turn direction).
#
##

# standard imports
import os
import sys
import re
import glob
import shutil
import argparse
import numpy as np

# mujoco imports
import mujoco

# repo root
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("TRAJOPT_ROOT_DIR") or os.path.dirname(HERE)
sys.path.append(ROOT)

# local imports
from utils.math_utils import quat_mul


########################################################################
# MOTION RESOLUTION / IO
########################################################################

def find_motion_dir(motion):
    """Resolve a motion to its folder: absolute path, a <robot>/<group>/<motion> subpath, or a
    bare name searched anywhere under trajectories/ (must be unique)."""
    motion_path = motion if os.path.isabs(motion) else os.path.join(HERE, motion)
    if not os.path.isdir(motion_path):
        matches = sorted(glob.glob(os.path.join(HERE, "**", os.path.basename(motion)),
                                   recursive=True))
        matches = [m for m in matches if os.path.isdir(m)]
        if len(matches) == 1:
            motion_path = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"'{motion}' is ambiguous, found {len(matches)}: "
                + ", ".join(os.path.relpath(m, HERE) for m in matches)
                + " -- disambiguate with <robot>/<group>/<motion>")
        else:
            raise NotADirectoryError(f"no motion folder: {motion_path}")
    return motion_path


def available_dofs(motion_dir):
    """Sorted list of dofs that have a qpos_<dof>dof.csv in `motion_dir`."""
    dofs = []
    for p in glob.glob(os.path.join(motion_dir, "qpos_*dof.csv")):
        m = re.match(r"qpos_(\d+)dof\.csv$", os.path.basename(p))
        if m:
            dofs.append(int(m.group(1)))
    return sorted(dofs)


def load_trajectory(csv_path):
    """Load an (N x nq) qpos CSV; base quaternion assumed already in MuJoCo wxyz order."""
    Q = np.loadtxt(csv_path, delimiter=",")
    if Q.ndim == 1:
        Q = Q[None, :]
    return Q


########################################################################
# TRANSFORMS  (free-joint base: cols 0:3 = pos xyz, cols 3:7 = quat wxyz)
########################################################################

def center_xy(Q):
    """Offset base x,y so the FIRST frame sits at the world origin. z / orientation / joints
    are untouched, so the forward stride and pose are preserved."""
    Q = Q.copy()
    Q[:, 0] -= Q[0, 0]
    Q[:, 1] -= Q[0, 1]
    return Q


def yaw_world_z(Q, deg):
    """Rotate the whole trajectory about the world +Z axis (through the origin) by `deg` degrees:
    rotate the base x,y position and left-multiply the base quaternion by the yaw rotation (a
    world-frame rotation). Base z and joint angles are untouched."""
    Q = Q.copy()
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    x, y = Q[:, 0].copy(), Q[:, 1].copy()
    Q[:, 0] = c * x - s * y                       # rotate base position about world +Z (CCW)
    Q[:, 1] = s * x + c * y
    q_yaw = np.array([np.cos(th / 2.0), 0.0, 0.0, np.sin(th / 2.0)])   # wxyz, about +Z
    Q[:, 3:7] = quat_mul(q_yaw[None, :], Q[:, 3:7])                    # world-frame (left) rotation
    return Q


_ALIGN_DEG = {"+x": 0.0, "x": 0.0, "-x": 180.0, "+y": 90.0, "y": 90.0, "-y": -90.0}


def base_heading_deg(q):
    """Start heading [deg] from the base wxyz quaternion: azimuth of the body z-axis ground projection
    (matches g1_gait._heading; robust to the crawl's ~90deg base pitch)."""
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2.0 * (y * z - w * x), 2.0 * (x * z + w * y))))


def resolve_align_target(spec):
    """--align DIR -> target world heading [deg]: +x|-x|+y|-y (where the base z-axis should point) or a
    raw angle in degrees."""
    key = str(spec).strip().lower()
    if key in _ALIGN_DEG:
        return _ALIGN_DEG[key]
    try:
        return float(spec)
    except ValueError:
        raise ValueError(f"--align must be +x/-x/+y/-y or an angle [deg], got {spec!r}")


########################################################################
# MIRROR  (new clip: sagittal x-z reflection, reverses turn direction)
########################################################################

# Reflecting in the plane the robot faces turns +wz into -wz (M Rz M = Rz(-theta), M=diag(1,-1,1)),
# swaps L/R limbs and flips roll/yaw signs -- the valid mirror gait of the bilaterally symmetric G1.

# Only qpos is mirrored; velocities are FD-recomputed downstream so they come out mirrored for free.
# The joint map is derived FROM THE MODEL, so it tracks joint-set changes.

def mirrored_dir_name(motion_dir):
    """<...>_periodic -> <...>_mirrored_periodic ; otherwise append _mirrored."""
    base = os.path.basename(motion_dir)
    if base.endswith("_periodic"):
        base = base[: -len("_periodic")] + "_mirrored_periodic"
    else:
        base = base + "_mirrored"
    return os.path.join(os.path.dirname(motion_dir), base)


def mirror_partner_name(name):
    """Swap left<->right in a joint/body name (names carry exactly one of the two, or neither)."""
    if "left" in name:
        return name.replace("left", "right")
    if "right" in name:
        return name.replace("right", "left")
    return name


def build_qpos_mirror(model):
    """(src_col, sign) over the full qpos so  qpos_mirrored[:, c] = sign[c] * qpos[:, src_col[c]].
    Base: pos y and quat x,z negate; hinges map to their L<->R partner, sign + for pitch(y) else -."""
    src = np.arange(model.nq)
    sign = np.ones(model.nq)
    sign[1] = sign[4] = sign[6] = -1.0            # base y ; quat x,z -> (w,-x,y,-z)
    qadr = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            qadr[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = int(model.jnt_qposadr[j])
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        partner = mirror_partner_name(nm)
        if partner not in qadr:
            raise KeyError(f"no mirror partner '{partner}' for joint '{nm}'")
        src[qadr[nm]] = qadr[partner]
        sign[qadr[nm]] = 1.0 if abs(model.jnt_axis[j][1]) > 0.5 else -1.0   # pitch(y) kept; roll/yaw flipped
    return src, sign


def mirror_qpos(Q, src, sign):
    """Apply the column map: (T, nq) -> (T, nq)."""
    return Q[:, src] * sign[None, :]


def _heading_angle(model, q):
    """Heading = atan2 of the base z-axis horizontal projection (robust to the ~90 deg base pitch)."""
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    zc = data.xmat[1].reshape(3, 3)[:, 2]         # world z-axis of the base body
    return float(np.arctan2(zc[1], zc[0]))


def verify_mirror(model, Q, Qm):
    """FK cross-check (mirrored body == reflected partner, mid-clip) + net-heading sign flip."""
    M = np.diag([1.0, -1.0, 1.0])
    k = Q.shape[0] // 2
    d0, dm = mujoco.MjData(model), mujoco.MjData(model)
    d0.qpos[:] = Q[k]; mujoco.mj_forward(model, d0)
    dm.qpos[:] = Qm[k]; mujoco.mj_forward(model, dm)
    worst = 0.0
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        pb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mirror_partner_name(nm))
        if pb >= 0:
            worst = max(worst, float(np.linalg.norm(dm.xpos[b] - M @ d0.xpos[pb])))
    h0 = _heading_angle(model, Q[-1]) - _heading_angle(model, Q[0])
    hm = _heading_angle(model, Qm[-1]) - _heading_angle(model, Qm[0])
    return worst, h0, hm


def mirror_motion(motion_dir):
    """Derive the sagittal mirror into a NEW folder from the CURRENT qpos/time (so it inherits any
    centering/retiming already applied); the source is left untouched. Returns (out_dir, report)."""
    dofs = available_dofs(motion_dir)
    if not dofs:
        raise FileNotFoundError(f"no qpos_<dof>dof.csv in {motion_dir}")
    out_dir = mirrored_dir_name(motion_dir)
    os.makedirs(out_dir, exist_ok=True)
    for tf in ("time.csv", "time_og.csv"):                    # timing unchanged -> copy verbatim
        src_t = os.path.join(motion_dir, tf)
        if os.path.exists(src_t):
            shutil.copy2(src_t, os.path.join(out_dir, tf))
    report = {}
    for dof in dofs:
        model = mujoco.MjModel.from_xml_path(
            os.path.join(ROOT, "models", "unitree_g1", f"g1_{dof}dof.xml"))
        Q = load_trajectory(os.path.join(motion_dir, f"qpos_{dof}dof.csv"))
        if Q.shape[1] != model.nq:
            raise ValueError(f"{dof}dof csv has {Q.shape[1]} cols but model nq={model.nq}")
        src, sign = build_qpos_mirror(model)
        Qm = mirror_qpos(Q, src, sign)
        np.savetxt(os.path.join(out_dir, f"qpos_{dof}dof.csv"), Qm, delimiter=",")
        report[dof] = (Q.shape[0],) + verify_mirror(model, Q, Qm)
    return out_dir, report


########################################################################
# RETIMING  (time.csv only; qpos frames untouched)
########################################################################

def load_time_from_backup(motion_dir):
    """Return (t_og, time_csv_path). Backs up time.csv -> time_og.csv once, then always reads the
    _og backup so re-runs re-derive from the original timing (retiming never compounds)."""
    csv = os.path.join(motion_dir, "time.csv")
    og = os.path.join(motion_dir, "time_og.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"no time.csv in {motion_dir}; cannot retime")
    if not os.path.exists(og):
        shutil.copy(csv, og)                          # preserve the true original on first run
    t = np.loadtxt(og, delimiter=",").astype(float).ravel()
    if t.size < 2:
        raise ValueError(f"time.csv in {motion_dir} has < 2 samples; cannot retime")
    return t, csv


def resolve_target_period(spec, spans):
    """Resolve the --period argument to a target span [s]: a number, or one of {mean,max,min}
    aggregated over the given clips' original spans."""
    spans = np.asarray(spans, dtype=float)
    key = str(spec).strip().lower()
    if key in ("mean", "avg", "average"):
        return float(spans.mean())
    if key == "max":
        return float(spans.max())
    if key == "min":
        return float(spans.min())
    try:
        T = float(spec)
    except ValueError:
        raise ValueError(f"--period must be a number or one of mean/max/min, got {spec!r}")
    if T <= 0:
        raise ValueError(f"--period must be positive, got {T}")
    return T


def retime_span(t, T_new):
    """Uniformly rescale timestamps so the total span (t[-1]-t[0]) becomes T_new, preserving t[0]
    and the relative spacing (uniform stays uniform). qpos is untouched."""
    t0 = float(t[0])
    span = float(t[-1] - t0)
    if span <= 0:
        raise ValueError("time.csv has non-positive span; cannot retime")
    return t0 + (t - t0) * (float(T_new) / span)


########################################################################
# IN-PLACE QPOS TRANSFORM
########################################################################

def transform_qpos(motion_dir, yaw, align=None):
    """Center (always) + optional align/yaw the qpos of one motion, in place, from the _og backup.
    Returns the sorted dof list touched."""
    dofs = available_dofs(motion_dir)
    if not dofs:
        raise FileNotFoundError(f"no qpos_<dof>dof.csv in {motion_dir}")
    target = resolve_align_target(align) if align is not None else None   # validate once, up front

    # per dof: back the original up once, then always re-derive FROM that backup, so repeated
    # runs never compound and never clobber the true original
    for dof in dofs:
        csv = os.path.join(motion_dir, f"qpos_{dof}dof.csv")
        og = os.path.join(motion_dir, f"qpos_{dof}dof_og.csv")
        if not os.path.exists(og):
            shutil.copy(csv, og)                  # preserve the true original on first run
        Q = load_trajectory(og)                   # always transform from the original
        Q = center_xy(Q)                          # always re-center base x,y to the origin
        if target is not None:                    # yaw so the START heading faces `align` in world
            Q = yaw_world_z(Q, target - base_heading_deg(Q[0, 3:7]))
        if yaw is not None:
            Q = yaw_world_z(Q, yaw)
        np.savetxt(csv, Q, delimiter=",")         # replace qpos_<dof>dof.csv in place
    return dofs


########################################################################
#  MAIN
########################################################################

def main():
    """Edit reference clips in place, or derive a mirrored copy. Keeps _og backups throughout.
        motion     one or more clip folders (bare name, group/motion subpath, or abs path)
        --yaw      rotate about world +Z by DEG, after the automatic re-centering
        --align    yaw so the START heading faces +x|-x|+y|-y (or a raw angle)
        --period   rescale each time.csv span to T seconds, or mean|max|min over the given clips
        --mirror   derive a sagittal mirror into <...>_mirrored[_periodic]/ (standalone)
    qpos ops always re-derive from qpos_<dof>dof_og.csv, so re-runs never compound. --mirror
    writes a NEW folder and cannot be combined with the in-place ops."""

    # command line
    p = argparse.ArgumentParser(
        description="Manipulate G1 reference trajectories: center/yaw qpos, retime periodic clips to "
                    "a common period, or derive a sagittal mirror (reverses turn). Keeps _og backups.")
    p.add_argument("motion", nargs="+",
                   help="one or more trajectory folders (bare name, group/motion subpath, or abs path)")
    p.add_argument("--yaw", type=float, default=None, metavar="DEG",
                   help="rotate the whole trajectory about the world +Z axis by DEG degrees "
                        "(applied after the automatic re-centering)")
    p.add_argument("--align", default=None, metavar="DIR",
                   help="yaw the clip so its START heading faces DIR in world: +x|-x|+y|-y or an angle "
                        "[deg] (after re-centering; heading = base body-z ground projection). Bakes the "
                        "yaw-alignment into the reference so the optimizer reads h_ref=0.")
    p.add_argument("--period", nargs="?", const="mean", default=None, metavar="T",
                   help="retime each clip's time.csv so its span (t_last - t_0) becomes T seconds: "
                        "a number, or one of mean/max/min over the given clips (bare --period = mean). "
                        "qpos is left untouched unless --yaw is also given.")
    p.add_argument("--mirror", action="store_true",
                   help="derive the sagittal MIRROR of each clip (reverses turn direction) into a new "
                        "<...>_mirrored_periodic folder, from the CURRENT qpos/time. Standalone: not "
                        "combinable with the in-place --yaw/--period (run those first, then --mirror).")
    args = p.parse_args()

    motion_dirs = [find_motion_dir(m) for m in args.motion]

    # mirror mode: derive NEW clips, then stop (distinct output model from the in-place ops)
    if args.mirror:
        if args.yaw is not None or args.period is not None or args.align is not None:
            p.error("--mirror derives a NEW clip and can't be combined with in-place --align/--yaw/--period; "
                    "run those first, then --mirror.")
        out0 = None
        for md in motion_dirs:
            out_dir, report = mirror_motion(md)
            out0 = out0 or out_dir
            print(f"mirror : {os.path.basename(md)}  ->  {os.path.basename(out_dir)}")
            for dof, (n, worst, h0, hm) in report.items():
                print(f"  {dof}dof : {n} frames | FK mirror residual={worst:.2e} m | "
                      f"net heading orig={np.degrees(h0):+.1f} -> mirrored={np.degrees(hm):+.1f} deg")
                if worst > 1e-3:
                    print("    WARNING: large FK mirror residual -- joint map may be wrong for this model.")
        print(f"replay : python trajectories/replay.py {os.path.basename(out0)} --axis")
        return

    # qpos runs on the classic path (no --period) or whenever a qpos op (--yaw) is requested. With
    # --period ALONE, qpos is left untouched so prior per-clip yaws survive a batch retime.
    do_qpos = (args.period is None) or (args.yaw is not None) or (args.align is not None)
    do_retime = args.period is not None

    # retime: resolve ONE common target span from all clips, then rescale each time.csv
    retime_by_dir = {}
    T = None
    if do_retime:
        times = [load_time_from_backup(md) for md in motion_dirs]      # [(t_og, csv_path), ...]
        spans = [float(t[-1] - t[0]) for (t, _) in times]
        T = resolve_target_period(args.period, spans)
        for md, (t, csv), span in zip(motion_dirs, times, spans):
            new_t = retime_span(t, T)
            np.savetxt(csv, new_t)                                     # 1-col %.18e, matches time.csv
            retime_by_dir[md] = (span, float(new_t[-1] - new_t[0]), t.size)

    # qpos: center (always) + optional yaw, per motion, from the _og backup
    qpos_by_dir = {md: transform_qpos(md, args.yaw, args.align) for md in motion_dirs} if do_qpos else {}

    # report
    qops = []
    if do_qpos:
        qops.append("center x,y -> origin")
        if args.align is not None:
            qops.append(f"align start heading -> {args.align}")
        if args.yaw is not None:
            qops.append(f"yaw {args.yaw:g} deg about world +Z")
    for md in motion_dirs:
        name = os.path.basename(md)
        print(f"motion : {name}")
        if do_qpos:
            dofs = qpos_by_dir[md]
            print(f"  qpos   : {', '.join(qops)}  ->  qpos_<dof>dof.csv  (og: qpos_<dof>dof_og.csv)  "
                  f"[dofs {', '.join(map(str, dofs))}]")
        if do_retime:
            old_span, new_span, n = retime_by_dir[md]
            print(f"  retime : span {old_span:.4f}s -> {new_span:.4f}s  "
                  f"(loop {n * old_span / (n - 1):.4f}s -> {n * new_span / (n - 1):.4f}s, "
                  f"{n} frames)  ->  time.csv  (og: time_og.csv)")
    if do_retime:
        print(f"target period: {T:.4f}s (span) applied to {len(motion_dirs)} clip(s)")
    print(f"replay : python trajectories/replay.py {os.path.basename(motion_dirs[0])} --axis")


if __name__ == "__main__":
    main()
