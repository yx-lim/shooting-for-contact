##
#
# Automatic periodic-cycle detection for retargeted human-motion clips.
#
##


# standard imports
import os
import sys
import time
import argparse
import numpy as np

# mujoco imports
import mujoco
import mujoco.viewer

# repo root
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("TRAJOPT_ROOT_DIR") or os.path.dirname(HERE)
TRAJ = os.path.join(ROOT, "trajectories")
sys.path.append(ROOT)

# local imports
from utils.math_utils import projected_gravity, differentiate_qpos


########################################################################
# MOTION RESOLUTION / IO
########################################################################

def find_motion_dir(motion):
    """Resolve a motion to its folder: absolute path, a <robot>/<group>/<motion> subpath, or a
    bare name searched anywhere under trajectories/ (must be unique)."""
    motion_path = motion if os.path.isabs(motion) else os.path.join(TRAJ, motion)
    if not os.path.isdir(motion_path):
        import glob
        matches = sorted(glob.glob(os.path.join(TRAJ, "**", os.path.basename(motion)),
                                   recursive=True))
        matches = [m for m in matches if os.path.isdir(m)]
        if len(matches) == 1:
            motion_path = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"'{motion}' is ambiguous, found {len(matches)}: "
                + ", ".join(os.path.relpath(m, TRAJ) for m in matches)
                + " -- disambiguate with <group>/<motion>")
        else:
            raise NotADirectoryError(f"no motion folder: {motion_path}")
    return motion_path


def resolve_dof(motion_dir, dof=None):
    """Return (dof, csv_path). If `dof` is given, require that qpos_<dof>dof.csv; else
    auto-detect from the files present, preferring 29, then 23, then smallest."""
    import glob, re
    if dof is not None:
        csv_path = os.path.join(motion_dir, f"qpos_{dof}dof.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"no qpos_{dof}dof.csv in {motion_dir}")
        return dof, csv_path
    avail = {}
    for p in glob.glob(os.path.join(motion_dir, "qpos_*dof.csv")):
        m = re.match(r"qpos_(\d+)dof\.csv$", os.path.basename(p))
        if m:
            avail[int(m.group(1))] = p
    if not avail:
        raise FileNotFoundError(f"no qpos_<dof>dof.csv in {motion_dir}")
    for pref in (29, 23):
        if pref in avail:
            return pref, avail[pref]
    d = min(avail)
    return d, avail[d]


def available_dofs(motion_dir):
    """Sorted list of dofs that have a qpos_<dof>dof.csv in `motion_dir`."""
    import glob, re
    dofs = []
    for p in glob.glob(os.path.join(motion_dir, "qpos_*dof.csv")):
        m = re.match(r"qpos_(\d+)dof\.csv$", os.path.basename(p))
        if m:
            dofs.append(int(m.group(1)))
    return sorted(dofs)


def load_trajectory(csv_path):
    """Load an (N x nq) qpos CSV."""
    Q = np.loadtxt(csv_path, delimiter=",")
    if Q.ndim == 1:
        Q = Q[None, :]
    return Q


def frame_times(motion_dir, n_frames):
    """Time stamp per qpos row from time.csv (used directly if its length matches, else resampled
    at its median dt). Falls back to 0.005 s if time.csv is absent."""
    tpath = os.path.join(motion_dir, "time.csv")
    if not os.path.exists(tpath):
        return np.arange(n_frames) * 0.005
    t = np.atleast_1d(np.loadtxt(tpath, delimiter=","))
    if len(t) == n_frames:
        return t
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.005
    return np.arange(n_frames) * dt


########################################################################
# FEATURE SIGNAL
########################################################################

def build_features(Q, V=None, use_base=True, use_vel=False, standardize=False):
    """Per-frame feature matrix (N, D) for periodicity scoring: joint angles, optional base
    features (projected gravity + height), optional velocities; `standardize` z-scores per channel."""
    cols = [Q[:, 7:]]                             # qj (N, nj) actuated-joint angles
    if use_base:
        cols.append(projected_gravity(Q[:, 3:7]))    # pg (N, 3) gravity in the body frame (no gimbal lock)
        cols.append(Q[:, 2:3])                        # pz (N, 1) base height
    if use_vel:
        if V is None:
            raise ValueError("build_features(use_vel=True) requires the velocity array V")
        if use_base:
            cols.append(V[:, 0:3])                    # vx, vy, vz (N, 3) base linear vel, WORLD
            cols.append(V[:, 3:6])                    # omega_b (N, 3) base angular vel, BODY
        cols.append(V[:, 6:])                         # qj_dot (N, nj) joint velocities
    feature = np.hstack(cols)
    if standardize:
        std = feature.std(axis=0)
        std[std < 1e-9] = 1.0
        feature = (feature - feature.mean(axis=0)) / std
    return feature


########################################################################
# DETECTION
########################################################################

def dist_curve(feature, tau_min, tau_max):
    """dist(tau) = mean_k ||feature[k+tau]-feature[k]||^2 over tau in [tau_min, tau_max]; dips at
    the period (poses realign), mean-invariant. Returns (taus, dist)."""
    taus = np.arange(tau_min, tau_max + 1)
    dist = np.empty(len(taus), dtype=float)
    for i, tau in enumerate(taus):
        diff = feature[tau:] - feature[:-tau]
        dist[i] = np.mean(np.sum(diff * diff, axis=1))
    return taus, dist


def pick_period(taus, dist, depth_tol=1.15):
    """Fundamental period: among interior local minima of dist, the SMALLEST tau within depth_tol
    of the deepest (favours the fundamental over 2x/3x multiples). Returns (P, is_fallback)."""
    minima = [i for i in range(1, len(dist) - 1) if dist[i] < dist[i - 1] and dist[i] <= dist[i + 1]]
    if not minima:
        return int(taus[int(np.argmin(dist))]), True
    best = min(minima, key=lambda i: dist[i])
    thresh = dist[best] * depth_tol
    for i in minima:                      # minima are in increasing-tau order
        if dist[i] <= thresh:
            return int(taus[i]), False
    return int(taus[best]), False


def _zscore(A):
    """Per-column z-score; guards near-constant columns (std < 1e-9 -> 1)."""
    A = np.asarray(A, dtype=float)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (A - mu) / sd


def closure_blocks(Q, V, use_base):
    """Per-frame z-scored POSITION and VELOCITY blocks for the cycle-start closure (equal weight per
    channel). pos = joints + base gravity/height; vel = base lin/ang + joint vels (None if V is None)."""
    pos = [Q[:, 7:]]
    if use_base:
        pos.append(projected_gravity(Q[:, 3:7]))
        pos.append(Q[:, 2:3])
    pos_feat = _zscore(np.hstack(pos))
    vel_feat = None
    if V is not None:
        vel = []
        if use_base:
            vel.append(V[:, 0:3]); vel.append(V[:, 3:6])
        vel.append(V[:, 6:])
        vel_feat = _zscore(np.hstack(vel))
    return pos_feat, vel_feat


def start_closure(pos, vel, P, start, stop, mode="combo"):
    """Closure over candidate starts: dpos=||pos[s+P]-pos[s]||, dvel=||vel[s+P]-vel[s]|| (z-scored);
    mode selects pos / vel / combo=sqrt(dpos^2+dvel^2). Returns (idx, score, dpos, dvel)."""
    idx = np.arange(start, stop - P)
    dpos = np.linalg.norm(pos[idx + P] - pos[idx], axis=1)
    dvel = (np.linalg.norm(vel[idx + P] - vel[idx], axis=1)
            if vel is not None else np.zeros_like(dpos))
    if mode in ("vel", "combo") and vel is None:
        raise ValueError(f"--closure {mode} needs velocities, but none were computed")

    if mode == "pos":
        score = dpos
    elif mode == "vel":
        score = dvel
    else:                                            # combo: equal weight across all z-scored channels
        score = np.sqrt(dpos ** 2 + dvel ** 2)
    return idx, score, dpos, dvel


def period_basin(taus, dist, P0, depth_tol=1.15):
    """Contiguous lags around P0 with dist within depth_tol of the global minimum -- the periods as
    periodic as the fundamental; confining refinement here avoids non-periodic lags. Returns (lo, hi)."""
    thr = float(dist.min()) * depth_tol
    i0 = int(np.argmin(np.abs(taus - P0)))
    lo = hi = i0
    while lo - 1 >= 0 and dist[lo - 1] <= thr:
        lo -= 1
    while hi + 1 < len(dist) and dist[hi + 1] <= thr:
        hi += 1
    return int(taus[lo]), int(taus[hi])


def refine_cycle(pos, vel, P_lo, P_hi, start, stop, mode="combo"):
    """Cycle (start, period) minimizing the `mode` closure: search every valid start s jointly with
    the period over the periodicity basin [P_lo, P_hi]. Returns (s, P, info={score, dpos, dvel})."""
    best = None
    for P in range(max(2, P_lo), P_hi + 1):
        if stop - P <= start:                        # no valid start at this period
            continue
        idx, score, dpos, dvel = start_closure(pos, vel, P, start, stop, mode)
        j = int(np.argmin(score))
        cand = (float(score[j]), int(idx[j]), int(P), float(dpos[j]), float(dvel[j]))
        if best is None or cand[0] < best[0]:
            best = cand
    sc, s, P, dp, dv = best
    return s, P, dict(score=sc, dpos=dp, dvel=dv)


def detect_cycle(feature, pos, vel, dt, mode="combo",
                 min_period_s=0.3, max_period_s=None, start=0, stop=None):
    """Detect one periodic cycle: averaged fundamental P0 from `feature` (dist_curve/pick_period),
    then a joint (start, period) refinement over the periodicity basin from the `mode` closure.
    Returns a dict (P, P0, s, duration, closure, dpos, dvel, refine_lo, refine_hi...)."""
    n = feature.shape[0]
    stop = n if stop is None else min(stop, n)
    win = stop - start
    tau_min = max(2, int(round(min_period_s / dt)))
    cap = win // 2                                   # need >= 2 cycles to see a recurrence
    tau_max = cap if max_period_s is None else min(int(round(max_period_s / dt)), cap)

    if tau_max <= tau_min:
        # window too short for two cycles -> treat the whole window as one cycle
        P = win - 1
        dpos = float(np.linalg.norm(pos[stop - 1] - pos[start]))
        dvel = float(np.linalg.norm(vel[stop - 1] - vel[start])) if vel is not None else 0.0
        return dict(P=P, P0=P, s=start, duration=P * dt, n_cycles=1.0, closure=dpos,
                    dpos=dpos, dvel=dvel, mode=mode, taus=None, dist=None,
                    fallback=True, refine_lo=P, refine_hi=P)

    taus, dist = dist_curve(feature[start:stop], tau_min, tau_max)
    P0, fb = pick_period(taus, dist)
    P_lo, P_hi = period_basin(taus, dist, P0)          # refine only over genuinely periodic lags
    s, P, cinfo = refine_cycle(pos, vel, P_lo, P_hi, start, stop, mode)
    return dict(P=P, P0=P0, s=s, duration=P * dt, n_cycles=win / P, closure=cinfo["score"],
                dpos=cinfo["dpos"], dvel=cinfo["dvel"], mode=mode,
                taus=taus, dist=dist, fallback=fb, refine_lo=P_lo, refine_hi=P_hi)


########################################################################
# OUTPUT
########################################################################

def save_cycle(motion_dir, s, length, t, dofs, suffix="periodic"):
    """Crop [s, s+length] from every dof's qpos CSV into <motion_dir>_<suffix>/, re-zeroing time and
    re-origining base x,y to the crop start (keeps the forward stride). Returns the output dir."""
    out_dir = os.path.join(os.path.dirname(motion_dir),
                           os.path.basename(motion_dir) + "_" + suffix)
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "time.csv"), t[s:s + length + 1] - t[s], delimiter=",")
    for dof in dofs:
        Q = load_trajectory(os.path.join(motion_dir, f"qpos_{dof}dof.csv"))
        if Q.shape[0] != len(t):
            raise ValueError(f"qpos_{dof}dof.csv has {Q.shape[0]} frames, "
                             f"but time.csv has {len(t)} -- dof files must align")
        crop = Q[s:s + length + 1].copy()
        crop[:, 0] -= crop[0, 0]                # re-origin base x to the crop start
        crop[:, 1] -= crop[0, 1]                # re-origin base y to the crop start
        np.savetxt(os.path.join(out_dir, f"qpos_{dof}dof.csv"), crop, delimiter=",")
    return out_dir


########################################################################
# DIAGNOSTIC PLOTS  (how the period / start are picked)
########################################################################

def plot_detection(feature, pos, vel, info, dt, depth_tol=1.15):
    """Diagnostic figure: top row = dist(tau) | self-distance matrix, bottom row = closure(s).
    Shown non-blocking and returned so the caller keeps it live beside the MuJoCo viewer."""
    import matplotlib.pyplot as plt

    P, s, n = info["P"], info["s"], feature.shape[0]
    P0 = info.get("P0", P)                             # averaged fundamental (before refinement)
    # 2x2 mosaic: top row = dist(tau) | recurrence; bottom row = closure spanning both columns
    fig, axd = plt.subplot_mosaic([["dist", "recur"],
                                   ["closure", "closure"]], figsize=(13, 9))

    # (1) self-distance curve -> period P
    a = axd["dist"]
    taus, dist = info["taus"], info["dist"]
    if taus is None:                                   # fallback path has no curve
        a.text(0.5, 0.5, "fallback: no dist(tau) curve\n(window too short for >=2 cycles)",
               ha="center", va="center", transform=a.transAxes)
    else:
        a.plot(taus, dist, "-", color="0.4", lw=1)
        minima = [i for i in range(1, len(dist) - 1)        # same rule as pick_period
                  if dist[i] < dist[i - 1] and dist[i] <= dist[i + 1]]
        if minima:
            a.plot(taus[minima], dist[minima], "o", ms=4, color="tab:blue", label="local minima")
            best = min(minima, key=lambda i: dist[i])
            a.plot(taus[best], dist[best], "s", ms=8, mfc="none", color="tab:orange", label="deepest")
            a.axhline(dist[best] * depth_tol, ls=":", color="tab:orange",
                      label=f"accept <= {depth_tol:g} x deepest")
        lo, hi = info.get("refine_lo"), info.get("refine_hi")
        if lo is not None and hi is not None and hi > lo:   # periodicity basin the refinement searches
            a.axvspan(lo, hi, color="tab:green", alpha=0.12, label=f"refine basin [{lo}, {hi}]")
        a.axvline(P0, color="tab:red", lw=1.5, label=f"P0 = {P0}")
        pi = int(np.argmin(np.abs(taus - P0)))          # black dot at the averaged fundamental
        a.plot(taus[pi], dist[pi], "o", ms=8, color="black", mec="white", mew=1.0, zorder=5)
        if P != P0:                                     # refinement nudged the period off P0
            a.axvline(P, color="tab:green", ls="--", lw=1.2, label=f"P = {P} (refined)")
        for mult in (2, 3):                             # harmonics P0 should beat
            if mult * P0 <= taus[-1]:
                a.axvline(mult * P0, color="tab:red", ls="--", lw=0.8, alpha=0.5)
        a.legend(fontsize=8).set_draggable(True)
    a.set(title=r"$\mathrm{dist}(\tau) = \langle\,\| f_{k+\tau} - f_k \|^2 \rangle_k$",
          xlabel=r"period in frames [$\tau$]", ylabel="feature distance [dist]")

    # (2) closure over candidate starts: pos (blue), vel (red), chosen-mode total (black); red bar = pick
    a = axd["closure"]
    mode = info.get("mode", "combo")
    idx, score, dpos, dvel = start_closure(pos, vel, P, 0, n, mode)
    total_lbl = {"combo": r"total  $\sqrt{\|\Delta p\|^2+\|\Delta v\|^2}$",
                 "pos":   r"total  $\|\Delta p\|$",
                 "vel":   r"total  $\|\Delta v\|$"}.get(mode, "total")
    a.plot(idx, dpos, "-", color="tab:blue", lw=1.2, alpha=0.6,
           label=r"pos  $\|p_{s+P}-p_s\|$")
    if vel is not None:
        a.plot(idx, dvel, "-", color="tab:red", lw=1.2, alpha=0.6,
               label=r"vel  $\|v_{s+P}-v_s\|$")
    a.plot(idx, score, "-", color="black", lw=1.8, alpha=0.6, label=total_lbl)
    js = int(np.argmin(score))
    a.axvline(s, color="tab:red", lw=1.5, label=f"s = {s}")   # same red pick-bar as the dist plot
    a.plot(idx[js], score[js], "o", ms=8, color="black", mec="white", mew=1.0, zorder=5)
    a.legend(fontsize=8).set_draggable(True)
    title_eq = {"combo": r"$\mathrm{closure}(s) = \sqrt{\|p_{s+P}-p_s\|^2 + \|v_{s+P}-v_s\|^2}$",
                "pos":   r"$\mathrm{closure}(s) = \|p_{s+P}-p_s\|$",
                "vel":   r"$\mathrm{closure}(s) = \|v_{s+P}-v_s\|$"}.get(mode, r"$\mathrm{closure}(s)$")
    a.set(title=title_eq, xlabel="frame start [s]", ylabel="closure")

    # (3) self-distance matrix: periodic structure shows as diagonal stripes spaced by P
    a = axd["recur"]
    step = max(1, n // 800)                            # cap memory/time for long clips
    fsub = feature[::step]
    D = np.empty((fsub.shape[0], fsub.shape[0]))
    for i in range(fsub.shape[0]):
        D[i] = np.linalg.norm(fsub - fsub[i], axis=1)
    im = a.imshow(D, origin="upper", cmap="viridis", extent=[0, n, n, 0], aspect="auto")
    for v in (s, s + P):                               # mark the chosen cycle window
        a.axvline(v, color="w", lw=0.8); a.axhline(v, color="w", lw=0.8)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    a.set(title=r"self-distance matrix  $\| f_i - f_j \|$", xlabel="frame [i]", ylabel="frame [j]")
    if step > 1:
        a.text(0.02, 0.98, f"subsampled x{step}", color="w", fontsize=8,
               ha="left", va="top", transform=a.transAxes)

    refine_note = "" if P == P0 else f"  (P0={P0}, refined {P - P0:+d})"
    fig.suptitle(f"gait detection   P={P} ({P * dt:.3f}s)   s={s}{refine_note}")
    fig.tight_layout()
    # non-blocking so the caller keeps it live beside the viewer (which pumps its GUI events)
    plt.show(block=False)
    plt.pause(0.001)                     # force the initial draw so the window appears now
    return fig


########################################################################
# INTERACTIVE TUNING  (opens the MuJoCo viewer)
########################################################################

# translucent blue tint for the start/end ghost poses (same as replay.py)
GHOST_RGBA = np.array([0.3, 0.6, 1.0, 0.1])

# world-frame RGB triad dimensions (X=red, Y=green, Z=blue); helper same as replay.py
AXIS_LEN = 0.2        # length of each axis cylinder [m]
AXIS_RADIUS = 0.008   # radius of each axis cylinder [m]
AXIS_ALPHA = 0.3      # opacity of the axis cylinders (0 = invisible, 1 = opaque)


def load_model(model_path, add_axes=False):
    """Load the model; if add_axes, append a world-frame RGB triad at the origin (X=red,
    Y=green, Z=blue) via MjSpec before compiling. The triad geoms are non-colliding and
    massless, so they are purely decorative and do not affect the dynamics."""
    if not add_axes:
        return mujoco.MjModel.from_xml_path(model_path)
    spec = mujoco.MjSpec.from_file(model_path)
    body = spec.worldbody.add_body(name="world_axes")
    for fromto, rgba in (
        ([0, 0, 0, AXIS_LEN, 0, 0], [1, 0, 0, AXIS_ALPHA]),   # +X red
        ([0, 0, 0, 0, AXIS_LEN, 0], [0, 1, 0, AXIS_ALPHA]),   # +Y green
        ([0, 0, 0, 0, 0, AXIS_LEN], [0, 0, 1, AXIS_ALPHA]),   # +Z blue
    ):
        g = body.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.fromto = fromto
        g.size = [AXIS_RADIUS, 0, 0]
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0
        g.density = 0.0
    return spec.compile()


def tune_cycle(motion_dir, dof, Q, t, dt, pos, vel, s0, P, dofs, auto_closure, axis=True, fig=None):
    """Slide the detected cycle window in the MuJoCo viewer: arrows nudge the start (1 / 10 frames),
    S saves the crop to <name>_periodic/, Space pauses; HUD shows the pos/vel seam closures."""
    model_path = os.path.join(ROOT, "models", "unitree_g1", f"g1_{dof}dof.xml")
    model = load_model(model_path, add_axes=axis)
    data = mujoco.MjData(model)
    if model.nq != Q.shape[1]:
        raise ValueError(f"model nq={model.nq} but motion has {Q.shape[1]} qpos")

    N = Q.shape[0]
    off_min, off_max = -s0, (N - 1 - P) - s0     # keep [s0+off, s0+off+P] within [0, N-1]
    clamp = lambda o: int(max(off_min, min(off_max, o)))

    # separate model/data to render the start/end poses as translucent ghost overlays
    gm = mujoco.MjModel.from_xml_path(model_path)
    gd, gopt, gpert = mujoco.MjData(gm), mujoco.MjvOption(), mujoco.MjvPerturb()

    def draw_ghosts(scn, poses, colors):
        """Overlay each pose as a translucent ghost in its own colour."""
        scn.ngeom = 0
        for q, rgba in zip(poses, colors):
            first = scn.ngeom
            gd.qpos[:] = q
            mujoco.mj_forward(gm, gd)
            mujoco.mjv_addGeoms(gm, gd, gopt, gpert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
            for i in range(first, scn.ngeom):
                scn.geoms[i].rgba[:] = rgba

    state = {"offset": 0, "paused": False, "save": False}

    def key_cb(keycode):
        if keycode == 262:   state["offset"] = clamp(state["offset"] + 1)     # right
        elif keycode == 263: state["offset"] = clamp(state["offset"] - 1)     # left
        elif keycode == 265: state["offset"] = clamp(state["offset"] + 10)    # up
        elif keycode == 264: state["offset"] = clamp(state["offset"] - 10)    # down
        elif keycode == 82:  state["offset"] = 0                              # R: reset
        elif keycode == 83:  state["save"] = True                             # S: save
        elif keycode == 32:  state["paused"] = not state["paused"]            # space

    print("\nTUNE: [Left/Right] shift 1   [Up/Down] shift 10   R reset   "
          "S save   Space pause   (close window to finish)")

    cycle_dur = max(P * dt, dt)
    frame_period = 1.0 / 50.0
    saved_any = False

    # if a diagnostic figure is open, pump its GUI events each frame so it stays interactive
    plt = None
    if fig is not None:
        import matplotlib.pyplot as plt

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = Q[s0][:3]     # center once on the initial frame; never re-track after
        phase, last, last_s = 0.0, time.perf_counter(), None
        while viewer.is_running():
            tick = time.perf_counter()
            if not state["paused"]:
                phase = (phase + (tick - last)) % cycle_dur     # advance the loop in real time
            last = tick

            s = s0 + clamp(state["offset"])
            c = min(int(round(phase / dt)), P)
            data.qpos[:] = Q[s + c]
            data.time = float(c * dt)
            mujoco.mj_forward(model, data)

            if s != last_s:                                     # window moved -> redraw ghosts
                draw_ghosts(viewer.user_scn, [Q[s], Q[s + P]], [GHOST_RGBA, GHOST_RGBA])
                last_s = s

            dpos = float(np.linalg.norm(pos[s + P] - pos[s]))
            dvel = float(np.linalg.norm(vel[s + P] - vel[s])) if vel is not None else 0.0
            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              f"TUNE   offset = {s - s0:+d} frames\n"
                              f"window [s, s+P] = [{s}, {s + P}]   P = {P}\n"
                              f"closure pos = {dpos:.4f}  vel = {dvel:.4f}   (auto score = {auto_closure:.4f})\n"
                              f"[Left/Right] 1   [Up/Down] 10   R reset   S save   Space pause",
                              None))
            viewer.sync()
            if plt is not None and plt.fignum_exists(fig.number):
                fig.canvas.flush_events()                       # keep the plot window responsive

            if state["save"]:
                state["save"] = False
                out_dir = save_cycle(motion_dir, s, P, t, dofs, suffix="periodic")
                saved_any = True
                print(f"saved  : {out_dir}  (offset {s - s0:+d}, frames [{s}, {s + P}], "
                      f"closure pos={dpos:.4f} vel={dvel:.4f})")

            sleep_for = frame_period - (time.perf_counter() - tick)
            if sleep_for > 0:
                time.sleep(sleep_for)

    if not saved_any:
        print("not saved (press S in the viewer to save a crop).")


########################################################################
#  MAIN
########################################################################

def main():
    """Detect one periodic cycle in a clip, then open the viewer to review and save it.
        motion        clip folder holding qpos_<dof>dof.csv + time.csv
        --dof         detect on this variant only (default: prefer 29, crop every variant)
        --min-period  shortest period to consider [s]
        --max-period  longest period to consider [s] (default: half the clip)
        --no-base     joints only, dropping base height + projected gravity from the features
        --closure     what picks the cycle START: pos | vel | combo (period detection is unaffected)
    Detection is symmetry-agnostic (see trajectories/README.md); nothing is written until S."""

    # command line
    p = argparse.ArgumentParser(
    description="Detect one periodic cycle in a qpos trajectory and save it.")
    p.add_argument("motion", help="trajectory subfolder name (e.g. walk1_subject1_crop148-376_z-0.025)")
    p.add_argument("--dof", type=int, choices=(23, 29), default=None,
                   help="which qpos_<dof>dof.csv to use; if omitted, auto-detected")
    p.add_argument("--min-period", type=float, default=0.2,
                   help="shortest period to consider, in seconds (default 0.2)")
    p.add_argument("--max-period", type=float, default=None,
                   help="longest period to consider, in seconds (default: half the clip)")
    p.add_argument("--no-base", dest="use_base", action="store_false",
                   help="use joints only; exclude the base features (height z + projected gravity)")
    p.add_argument("--closure", choices=("pos", "vel", "combo"), default="combo",
                   help="what drives the cycle-START (phase) pick: 'pos' minimizes pose closure "
                        "||pos[s+P]-pos[s]||, 'vel' minimizes velocity closure ||vel[s+P]-vel[s]||, "
                        "'combo' (default) uses both. All channels are per-channel z-scored (equal "
                        "weight), so combo = sqrt(pos^2 + vel^2). Period detection is unaffected.")
    args = p.parse_args()

    motion_dir = find_motion_dir(args.motion)
    # detect on one dof (prefer 29) but crop every available dof with the same cycle; --dof restricts
    dof, csv_path = resolve_dof(motion_dir, args.dof)
    dofs = [args.dof] if args.dof is not None else available_dofs(motion_dir)
    Q = load_trajectory(csv_path)
    n = Q.shape[0]
    t = frame_times(motion_dir, n)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.005

    # velocities (quaternion-aware FD) are always computed: they feed the period feature and closure
    model = mujoco.MjModel.from_xml_path(
        os.path.join(ROOT, "models", "unitree_g1", f"g1_{dof}dof.xml"))
    V = differentiate_qpos(model, Q, dt, centered=True)

    # features are always per-channel z-scored (equal weight) with velocities included
    feature = build_features(Q, V=V, use_base=args.use_base, use_vel=True, standardize=True)
    pos_feat, vel_feat = closure_blocks(Q, V, args.use_base)

    print(f"motion : {csv_path}  (detecting on {dof}-dof)")
    print(f"frames : {n}  ({t[-1] - t[0]:.2f} s, dt={dt * 1e3:.2f} ms, {1 / dt:.1f} Hz)  "
          f"features={feature.shape[1]}d (+vel)")
    print(f"closure: mode={args.closure}")

    info = detect_cycle(feature, pos_feat, vel_feat, dt, mode=args.closure,
                        min_period_s=args.min_period, max_period_s=args.max_period)
    P, s, P0 = info["P"], info["s"], info.get("P0", info["P"])
    if info["fallback"]:
        print("  WARNING: window too short for >=2 cycles (or no clear minimum) -- "
              "treating the whole window as one cycle.")
    lo, hi = info.get("refine_lo", P), info.get("refine_hi", P)
    change = "no change" if P == P0 else f"refined {P - P0:+d} from P0={P0}"
    refnote = f"   [basin [{lo},{hi}]: {change}]" if hi > lo else ""
    print(f"period : {P} frames  ({info['duration']:.3f} s)  "
          f"~{info['n_cycles']:.1f} cycles in clip{refnote}")
    print(f"cycle  : frames [{s}, {s + P}]  (score={info['closure']:.4f}  "
          f"pos closure={info['dpos']:.4f}  vel closure={info['dvel']:.4f})")

    # diagnostic plots (always shown, live beside the viewer); saving happens on S in tune_cycle
    fig = plot_detection(feature, pos_feat, vel_feat, info, dt)
    tune_cycle(motion_dir, dof, Q, t, dt, pos_feat, vel_feat, s, P, dofs, info["closure"], fig=fig)


if __name__ == "__main__":
    main()
