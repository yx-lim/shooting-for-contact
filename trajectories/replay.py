##
#
# Replay of a reference trajectory on the Unitree G1 in the MuJoCo viewer.
#
##

# standard imports
import sys
import os
import re
import glob
import time
import argparse
import numpy as np

# mujoco imports
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))

# repo root
ROOT = os.getenv("TRAJOPT_ROOT_DIR") or os.path.dirname(HERE)
sys.path.append(ROOT)


def load_trajectory(csv_path):
    """Load an (N x nq) qpos CSV; base quaternion assumed already in MuJoCo wxyz order."""
    Q = np.loadtxt(csv_path, delimiter=",")
    if Q.ndim == 1:
        Q = Q[None, :]
    return Q


def frame_times(traj_dir, n_frames):
    """Time stamp per qpos row from time.csv: used directly if its length matches,
    else resampled at its median dt. Falls back to 0.005 s if time.csv is absent."""
    tpath = os.path.join(traj_dir, "time.csv")
    if not os.path.exists(tpath):
        return np.arange(n_frames) * 0.005
    t = np.atleast_1d(np.loadtxt(tpath, delimiter=","))
    if len(t) == n_frames:
        return t
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.005
    return np.arange(n_frames) * dt


def find_motion_dir(motion):
    """Resolve a motion to its folder: absolute path, a <robot>/<group>/<motion> subpath, or a
    bare name searched anywhere under trajectories/ (must be unique)."""
    motion_path = motion if os.path.isabs(motion) else os.path.join(HERE, motion)
    if not os.path.isdir(motion_path):
        # fallback: search trajectories/ at any depth for a uniquely-named clip folder
        matches = sorted(glob.glob(os.path.join(HERE, "**", os.path.basename(motion)),
                                   recursive=True))
        matches = [m for m in matches if os.path.isdir(m)]
        if len(matches) == 1:
            motion_path = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"'{motion}' is ambiguous, found {len(matches)}: "
                + ", ".join(os.path.relpath(m, HERE) for m in matches)
                + " -- disambiguate with <group>/<motion>")
        else:
            raise NotADirectoryError(f"no motion folder: {motion_path}")
    return motion_path


def resolve_dof(motion_dir, dof):
    """Return (dof, csv_path). If `dof` is given, require that qpos_<dof>dof.csv;
    otherwise auto-detect from the files present, preferring 29, then 23, then smallest."""
    # explicit dof wins: take that exact variant or fail
    if dof is not None:
        csv_path = os.path.join(motion_dir, f"qpos_{dof}dof.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"no qpos_{dof}dof.csv in {motion_dir}")
        return dof, csv_path

    # otherwise collect every variant the folder ships, then prefer 29 -> 23 -> smallest
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


def infer_dof(ncols):
    """Map a qpos column count to the G1 dof (nq=36 -> 29-dof, nq=30 -> 23-dof)."""
    mapping = {36: 29, 30: 23}
    if ncols not in mapping:
        raise ValueError(f"cannot infer dof from {ncols} qpos columns "
                         f"(expected 36 for 29-dof or 30 for 23-dof)")
    return mapping[ncols]


def find_pose_csv(motion):
    """Resolve `motion` to a single loose qpos .csv -- a static pose or a clip not stored in the
    folder+qpos_<dof>dof.csv layout. Accepts a direct path (with or without the .csv suffix,
    absolute or relative to trajectories/) or a bare name found uniquely anywhere under
    trajectories/. Returns the path, or None if it does not resolve to a file (in which case the
    caller falls back to the motion-folder loader)."""
    # try the name as given, then with .csv appended, both bare and under trajectories/
    for cand in (motion, os.path.join(HERE, motion),
                 motion + ".csv", os.path.join(HERE, motion + ".csv")):
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # fall back to a unique match at any depth
    base = os.path.basename(motion)
    base = base if base.endswith(".csv") else base + ".csv"
    matches = sorted(m for m in glob.glob(os.path.join(HERE, "**", base), recursive=True)
                     if os.path.isfile(m))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"'{motion}' is ambiguous, found {len(matches)}: "
            + ", ".join(os.path.relpath(m, HERE) for m in matches) + " -- give the full path")
    return None


# translucent blue tint applied to every ghost geom (same as examples/replay.py)
GHOST_RGBA = np.array([0.3, 0.6, 1.0, 0.1])


def make_ghost(model_path):
    """Build the (model, data, opt, pert) needed to render reference poses as ghosts."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d, mujoco.MjvOption(), mujoco.MjvPerturb()


def draw_ghosts(scn, m, d, opt, pert, poses):
    """Overlay one translucent blue ghost per qpos in `poses` into the viewer's user scene.
    Used to mark the start and end of a periodic gait cycle. Only dynamic geoms (the robot
    bodies) are added, so the static floor is not duplicated. The poses are static, so this
    is called once; mjv_addGeoms appends, so successive poses accumulate in the same scene."""
    scn.ngeom = 0                                          # clear any previous ghosts
    for qpos in poses:
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        mujoco.mjv_addGeoms(m, d, opt, pert,
                            int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
    for i in range(scn.ngeom):                             # tint + make translucent
        scn.geoms[i].rgba[:] = GHOST_RGBA


# world-frame RGB triad dimensions (X=red, Y=green, Z=blue); helper same as examples/replay.py
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


def main():
    """Play a reference clip (or hold a single pose) in the MuJoCo viewer, paced by wall clock.
        motion      a clip folder with qpos_<dof>dof.csv, or a loose qpos .csv
        --dof       which variant to play, and the matching g1_<dof>dof model
        --speed     playback multiplier
        --no-ghost  suppress the start/end ghosts drawn for _periodic clips
        --no-axis   suppress the world-frame RGB triad
    Space pauses, arrows step +/-1 and +/-10 frames while paused, R restarts."""

    # command line
    p = argparse.ArgumentParser(
        description="Replay a G1 reference trajectory, or view a single static pose.")
    p.add_argument("motion",
                   help="a trajectory subfolder (folder with qpos_<dof>dof.csv, e.g. "
                        "kino_180_twist_jump), OR a path to a single loose qpos .csv (one static "
                        "pose, or a clip not stored in the folder layout)")
    p.add_argument("--dof", type=int, choices=(23, 29), default=None,
                   help="which qpos_<dof>dof.csv to play, and the matching g1_<dof>dof "
                        "model; if omitted, auto-detected from the files present")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (2.0 = twice as fast)")
    p.add_argument("--no-ghost", action="store_true",
                   help="don't draw start/end ghosts for _periodic motions")
    p.add_argument("--no-axis", action="store_true",
                   help="don't draw the world-frame RGB axis triad (X=red, Y=green, Z=blue); "
                        "the triad is drawn at the origin by default")
    args = p.parse_args()

    # resolve either a single loose qpos .csv (static pose / loose clip) or a motion folder
    csv_path = find_pose_csv(args.motion)
    if csv_path is not None:
        Q = load_trajectory(csv_path)
        dof = args.dof if args.dof is not None else infer_dof(Q.shape[1])
        csv_dir = os.path.dirname(csv_path)
        show_ghosts = False                       # a loose pose/clip is not a _periodic gait
    else:
        motion_dir = find_motion_dir(args.motion)
        dof, csv_path = resolve_dof(motion_dir, args.dof)
        Q = load_trajectory(csv_path)
        csv_dir = os.path.dirname(csv_path)
        # periodic gait crops (saved by gait/parse_data.py) get translucent start/end ghosts
        show_ghosts = ("_periodic" in os.path.basename(motion_dir)) and not args.no_ghost

    # model is the non-_feet g1 matching the resolved dof
    model_path = os.path.join(ROOT, "models", "unitree_g1", f"g1_{dof}dof.xml")

    # timing: per-frame stamps from the clip, and how long playback will take at this speed
    t_frames = frame_times(csv_dir, Q.shape[0])
    speed = max(args.speed, 1e-6)
    N = Q.shape[0]
    total = float(t_frames[-1] - t_frames[0])
    wall = total / speed if total > 0 else 0.0
    print(f"model : {model_path}")
    print(f"motion: {csv_path}  ({N} frame{'s' if N != 1 else ''}, {total:.2f} s, "
          f"speed={args.speed}x -> {wall:.2f} s wall)")

    model = load_model(model_path, add_axes=not args.no_axis)
    data = mujoco.MjData(model)
    if model.nq != Q.shape[1]:
        raise ValueError(f"model nq={model.nq} but motion has {Q.shape[1]} qpos")

    # wall-clock-driven playback at ~50 Hz: each tick advances a simulated-time accumulator by
    # the elapsed wall time (x speed) and maps it to a sample, so the clip's own rate is irrelevant
    t0_sim = float(t_frames[0])
    target_fps = 50.0
    frame_period = 1.0 / target_fps

    # playback state shared with the key callback (which runs on the viewer's GUI thread)
    state = {"paused": False, "k": 0, "sim_t": 0.0}

    # GLFW keycodes -> playback actions; every step key also pauses so the frame stays put
    def key_cb(keycode):
        if keycode == 32:                              # space: pause / resume
            state["paused"] = not state["paused"]
        elif keycode == 262:                           # right: step +1 frame (and pause)
            state["paused"] = True; state["k"] = min(state["k"] + 1, N - 1)
        elif keycode == 263:                           # left: step -1 frame (and pause)
            state["paused"] = True; state["k"] = max(state["k"] - 1, 0)
        elif keycode == 265:                           # up: step +10 frames (and pause)
            state["paused"] = True; state["k"] = min(state["k"] + 10, N - 1)
        elif keycode == 264:                           # down: step -10 frames (and pause)
            state["paused"] = True; state["k"] = max(state["k"] - 10, 0)
        elif keycode == 82:                            # R: restart from the first frame
            state["k"] = 0; state["sim_t"] = 0.0

    print("controls: [Space] pause/resume   [Left/Right] step -/+1 frame   "
          "[Up/Down] step -/+10 frames   [R] restart")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        # static blue ghosts at the first and last pose mark the gait cycle's start/end
        if show_ghosts:
            gm, gd, gopt, gpert = make_ghost(model_path)
            draw_ghosts(viewer.user_scn, gm, gd, gopt, gpert, [Q[0], Q[-1]])

        if N == 1:
            # single frame: hold the static pose and keep the window responsive
            data.qpos[:] = Q[0]
            data.time = 0.0
            mujoco.mj_forward(model, data)
            viewer.set_texts((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "static pose", None))
            while viewer.is_running():
                viewer.sync()
                time.sleep(frame_period)
            return

        last = time.perf_counter()
        while viewer.is_running():
            tick = time.perf_counter()
            if state["paused"] == True:
                # paused: the frame index is authoritative (arrow keys set it); keep the
                # time accumulator aligned so resuming continues from the shown frame
                state["sim_t"] = float(t_frames[state["k"]] - t0_sim)
            else:
                # playing: advance simulated time and wrap at the clip end to loop forever
                state["sim_t"] += (tick - last) * speed
                if total > 0 and state["sim_t"] > total:
                    state["sim_t"] %= total
                state["k"] = min(int(np.searchsorted(t_frames, t0_sim + state["sim_t"])),
                                 N - 1)
            last = tick

            k = state["k"]
            data.qpos[:] = Q[k]
            data.time = float(t_frames[k])
            mujoco.mj_forward(model, data)             # update derived quantities for rendering

            # top-left overlay: motion time / frame, speed scaling, pause state, control hints
            paused = "   [PAUSED]" if state["paused"] == True else ""
            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              f"time = {float(t_frames[k]):.2f} s   frame {k}/{N - 1}\n"
                              f"speed = {speed:g}x{paused}\n"
                              f"[Space] pause   [Left/Right] +/-1   [Up/Down] +/-10 frames   [R] restart",
                              None))
            viewer.sync()

            # pace this display tick to ~target_fps in wall-clock time
            sleep_for = frame_period - (time.perf_counter() - tick)
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
