##
#
# Replay solution trajectories
#
##

# standard imports
import argparse
import glob
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# mujoco imports
import mujoco
import mujoco.viewer

ROOT = os.getenv("TRAJOPT_ROOT_DIR")
sys.path.append(ROOT)

# local imports
from utils.file_utils import load_trajectory as load_trajectory_npz


########################################################################
# LOAD
########################################################################

def load_trajectory(name):
    """Load a trajectory .npz saved by an example solve.

    `name` is resolved under examples/ as:
      - "<folder>"        -> examples/<folder>/<folder>.npz   (one task per folder)
      - "<folder>/<stem>" -> examples/<folder>/<stem>.npz     (several tasks in one folder,
                             e.g. g1_tracking_mpc/g1_23dof_tracking_mpc)
    If neither resolves, a bare "<stem>" is searched for as examples/*/<stem>.npz, so a task
    that lives beside siblings in a shared folder is still found by its name alone (the match
    must be unique).

    Returns (t, x, u, model_path, ref, spline_type, traj_path); the path is handed back so the
    caller can read optional extra arrays (e.g. the gait TO's `defects`/`periodicity`) straight
    from the raw .npz.
    """
    rel = name if ("/" in name or os.sep in name) else os.path.join(name, name)
    traj_path = os.path.join(ROOT, "examples", rel + ".npz")
    if not os.path.exists(traj_path):
        # fallback: find examples/*/<stem>.npz (one folder deep) by its bare name
        stem = os.path.basename(name)
        matches = sorted(glob.glob(os.path.join(ROOT, "examples", "*", stem + ".npz")))
        if len(matches) == 1:
            traj_path = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"'{name}' is ambiguous, found {len(matches)}: "
                + ", ".join(os.path.relpath(m, ROOT) for m in matches)
                + " -- disambiguate with <folder>/<stem>")
        else:
            raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    # unpack the data (shared .npz schema with utils.file_utils.save_trajectory); also hand back
    # the resolved path so the caller can pull optional extras (defects/periodicity) from the npz
    return (*load_trajectory_npz(traj_path), traj_path)


########################################################################
# PLOT
########################################################################

def plot_trajectory(t, x, u, nq, nv, name, spline_type="zero"):
    """ Plot positions, velocities, and inputs vs time (non-blocking).

    Inputs are drawn to match the control spline: "zero" -> zero-order-hold steps,
    anything else ("linear"/"cubic") -> connected lines (ramps). """

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    # positions
    for i in range(nq):
        axs[0].plot(t, x[:, i], label=f"q[{i}]")
    axs[0].set_ylabel("position")
    axs[0].legend(loc="upper right").set_draggable(True)
    axs[0].grid(True)

    # velocities
    for i in range(nv):
        axs[1].plot(t, x[:, nq + i], label=f"v[{i}]")
    axs[1].set_ylabel("velocity")
    axs[1].legend(loc="upper right").set_draggable(True)
    axs[1].grid(True)

    # inputs (one fewer sample than states). Draw as zero-order-hold steps or as linear
    # ramps to match the spline the controls were parametrized with.
    stepped = spline_type == "zero"
    for i in range(u.shape[1]):
        if stepped:
            axs[2].step(t[:len(u)], u[:, i], where="post", label=f"u[{i}]")
        else:
            axs[2].plot(t[:len(u)], u[:, i], label=f"u[{i}]")
    axs[2].set_ylabel("input")
    axs[2].set_xlabel("time [s]")
    axs[2].legend(loc="upper right").set_draggable(True)
    axs[2].grid(True)

    fig.suptitle(f"trajectory: {name}")
    fig.tight_layout()

    # non-blocking: draw the window and return so replay can continue
    plt.show(block=False)
    plt.pause(0.001)
    return fig


def plot_defects(defects, nv, name, node_dt=None):
    """Plot the multiple-shooting dynamics defects (non-blocking).

    `defects` (N, ndx=2*nv) holds, per shooting interval i, the tangent residual
    state_diff(x_{i+1}, Phi_K(x_i, p)) = [dq(nv) | dv(nv)] -- how far the K-step rollout of
    interval i lands from the next node. Two curves show the per-interval infinity norm of the
    position- and velocity-tangent halves (log y), so spikes pinpoint the least dynamically
    feasible intervals. x-axis is interval-start time when node_dt is known, else interval index.
    """
    defects = np.asarray(defects)
    N = defects.shape[0]
    xs = node_dt * np.arange(N) if node_dt else np.arange(N)
    xlabel = "interval start time [s]" if node_dt else "interval index"
    pos = np.max(np.abs(defects[:, :nv]), axis=1)        # position-tangent (dq) per interval
    vel = np.max(np.abs(defects[:, nv:]), axis=1)        # velocity-tangent (dv) per interval

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(xs, pos, ".-", label="|defect pos|_inf")
    ax.semilogy(xs, vel, ".-", label="|defect vel|_inf")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("defect (tangent)")
    ax.set_title(f"multiple-shooting defects: {name}   "
                 f"(max |pos|={pos.max():.1e}, |vel|={vel.max():.1e})")
    ax.grid(True, which="both")
    ax.legend(loc="upper right").set_draggable(True)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    return fig


def print_periodicity(periodicity, nv):
    """Print the periodic-closure constraint error |state_diff(x_N, x_0)| over the closed
    tangent components (everything except base x,y). Split into pose (pz/orientation/joints,
    the first nv-2 entries) and velocity (the rest, present only when velocities were closed)."""
    r = np.asarray(periodicity).reshape(-1)
    pose, vel = r[:nv - 2], r[nv - 2:]
    msg = f"periodicity closure error: |pose|_max={np.abs(pose).max():.3e}"
    msg += (f"  |vel|_max={np.abs(vel).max():.3e}" if vel.size else "  vel: free")
    print(msg)


########################################################################
# GHOST REFERENCE
########################################################################

# translucent tint applied to every ghost geom
GHOST_RGBA = np.array([0.3, 0.6, 1.0, 0.1])


def make_ghost(model_path):
    """Build the (model, data, opt, pert) needed to render a reference robot as a ghost."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d, mujoco.MjvOption(), mujoco.MjvPerturb()


def draw_ghost(scn, m, d, opt, pert, qpos):
    """Overlay the reference robot at qpos into the viewer's user scene `scn` as a
    translucent ghost. Clears any previous ghost geoms and re-adds them at the new pose,
    so this is called once per frame. Only dynamic geoms (the robot bodies) are added --
    the static floor is skipped so it is not duplicated."""
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    scn.ngeom = 0                                          # clear last frame's ghost
    mujoco.mjv_addGeoms(m, d, opt, pert,
                        int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
    for i in range(scn.ngeom):                             # tint + make translucent
        scn.geoms[i].rgba[:] = GHOST_RGBA


########################################################################
# WORLD-FRAME AXIS
########################################################################

# world-frame RGB triad dimensions (X=red, Y=green, Z=blue)
AXIS_LEN = 0.2        # length of each axis cylinder [m]
AXIS_RADIUS = 0.008   # radius of each axis cylinder [m]
AXIS_ALPHA = 0.3      # opacity of the axis cylinders (0 = invisible, 1 = opaque)


def _resolve_model_path(model_path):
    """Rebase a baked-in model path onto the local repo when it doesn't exist here.

    Clips generated on another machine store an absolute model path under that machine's $HOME
    (e.g. /home/<other-user>/.../mj-nlp/models/unitree_g1/g1_29dof.xml). When replaying such a clip
    locally that path is missing, so remap the 'models/...' suffix onto the local ROOT."""
    if os.path.exists(model_path):
        return model_path
    norm = model_path.replace(os.sep, "/")
    idx = norm.rfind("/models/")
    if idx != -1 and ROOT:
        cand = os.path.join(ROOT, norm[idx + 1:])
        if os.path.exists(cand):
            return cand
    return model_path  # unchanged -> mujoco raises the original, informative error


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


########################################################################
# REPLAY
########################################################################

def replay(name, speed=1.0, plot=True, ghost=True, axis=False):
    """ Replay a saved trajectory in a viewer (optionally plot it first). """

    t, x, u, model_path, ref, spline_type, traj_path = load_trajectory(name)

    # rebase the baked-in model path onto the local repo (clips copied from another machine store
    # an absolute path under a different $HOME); fixes both the model and the ghost below.
    model_path = _resolve_model_path(model_path)

    # load the model the trajectory was generated with (optionally with a world-frame axis)
    model = load_model(model_path, add_axes=axis)
    data = mujoco.MjData(model)
    nq, nv = model.nq, model.nv

    # optional gait-TO diagnostics saved alongside the trajectory (defects / periodicity)
    npz = np.load(traj_path)
    defects = npz["defects"] if "defects" in npz.files else None
    periodicity = npz["periodicity"] if "periodicity" in npz.files else None
    node_dt = float(npz["node_dt"]) if "node_dt" in npz.files else None

    # ghost reference: shown for any tracking/gait result that saved one
    show_ghost = ghost and (("tracking" in name) or ("gait" in name)) and (ref is not None)
    if show_ghost:
        gm, gd, gopt, gpert = make_ghost(model_path)

    # show plots (non-blocking) and keep them open alongside the replay
    figs = []
    if plot:
        figs.append(plot_trajectory(t, x, u, nq, nv, name, spline_type))
        # gait-TO diagnostics: plot the multiple-shooting defects, print the periodicity error
        if defects is not None:
            figs.append(plot_defects(defects, nv, name, node_dt))
    if periodicity is not None:
        print_periodicity(periodicity, nv)

    N = x.shape[0]
    sim_t0 = float(t[0])
    sim_total = float(t[-1] - t[0])
    print(f"Replaying '{name}': N={N}, duration={sim_total:.2f}s, speed={speed}x")

    # wall-clock-driven playback at ~50 Hz. Space pauses; the arrow keys step frames while
    # paused (same key scheme as trajectories/gait_period.py and trajectories/replay.py).
    target_fps = 50.0
    frame_period = 1.0 / target_fps

    # playback state shared with the key callback (which runs on the viewer's GUI thread).
    # "forces" is applied to viewer.opt in the loop below -- the viewer does not exist yet here.
    state = {"paused": False, "k": 0, "sim_t": 0.0, "forces": False}

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
        elif keycode == 70:                            # F: toggle contact points + force arrows
            state["forces"] = not state["forces"]

    print("controls: [Space] pause/resume   [Left/Right] step -/+1 frame   "
          "[Up/Down] step -/+10 frames   [R] restart   [F] contact forces")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        try:
            # loop the trajectory forever until the window is closed or Ctrl+C
            last = time.perf_counter()
            while viewer.is_running():
                tick = time.perf_counter()
                if state["paused"] == True:
                    # paused: the frame index is authoritative (arrow keys set it); keep the
                    # time accumulator aligned so resuming continues from the shown frame
                    state["sim_t"] = float(t[state["k"]] - sim_t0)
                else:
                    # playing: advance simulated time and wrap at the clip end to loop forever
                    state["sim_t"] += (tick - last) * speed
                    if sim_total > 0 and state["sim_t"] > sim_total:
                        state["sim_t"] %= sim_total
                    state["k"] = min(int(np.searchsorted(t, sim_t0 + state["sim_t"])), N - 1)
                last = tick

                k = state["k"]
                # set state from the trajectory and refresh derived quantities
                data.qpos[:] = x[k, :nq]
                data.qvel[:] = x[k, nq:nq + nv]
                data.time = float(t[k])
                mujoco.mj_forward(model, data)

                # overlay the reference robot as a translucent ghost (tracking only)
                if show_ghost:
                    draw_ghost(viewer.user_scn, gm, gd, gopt, gpert,
                               ref[min(k, ref.shape[0] - 1), :nq])

                # contact visualization, toggled by [F]: force arrows plus the contact points
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = state["forces"]
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = state["forces"]

                # top-left overlay: sim time / frame, speed scaling, pause state, control hints
                paused = "   [PAUSED]" if state["paused"] == True else ""
                forces = "   [FORCES]" if state["forces"] else ""
                txt = (f"time = {float(t[k]):.2f} s   frame {k}/{N - 1}\n"
                       f"speed = {speed:g}x{paused}{forces}\n"
                       f"[Space] pause   [Left/Right] +/-1   [Up/Down] +/-10   [R] restart   "
                       f"[F] forces")
                viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  txt, None))

                viewer.sync()

                # keep the (non-blocking) plot windows responsive
                if figs and plt.get_fignums():
                    for f in figs:
                        f.canvas.flush_events()

                # pace this display tick to ~target_fps in wall-clock time
                sleep_for = frame_period - (time.perf_counter() - tick)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            pass


########################################################################
# MAIN
########################################################################

if __name__ == "__main__":

    # parse command line args
    parser = argparse.ArgumentParser(description="Replay a saved trajectory.")
    parser.add_argument("name", help="example folder in examples/ (loads <name>/<name>.npz)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--no-plot", action="store_true", help="skip the state / input plots")
    parser.add_argument("--no-ghost", action="store_true", help="hide the reference ghost")
    parser.add_argument("--axis", action="store_true",
                        help="draw a world-frame RGB axis triad (X=red, Y=green, Z=blue) at the origin")
    args = parser.parse_args()

    # replay the trajectory
    replay(args.name, speed=args.speed, plot=not args.no_plot,
           ghost=not args.no_ghost, axis=args.axis)
