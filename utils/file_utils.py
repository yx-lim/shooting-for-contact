##
#
# File Utils I/O helpers shared by the examples.
#
##

# standard imports
import glob
import os
import re
import numpy as np

# custom imports
from utils.math_utils import differentiate_qpos


########################################################################
# REFERENCE LOADING
########################################################################

def resolve_qpos_csv(traj_dir, model=None, dof=None):
    """Locate the qpos CSV inside a clip folder.

    A clip stores its poses as `qpos_<dof>dof.csv`, where <dof> counts the JOINTS, so the file
    has nq = 7 + dof columns for a floating-base model (G1 29-dof -> qpos_29dof.csv, 36 cols;
    Go2 -> qpos_12dof.csv, 19 cols). A clip may ship several dof variants (see
    trajectories/gait_period.py, which crops them all identically), so the right one is picked as:

      1. `dof`, when given -- that exact variant, or an error;
      2. else the variant whose dof matches `model` (dof = model.nq - 7), when a model is given;
      3. else the only variant present (ambiguous if there are several).

    Returns (dof, csv_path).
    """
    # explicit dof wins: take that exact variant or fail
    if dof is not None:
        csv_path = os.path.join(traj_dir, f"qpos_{int(dof)}dof.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"no qpos_{int(dof)}dof.csv in {traj_dir}")
        return int(dof), csv_path

    # otherwise scan the folder for every qpos_<dof>dof.csv it ships
    found = {}                                        # dof -> path
    for p in sorted(glob.glob(os.path.join(traj_dir, "qpos_*dof.csv"))):
        m = re.match(r"qpos_(\d+)dof\.csv$", os.path.basename(p))
        if m:
            found[int(m.group(1))] = p
    if not found:
        raise FileNotFoundError(f"no qpos_<dof>dof.csv in {traj_dir}")

    # with a model, pick the variant whose joint count matches it
    if model is not None:
        want = int(model.nq) - 7                      # joints = nq - (3 base pos + 4 base quat)
        if want in found:
            return want, found[want]
        raise FileNotFoundError(
            f"model nq={model.nq} wants qpos_{want}dof.csv but {traj_dir} has "
            + ", ".join(f"qpos_{d}dof.csv" for d in sorted(found)))

    # no model and no dof: only unambiguous if the clip ships exactly one variant
    if len(found) > 1:
        raise ValueError(f"{traj_dir} has several dof variants ("
                         + ", ".join(str(d) for d in sorted(found))
                         + "); pass dof= to choose one")
    return next(iter(found.items()))                  # (dof, path)


def load_reference(traj_dir, model, pz_offset=0.0, centered=True, dof=None):
    """Load a clip's qpos CSV and build the full state reference [q, v]; returns
    (X_ref (N, nx), dt_ref).
        traj_dir    clip folder -- poses from qpos_<dof>dof.csv, dt from time.csv's sample period
        model       picks the dof variant (resolve_qpos_csv), so one loader serves G1 and Go2
        pz_offset   [m] added to base qpos[2], lifting the whole clip; velocities unaffected
        centered    symmetric finite differences for the velocities instead of forward
        dof         override the variant `model` would select
    Poses are assumed already in MuJoCo order with a wxyz base quaternion; velocities come from
    quaternion-aware differencing at the clip's own dt."""

    # load the poses and check them against the model (a single-frame clip loads as one row)
    _, csv_path = resolve_qpos_csv(traj_dir, model, dof)
    Q = np.loadtxt(csv_path, delimiter=",")
    if Q.ndim == 1:
        Q = Q[None, :]
    if Q.shape[1] != model.nq:
        raise ValueError(f"{csv_path}: qpos has {Q.shape[1]} cols but model nq={model.nq}")

    # raise the base (and thus the whole robot via FK) by pz_offset [m]; base z is qpos[2]
    Q[:, 2] += pz_offset

    # per-frame dt from time.csv (its sample period), falling back to the model timestep
    tpath = os.path.join(traj_dir, "time.csv")
    if os.path.exists(tpath):
        t = np.atleast_1d(np.loadtxt(tpath, delimiter=","))
        dt_ref = float(np.median(np.diff(t))) if len(t) > 1 else float(model.opt.timestep)
    else:
        dt_ref = float(model.opt.timestep)

    # quaternion-aware velocities, then stack into [q, v]
    V = differentiate_qpos(model, Q, dt_ref, centered=centered)
    X_ref = np.hstack([Q, V])                      # (N, nq + nv)
    return X_ref, dt_ref


########################################################################
# SOLVED-TRAJECTORY SAVE / LOAD  (.npz schema read by examples/replay.py)
########################################################################

def save_trajectory(path, time, state, input, model, spline_type, reference=None, **extra):
    """Save a solved trajectory to <path> as a .npz; returns path.
        time         (N+1,)      time stamps
        state        (N+1, nx)   solved state trajectory
        input        (N, nu)     solved control
        model        str         path to the MuJoCo XML the solve used
        spline_type  str         how replay draws the inputs ("zero" -> steps, else ramps)
        reference    (N+1, nx)   optional tracking ghost drawn during replay
        **extra                  named arrays/scalars (e.g. `defects`); None-valued ones dropped
    load_trajectory returns only the core fields -- read extras from the raw .npz via np.load."""
    
    # the core schema every consumer expects
    data = {"time": time, "state": state, "input": input,
            "model": model, "spline_type": spline_type}

    # optional fields: the ghost reference, then any caller-supplied extras (None ones dropped)
    if reference is not None:
        data["reference"] = reference
    for key, value in extra.items():
        if value is not None:
            data[key] = value
    np.savez(path, **data)
    return path


def load_trajectory(path):
    """Load a trajectory .npz saved by save_trajectory / an example solve.

    Returns (t, x, u, model, ref, spline_type):
        t           (N+1,)        time stamps
        x           (N+1, nx)     state
        u           (N, nu)       input
        model       str           path to the MuJoCo XML
        ref         (N+1, nx)     tracking reference, or None if absent
        spline_type str           control spline basis; defaults to "zero" for older files,
                                   matching the previous always-stepped rendering.
    """
    # required fields
    data = np.load(path)
    t = data["time"]             # (N+1,)
    x = data["state"]            # (N+1, nx)
    u = data["input"]            # (N, nu)
    model = str(data["model"])   # path to the MuJoCo XML

    # optional fields, absent in older files
    ref = data["reference"] if "reference" in data.files else None  # (N+1, nx), optional
    spline_type = str(data["spline_type"]) if "spline_type" in data.files else "zero"
    return t, x, u, model, ref, spline_type
