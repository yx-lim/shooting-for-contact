#!/usr/bin/env python3
"""Convert a shooting-for-contact/DSMS G1 result to mjlab CSV input.

DSMS input (.npz produced by utils.file_utils.save_trajectory):
    time:  (N,) timestamps
    state: (N, nx), with G1 qpos in state[:, :36]

G1 DSMS qpos layout:
    base_x, base_y, base_z,
    base_qw, base_qx, base_qy, base_qz,
    29 joint positions

mjlab CSV layout:
    base_x, base_y, base_z,
    base_qx, base_qy, base_qz, base_qw,
    29 joint positions

The output CSV is headerless because mjlab loads it with numpy.loadtxt().

python scripts/dsms_to_mjlab.py \
  examples/g1_tracking_mpc/g1_tracking_lunges.npz \
  --output-csv examples/g1_tracking_mpc/g1_tracking_lunges_mjlab.csv \
  --normalize-quaternions

"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_OUTPUT_FPS = 50.0
DEFAULT_MAX_DT_JITTER = 0.02
G1_JOINT_COUNT = 29
G1_QPOS_COUNT = 7 + G1_JOINT_COUNT  # 36
MJLAB_ROW_WIDTH = G1_QPOS_COUNT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a shooting-for-contact/DSMS G1 .npz trajectory to "
            "mjlab's headerless CSV format."
        )
    )
    parser.add_argument(
        "input_npz",
        type=Path,
        help="DSMS .npz produced by save_trajectory().",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        help="Output CSV. Default: <input_stem>_mjlab.csv",
    )
    parser.add_argument(
        "--state-key",
        default="state",
        help="NPZ key containing solved states. Default: state.",
    )
    parser.add_argument(
        "--time-key",
        default="time",
        help="NPZ key containing timestamps. Default: time.",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=None,
        help="Optional FPS override. Default: infer from the DSMS time array.",
    )
    parser.add_argument(
        "--max-dt-jitter",
        type=float,
        default=DEFAULT_MAX_DT_JITTER,
        help=(
            "Maximum allowed relative frame-spacing deviation from median dt. "
            f"Default: {DEFAULT_MAX_DT_JITTER:.0%}."
        ),
    )
    parser.add_argument(
        "--allow-nonuniform-time",
        action="store_true",
        help="Allow nonuniform timestamps and use their median dt.",
    )
    parser.add_argument(
        "--normalize-quaternions",
        action="store_true",
        help="Normalize each base quaternion before writing xyzw.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=DEFAULT_OUTPUT_FPS,
        help=(
            "Output FPS passed to mjlab's csv_to_npz converter. "
            f"Default: {DEFAULT_OUTPUT_FPS:g}."
        ),
    )
    parser.add_argument(
        "--run-converter",
        action="store_true",
        help="Run mjlab.scripts.csv_to_npz after creating the CSV.",
    )
    parser.add_argument(
        "--mjlab-dir",
        type=Path,
        help=(
            "Path to the mjlab repository. Required with --run-converter "
            "unless the current directory is mjlab."
        ),
    )
    parser.add_argument(
        "--output-name",
        help="Motion name passed to mjlab. Default: output CSV stem.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device passed to mjlab's converter. Default: cuda:0.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Ask mjlab's converter to render the motion.",
    )
    return parser


def load_dsms_npz(
    path: Path,
    *,
    state_key: str,
    time_key: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input NPZ does not exist: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = tuple(data.files)
            if state_key not in data:
                raise ValueError(
                    f"Missing {state_key!r}. Available NPZ keys: {keys}"
                )
            if time_key not in data:
                raise ValueError(
                    f"Missing {time_key!r}. Available NPZ keys: {keys}"
                )
            states = np.asarray(data[state_key], dtype=float)
            times = np.asarray(data[time_key], dtype=float).reshape(-1)
    except OSError as exc:
        raise ValueError(f"Could not open {path}: {exc}") from exc

    if states.ndim != 2:
        raise ValueError(
            f"{state_key!r} must be 2D; found shape {states.shape}."
        )
    if states.shape[0] != times.shape[0]:
        raise ValueError(
            f"State/time length mismatch: {states.shape[0]} states and "
            f"{times.shape[0]} timestamps."
        )
    if states.shape[0] == 0:
        raise ValueError("The DSMS trajectory is empty.")
    if states.shape[1] < G1_QPOS_COUNT:
        raise ValueError(
            f"Expected at least {G1_QPOS_COUNT} state columns for G1 qpos; "
            f"found {states.shape[1]}."
        )
    if not np.all(np.isfinite(states)):
        frame, column = np.argwhere(~np.isfinite(states))[0]
        raise ValueError(
            f"Non-finite state value at frame {frame + 1}, column {column + 1}."
        )
    if not np.all(np.isfinite(times)):
        frame = int(np.flatnonzero(~np.isfinite(times))[0])
        raise ValueError(f"Non-finite timestamp at frame {frame + 1}.")

    return times, states, keys


def detect_input_fps(
    times: Sequence[float],
    *,
    manual_fps: float | None,
    max_dt_jitter: float,
    allow_nonuniform_time: bool,
) -> tuple[float, float, float]:
    if manual_fps is not None and (
        manual_fps <= 0 or not math.isfinite(manual_fps)
    ):
        raise ValueError("--input-fps must be a positive finite number.")
    if max_dt_jitter < 0 or not math.isfinite(max_dt_jitter):
        raise ValueError("--max-dt-jitter must be non-negative and finite.")

    if len(times) < 2:
        if manual_fps is None:
            raise ValueError(
                "At least two frames are needed to infer FPS. "
                "Provide --input-fps for a single-frame trajectory."
            )
        return manual_fps, 1.0 / manual_fps, 0.0

    values = [float(value) for value in times]
    deltas = [b - a for a, b in zip(values, values[1:])]
    if any(dt <= 0 for dt in deltas):
        raise ValueError("DSMS time values must be strictly increasing.")

    median_dt = statistics.median(deltas)
    detected_fps = 1.0 / median_dt
    max_relative_jitter = max(
        abs(dt - median_dt) / median_dt for dt in deltas
    )

    if max_relative_jitter > max_dt_jitter:
        message = (
            f"Timestamp spacing is nonuniform: median dt={median_dt:.9g} s "
            f"({detected_fps:.6g} Hz), maximum relative deviation="
            f"{max_relative_jitter:.2%}, allowed={max_dt_jitter:.2%}."
        )
        if not allow_nonuniform_time:
            raise ValueError(
                message + " Pass --allow-nonuniform-time to use the median dt."
            )
        print(f"[WARN] {message}", file=sys.stderr)

    if manual_fps is not None:
        manual_dt = 1.0 / manual_fps
        if abs(manual_dt - median_dt) / median_dt > 0.02:
            print(
                (
                    f"[WARN] DSMS timestamps indicate {detected_fps:.6g} Hz, "
                    f"but --input-fps={manual_fps:.6g} was supplied. "
                    "The override will be used."
                ),
                file=sys.stderr,
            )
        return manual_fps, median_dt, max_relative_jitter

    return detected_fps, median_dt, max_relative_jitter


def convert_states(
    states: np.ndarray,
    *,
    normalize_quaternions: bool,
) -> tuple[list[list[float]], int]:
    rows: list[list[float]] = []
    non_unit_count = 0

    for frame_index, qpos in enumerate(states[:, :G1_QPOS_COUNT], start=1):
        # DSMS/MuJoCo: xyz + qw,qx,qy,qz + joints
        xyz = [float(value) for value in qpos[0:3]]
        qw, qx, qy, qz = [float(value) for value in qpos[3:7]]
        quat_xyzw = [qx, qy, qz, qw]

        norm = math.sqrt(sum(value * value for value in quat_xyzw))
        if norm < 1e-12:
            raise ValueError(
                f"Frame {frame_index}: base quaternion has zero magnitude."
            )
        if abs(norm - 1.0) > 1e-3:
            non_unit_count += 1
        if normalize_quaternions:
            quat_xyzw = [value / norm for value in quat_xyzw]

        joints = [float(value) for value in qpos[7:36]]
        output_row = xyz + quat_xyzw + joints
        if len(output_row) != MJLAB_ROW_WIDTH:
            raise RuntimeError("Internal mjlab row-width check failed.")
        rows.append(output_row)

    return rows, non_unit_count


def write_headerless_csv(
    path: Path,
    rows: Sequence[Sequence[float]],
) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        for row in rows:
            writer.writerow([f"{value:.10g}" for value in row])
    return path


def resolve_mjlab_dir(requested: Path | None) -> Path:
    candidate = (requested or Path.cwd()).expanduser().resolve()
    if not (candidate / "pyproject.toml").exists() or not (
        candidate / "src" / "mjlab"
    ).exists():
        raise ValueError(
            f"{candidate} does not look like the mjlab repository. "
            "Pass its path using --mjlab-dir."
        )
    return candidate


def run_mjlab_converter(
    *,
    mjlab_dir: Path,
    csv_path: Path,
    input_fps: float,
    output_fps: float,
    output_name: str,
    device: str,
    render: bool,
) -> None:
    if output_fps <= 0 or not math.isfinite(output_fps):
        raise ValueError("--output-fps must be a positive finite number.")

    command = [
        "uv",
        "run",
        "-m",
        "mjlab.scripts.csv_to_npz",
        "--input-file",
        str(csv_path),
        "--output-name",
        output_name,
        "--input-fps",
        f"{input_fps:.12g}",
        "--output-fps",
        f"{output_fps:.12g}",
        "--device",
        device,
        "--render",
        "True" if render else "False",
    ]
    print("\nRunning mjlab converter:")
    print("  " + " ".join(command))
    subprocess.run(command, cwd=mjlab_dir, check=True)


def main() -> int:
    args = build_parser().parse_args()
    output_csv = args.output_csv or args.input_npz.with_name(
        f"{args.input_npz.stem}_mjlab.csv"
    )

    times, states, keys = load_dsms_npz(
        args.input_npz,
        state_key=args.state_key,
        time_key=args.time_key,
    )
    input_fps, median_dt, max_jitter = detect_input_fps(
        times,
        manual_fps=args.input_fps,
        max_dt_jitter=args.max_dt_jitter,
        allow_nonuniform_time=args.allow_nonuniform_time,
    )
    rows, non_unit_count = convert_states(
        states,
        normalize_quaternions=args.normalize_quaternions,
    )
    output_csv = write_headerless_csv(output_csv, rows)

    print(f"Converted {len(rows)} frames.")
    print(f"Input:  {args.input_npz.expanduser().resolve()}")
    print(f"Output: {output_csv}")
    print(f"NPZ keys: {', '.join(keys)}")
    print(f"DSMS state shape: {states.shape}")
    print("Extracted G1 qpos: state[:, :36]")
    print(f"Detected median dt: {median_dt:.9g} s")
    print(f"Input frequency supplied to mjlab: {input_fps:.9g} Hz")
    print(f"mjlab output frequency: {args.output_fps:g} Hz")
    print("Output layout: xyz + quaternion xyzw + 29 G1 joints")
    print("Output CSV is headerless.")

    if max_jitter > args.max_dt_jitter:
        print(
            f"[WARN] Maximum timestamp jitter was {max_jitter:.2%}.",
            file=sys.stderr,
        )
    if non_unit_count:
        action = "were normalized" if args.normalize_quaternions else "were unchanged"
        print(
            f"[WARN] {non_unit_count} quaternion(s) differed from unit length "
            f"by more than 1e-3 and {action}.",
            file=sys.stderr,
        )

    output_name = args.output_name or output_csv.stem
    if args.run_converter:
        run_mjlab_converter(
            mjlab_dir=resolve_mjlab_dir(args.mjlab_dir),
            csv_path=output_csv,
            input_fps=input_fps,
            output_fps=args.output_fps,
            output_name=output_name,
            device=args.device,
            render=args.render,
        )
    else:
        print("\nNext command from inside the mjlab repository:")
        print(
            "  uv run -m mjlab.scripts.csv_to_npz"
            f" --input-file {output_csv}"
            f" --output-name {output_name}"
            f" --input-fps {input_fps:.9g}"
            f" --output-fps {args.output_fps:g}"
            " --render True"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
