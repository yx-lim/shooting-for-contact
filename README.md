<div align="center">

# Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting

[📄 **Paper**](https://shooting-for-contact.github.io/) &nbsp;·&nbsp; [🌐 **Project Page**](https://shooting-for-contact.github.io/)

</div>

Reference implementation of Direct Simulation-based Multiple Shooting (DSMS): trajectory optimization and MPC with MuJoCo in the loop, solved as a multiple-shooting NLP with IPOPT. Dynamics, derivatives, and contact all come from MuJoCo simulator.

---

## Installation
The prerequisites are [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) (Miniconda is enough) and `make` — everything else, including IPOPT and its bundled MUMPS solver, is installed into the env. 

The Python environment is defined in [`environment.yml`](environment.yml). From the repo root, run:
```bash
make install
```

This creates an env named `dsms` and sets the `TRAJOPT_ROOT_DIR` environment variable. Then activate it:
```bash
conda activate dsms
```

To uninstall the environment:
```bash
conda deactivate # if you are currently in the env
make uninstall
```

---

## Repository Layout
```
src/              
  multi_shooting.py   the NLP: decision vector, defect constraints, IPOPT callbacks
  mpc.py              receding-horizon wrapper around it
  dynamics.py         MuJoCo wrapper: rollouts, finite-difference Jacobians, manifold state ops
  spline.py           control parametrization (zero-order hold / linear)
  end_effector.py     Cartesian body tracking (position, orientation, velocity)

utils/            math_utils.py (quaternion-aware state ops), file_utils.py (clip + .npz I/O)
models/           MuJoCo XMLs: unitree_g1, unitree_go2, cartpole, triple_cartpole, hopper
trajectories/     reference clips under g1/ and go2/, plus four scripts that make and inspect them
examples/         one folder per problem; each solves a motion and writes a .npz beside itself
```

---

## Running The Examples
With the env active, run any example — it solves a motion and saves a `.npz` in its folder, which you can then replay in the viewer (see below).

Each block below is the solve followed by its replay. Examples are setup in `__main__` of each example file.

### Toy Systems
```bash
# cartpole swing-up: one-shot trajectory optimization, hard terminal constraint
python examples/cartpole/cartpole.py
python examples/replay.py cartpole

# cartpole swing-up: the same task under a receding-horizon MPC
python examples/cartpole_mpc/cartpole_mpc.py
python examples/replay.py cartpole_mpc

# triple cartpole swing-up: three chained links, MPC
python examples/triple_cartpole_mpc/triple_cartpole_mpc.py
python examples/replay.py triple_cartpole_mpc

# hopper: velocity-tracking hop over flat ground, MPC
python examples/hopper_mpc/hopper_mpc.py
python examples/replay.py hopper_mpc
```

### Unitree Go2 Quadruped
```bash
# reference tracking with a receding-horizon MPC   
#   motions: hopturn | pronking
python examples/go2_tracking_mpc/go2_tracking_mpc.py hopturn
python examples/replay.py go2_tracking_hopturn --speed 0.5
```

### Unitree G1 Humanoid
```bash
# squat down + reach hands forward
python examples/g1_squat_mpc/g1_squat_mpc.py
python examples/replay.py squat

# gait: one-shot trajectory optimization over a whole periodic clip
#   motions: walk_fwd | run_fwd | run_bck | crawl_fwd
python examples/g1_gait/g1_gait.py run_fwd
python examples/replay.py g1_gait_run_fwd --speed 0.5

# reference tracking with receding-horizon replanning, same template as the Go2
#   motions: jump | crawl_fwd | crawl_bck | crawl_turn
python examples/g1_tracking_mpc/g1_tracking_mpc.py crawl_fwd
python examples/replay.py g1_tracking_crawl_fwd
```

Here, `actuator_mode`, in both tracking examples, picks how the control is interpreted: `"torque"`
converts the XML's PD servos to normalized `<motor>` actuators (u is a torque in [-1,1]),
`"position"` keeps them (u is a joint-angle target). Same cost and closed loop either way.

### Replaying a result
`examples/replay.py` opens the MuJoCo viewer and plots states and inputs. The robot examples write `<robot>_<example>_<motion>.npz` and the toy ones are named after their folder, so the bare name is always enough — no folder path needed. A translucent reference "ghost" is overlaid automatically whenever the result carries one.

Flags: `--speed` (playback rate), `--no-plot`, `--no-ghost`, `--axis` (world-frame RGB triad). In the viewer: `Space` pause, `Left`/`Right` step ±1 frame, `Up`/`Down` step ±10, `R` restart, `F` toggle the contact points and contact-force arrows.
```bash
python examples/replay.py g1_gait_run_fwd --speed 0.1
```

---

## Recommended: HSL linear solvers for IPOPT
Each IPOPT iteration factorizes a large sparse KKT system, and that factorization is where a big chunk of the runtime goes. Swapping the bundled MUMPS for an HSL solver is a big speedup for the larger problems (e.g., humanoid).

HSL is free for academic use. License and download `coinhsl-x.y.z.tar.gz` from the [HSL website](https://licences.stfc.ac.uk/product/coin-hsl), then build it with [Meson](https://mesonbuild.com):
```bash
pip install --user meson ninja            # skip if you have them; needs ~/.local/bin on PATH
tar -xzf ~/Downloads/coinhsl-2023.11.17.tar.gz && cd coinhsl-2023.11.17
meson setup builddir --prefix=$HOME/hsl && meson compile -C builddir && meson install -C builddir
echo 'export LD_LIBRARY_PATH=$HOME/hsl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
```
Open a new shell, then set the solver in the example's `config.py` — `hsllib` is passed
automatically for any `ma*` solver, so this is the only edit:
```python
linear_solver: str = "ma57"   # "mumps" | "ma27" small dense | "ma57" general | "ma97" large sparse
```

---

## Citation
If you find this work useful, please consider citing it:
```bibtex
@article{esteban2026shooting,
  title={Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting},
  author={Sergio A. Esteban and Jason H. K. Siu and Derrick Mach and Junheng Li and Vince Kurtz and Joel W. Burdick and Aaron D. Ames},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
