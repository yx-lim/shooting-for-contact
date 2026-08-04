##
#
# Model Predictive Control (MPC) Class
#
##

# standard imports
from dataclasses import dataclass
from typing import Optional
import time as _time
import numpy as np

# custom imports
from src.multi_shooting import MultiShootingBase, MultiShootingConfig, fmt_min_sec
from src.dynamics import DynamicsConfig


########################################################################
# CONFIG
########################################################################

@dataclass
class ModelPredictiveControlConfig(MultiShootingConfig):

    # total closed-loop sim time [s]
    T: float = 3.0

    # mpc resolve interval [s]
    mpc_dt: Optional[float] = 0.02

    # print a status line per MPC step
    verbose: bool = True

    # max_iter for the first solve only to seed with a high-quality solution
    max_iter_initial: Optional[int] = None

    # keep best solution of the fixed IPOPT iterations
    keep_best_sol: bool = False


########################################################################
# MODEL PREDICTIVE CONTROL
########################################################################

class ModelPredictiveControl(MultiShootingBase):
    """ Receding-horizon controller. """

    def __init__(self, dynamics_config: DynamicsConfig, config: ModelPredictiveControlConfig):

        # build the horizon NLP (dynamics, bounds, cyipopt problem, etc.)
        super().__init__(dynamics_config, config)

        # number of closed-loop plant steps from the total sim time T; at least one, else run()
        # would loop zero times and divide by a zero solve count when it reports the averages
        self.n_steps = int(round(config.T / self.dt))
        if self.n_steps < 1:
            raise ValueError(
                f"T = {config.T} s is shorter than half a sim step (dt = {self.dt} s), so the "
                f"closed loop would run zero steps; set T >= {self.dt} s."
            )

        # control period -> number of plant steps applied per NLP solve
        mpc_dt = config.mpc_dt if config.mpc_dt is not None else self.dt
        self.steps_per_solve = max(1, int(round(mpc_dt / self.dt)))
        if self.steps_per_solve > self.N_sim:
            raise ValueError(
                f"mpc_dt/sim_dt = {self.steps_per_solve} exceeds the horizon's "
                f"N_sim = {self.N_sim} sim steps; there are not enough planned "
                f"controls to apply between re-solves."
            )
        # with K-step shooting intervals the warm start shifts whole intervals,
        # so the re-solve period must land on the node grid
        if self.K > 1 and self.steps_per_solve % self.K != 0:
            raise ValueError(
                f"mpc_dt = {mpc_dt} must be an integer multiple of node_dt = "
                f"{self.K * self.dt} (steps_per_solve = {self.steps_per_solve}, "
                f"K = {self.K}) so the warm start can shift whole shooting intervals."
            )

        # print a status line per MPC step
        self.verbose = config.verbose

        # first-solve iteration budget
        self.max_iter_initial = config.max_iter_initial
        self._max_iter_steady = (config.ipopt_options or {}).get("max_iter", 3000)

        # sim time [s]
        self.t_now = 0.0

        # warm-start trajectory
        self.X_ws: Optional[np.ndarray] = None
        self.U_ws: Optional[np.ndarray] = None

    ############################################################
    # WARM START
    ############################################################

    def set_warm_start(self, X_ws: np.ndarray, U_ws: np.ndarray) -> None:
        """Set the warm-start trajectory. X_ws may be given on the shooting-node
        grid (N+1, nx) or the sim grid (N_sim+1, nx; subsampled at the nodes);
        U_ws is per sim step (N_sim, nu)."""
        self.X_ws = self.subsample_nodes(X_ws)
        self.U_ws = np.asarray(U_ws, dtype=float).reshape(self.N_sim, self.nu)

    def _shift(self, X: np.ndarray, U: np.ndarray) -> None:
        """Shift the solved trajectory forward by steps_per_solve sim steps for
        the next solve: U shifts on the sim grid, X by whole shooting nodes
        (steps_per_solve/K of them; divisibility is enforced in __init__)."""
        s = self.steps_per_solve
        sn = s // self.K                                              # nodes to drop
        self.X_ws = np.vstack([X[sn:], np.repeat(X[-1:], sn, axis=0)])  # repeat x_N
        self.U_ws = np.vstack([U[s:], np.repeat(U[-1:], s, axis=0)])    # repeat u_last

    ############################################################
    # SOLVE ONE STEP
    ############################################################

    def solve_step(self, x: np.ndarray):
        """Solve the horizon problem from state x and return (u0, X, U, info). """
        x = np.asarray(x, dtype=float)

        # update the x_0 = x_init constraint to the current state
        self.set_initial_state(x)

        # set warm start
        if self.X_ws is None or self.U_ws is None:
            raise RuntimeError(
                "warm start not set; call set_warm_start(X_ws, U_ws) before solving."
            )
        X_ws, U_ws = self.X_ws, self.U_ws
        X_ws[0] = x  # keep the warm start consistent with the measured state

        # solve the horizon NLP (quiet: the per-solve timing is summarized in run())
        X, U, info = self.solve(X_ws, U_ws, verbose=False)

        # shift for the next solve
        self._shift(X, U)

        return U[0], X, U, info
    
    @staticmethod
    def _print_step(k: int, n: int, solve_time: float, info: dict) -> None:
        """Print one MPC step: index, solve time, objective and IPOPT status."""
        # IPOPT status
        status = info.get("status_msg", "")
        if isinstance(status, (bytes, bytearray)):
            status = status.decode()

        # message
        print(f"[mpc {k:4d}/{n}] "
              f"solve = {solve_time:6.1f} ms | "
              f"obj = {info['obj_val']:.4e} | "
              f"status = {status}")


    ############################################################
    # CLOSED-LOOP ROLLOUT
    ############################################################

    def run(self, x0: np.ndarray):
        """Run the closed loop simulation.

        Args:
            x0   : (nx,)            initial plant state
        Returns
            t    : (n_steps+1,)     time stamps
            X_cl : (n_steps+1, nx)  closed-loop states
            U_cl : (n_steps,   nu)  applied controls
        """
        x = np.asarray(x0, dtype=float)
        n = self.n_steps

        # initialize logs
        X_cl = np.zeros((n + 1, self.nx))
        U_cl = np.zeros((n, self.nu))
        X_cl[0] = x

        # wall-clock accounting for the whole closed loop
        run_t0 = _time.perf_counter()
        n_solves = 0
        solve_total = 0.0      # cumulative NLP solve time [ms]

        # give the first solve a (typically larger) iteration budget for a high-quality seed
        if self.max_iter_initial is not None:
            self.problem.add_option("max_iter", int(self.max_iter_initial))
            print(f"Solving warm start with {self.max_iter_initial} IPOPT iterations...")

        # simulate the closed loop, re-solving every steps_per_solve plant steps
        k = 0
        while k < n:

            # sim time at this horizon's start (for references)
            self.t_now = k * self.dt

            # solve the horizon problem at the current state
            t0 = _time.perf_counter()
            _, _, U, info = self.solve_step(x)
            solve_time = 1e3 * (_time.perf_counter() - t0)
            n_solves += 1
            solve_total += solve_time

            # after the first solve, revert to the steady-state iteration cap
            if n_solves == 1 and self.max_iter_initial is not None:
                self.problem.add_option("max_iter", self._max_iter_steady)

            # apply the first steps_per_solve planned controls open-loop
            for j in range(self.steps_per_solve):
                if k >= n:
                    break
                u = U[j]
                x = self.dyn.dynamics(x, u)
                U_cl[k] = u
                X_cl[k + 1] = x
                k += 1

            # print a status line per re-solve
            if self.verbose:
                self._print_step(k, n, solve_time, info)

        # total wall time and per-solve average
        run_total = _time.perf_counter() - run_t0
        print(f"[mpc] simulated {n * self.dt:.2f} s ({n} steps) in "
              f"{fmt_min_sec(run_total)} wall | {n_solves} solves, "
              f"avg {solve_total / n_solves:.1f} ms/solve "
              f"({fmt_min_sec(solve_total / 1e3)} total in the NLP)")

        # time stamps for the n+1 logged states
        t = self.dt * np.arange(n + 1)

        return t, X_cl, U_cl
