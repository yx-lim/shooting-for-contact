##
#
# Multiple Shooting NLP Class
#    for MuJoCo dynamics using IPOPT via cyipopt.
#    Reference: https://cyipopt.readthedocs.io/stable/
#
##

# for abstract classes
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import time as _time

# standard imports
import numpy as np

# python with Interior-Point OPTimizer (IPOPT)
import cyipopt

# custom imports
from src.dynamics import Dynamics, DynamicsConfig
from src.spline import SplineConfig, make_spline


########################################################################
# HELPER FUNCTIONS
########################################################################

def fmt_min_sec(seconds: float) -> str:
    """Format a duration in seconds as 'M min, S.S sec' (e.g. 75.3 -> '1 min, 15.3 sec')."""
    m = int(seconds // 60)
    return f"{m} min, {seconds - 60 * m:.1f} sec"


########################################################################
# CONFIG
########################################################################

@dataclass
class MultiShootingConfig:

    # number of shooting intervals
    N: int

    # shooting-node spacing [s]
    node_dt: Optional[float] = None

    # impose the hard terminal constraint x_N = x_goal.
    # The goal value itself is declared on the subclass (self.x_goal), not here.
    terminal_constraint: bool = False

    # state constraints
    x_lb: Optional[object] = None
    x_ub: Optional[object] = None

    # input constraints. None -> default to the model's control range (dyn.u_lb / dyn.u_ub);
    # pass an explicit scalar/vector to override (e.g. force actuators with no ctrlrange).
    u_lb: Optional[object] = None
    u_ub: Optional[object] = None

    # optional spline parametrization of the controls, u = h(p), with p the
    # decision variables. None -> per-step controls (ZOH with M = N).
    spline: Optional[SplineConfig] = None

    # additional options to pass to IPOPT solver (e.g. "linear_solver": "ma57")
    ipopt_options: Optional[dict] = None

    # keep the best iterate seen (min of the merit obj + keep_best_sol_rho * inf_pr) and return it
    # instead of IPOPT's LAST iterate; large rho weights feasibility (obj as tiebreak).
    keep_best_sol: bool = True
    keep_best_sol_rho: float = 1e2


########################################################################
# OPTIMIZATION CONFIG
########################################################################

class MultiShootingBase(ABC):
    """ Multiple-shooting NLP for MuJoCo dynamics, K = node_dt / sim_dt integration steps per
    interval (K = 1 if node_dt is None). Variables z = [X_0..X_N, p], with u_k = h(p, t_k).

        min  dt * sum_{k=0}^{N*K-1} l(z_k, u_k, k) + l_f(X_N)
        s.t. X_{i+1} = Phi_K(X_i, p)   i = 0..N-1, the K-step flow map
             X_0 = x_init,  X_N = x_goal (optional),  u_lb <= u_k <= u_ub

    Stage costs stay on the FINE sim grid (stage k IS the global sim step), so a subclass's cost
    callbacks never depend on K. Defects use a forward sensitivity recursion, gradients a
    backward adjoint pass -- at K = 1 both collapse to one node per step. """

    def __init__(self, dynamics_config: DynamicsConfig, config: MultiShootingConfig):
        
        self.cfg = config

        # build the dynamics config
        self.dyn = Dynamics(dynamics_config)
        self.N = config.N
        self.nx = self.dyn.nx     # full state size [qpos, qvel]
        self.ndx = self.dyn.ndx   # tangent state size 2*nv (defect rows)
        self.nu = self.dyn.nu

        # floating base: defects live in the tangent space and the free-joint
        # quaternion needs a unit-norm constraint per node to pin its gauge.
        self.is_floating = self.dyn.has_3d_floating_base
        self.quat_addr = self.dyn.quat_addr          # qpos index of the base quaternion

        # discrete timestep; stage (running) costs are scaled by dt so the
        # objective approximates the integral cost int l(x,u) dt + l_f(x_N).
        self.dt = self.dyn.model.opt.timestep

        # K internal integration steps per shooting interval (node_dt = K * dt). K shrinks the
        # NLP, not the physics (one FD Jacobian per sim step either way); 2-10 is the useful range.
        if config.node_dt is None:
            self.K = 1
        else:
            K = int(round(config.node_dt / self.dt))
            if K < 1 or abs(config.node_dt - K * self.dt) > 1e-9:
                raise ValueError(
                    f"node_dt = {config.node_dt} must be a positive integer "
                    f"multiple of the sim timestep dt = {self.dt}")
            self.K = K
        self.N_sim = self.N * self.K     # total sim steps over the horizon

        # initial condition x_0 = x_init; NOT in the config -- must be set via
        # set_initial_state(x0) before each solve (re-set every MPC step).
        self.x_init: Optional[np.ndarray] = None

        # terminal target x_N = x_goal; declared by the subclass (e.g. for the
        # cost). Only enforced as a hard constraint if config.terminal_constraint.
        self.x_goal: Optional[np.ndarray] = None

        # control parametrization U = spline.evaluate(p), one control per sim step (N_sim total).
        # config.spline=None -> ZOH with M = N knots, i.e. one constant control per interval.
        if config.spline is None:
            self.spline = make_spline(SplineConfig(M=self.N, spline_type="zero"),
                                      self.N_sim, self.nu)
        else:
            self.spline = make_spline(config.spline, self.N_sim, self.nu)
        self.n_params = self.spline.dim          # size of the control block p

        # sparse control Jacobian structure: per shooting interval, the union of
        # the columns of H touched by its K internal steps (contiguous knots).
        H0 = self.spline.jacobian(np.zeros(self.n_params))      # (N_sim*nu, n_params)
        self._p_spans = []
        Knu = self.K * self.nu
        for i in range(self.N):
            nz = np.flatnonzero(np.any(H0[i * Knu:(i + 1) * Knu, :] != 0.0, axis=0))
            if nz.size == 0:                                    # degenerate: keep 1 col
                self._p_spans.append((0, 1))
            else:
                self._p_spans.append((int(nz[0]), int(nz[-1] - nz[0] + 1)))

        # decision variable node sizes:  z = [x_0..x_N, p]
        self.n_states = (self.N + 1) * self.nx
        self.n_vars = self.n_states + self.n_params

        # Optional terminal constraint x_N = x_goal.
        self.has_terminal = config.terminal_constraint

        # Equality-constraint rows, all tangent-space residuals of width nt, laid out as
        # [initial | defect_0..N-1 | terminal? | norm_0..N?]; the last two blocks are optional.
        nt = self.ndx
        self.n_constraints = (self.N + 1) * nt              # initial + N defects
        self._row_terminal = (self.N + 1) * nt              # where the terminal block sits
        if self.has_terminal:
            self.n_constraints += nt
        self._row_norm = self.n_constraints                 # norm constraints appended last
        if self.is_floating:
            self.n_constraints += (self.N + 1)

        # Sparse constraint-Jacobian structure, block-banded since a defect row touches only
        # x_k, x_{k+1} and p. Declared once as blocks of (row0, nrows, col0, ncols).
        blocks = [(0, nt, 0, self.nx)]                            # initial: dc_0/dx_0
        for k in range(self.N):
            row = nt + k * nt
            blocks.append((row, nt, k * self.nx, self.nx))        # dc_k/dx_k
            blocks.append((row, nt, (k + 1) * self.nx, self.nx))  # dc_k/dx_{k+1}
            c0, nc = self._p_spans[k]                             # dc_k/dp: only the knots
            blocks.append((row, nt, self.n_states + c0, nc))      # supporting step k
        if self.has_terminal:
            blocks.append((self._row_terminal, nt, self.N * self.nx, self.nx))
        if self.is_floating:
            assert self.quat_addr is not None
            for k in range(self.N + 1):
                blocks.append((self._row_norm + k, 1, k * self.nx + self.quat_addr, 4))

        # expand blocks into (row, col) index arrays in row-major order per block
        rows, cols = [], []
        for (r0, nr, c0, nc) in blocks:
            rows.append(np.repeat(np.arange(r0, r0 + nr), nc))
            cols.append(np.tile(np.arange(c0, c0 + nc), nr))
        self._jac_rows = np.concatenate(rows).astype(int)
        self._jac_cols = np.concatenate(cols).astype(int)

        # Input constraints as box bounds on the control points p; a bound the config leaves
        # unset falls back to the model's own ctrlrange (dyn.u_lb / dyn.u_ub).
        lb = -np.inf * np.ones(self.n_vars)
        ub = np.inf * np.ones(self.n_vars)
        u_lb = self._bound_vec(config.u_lb, self.nu)
        u_ub = self._bound_vec(config.u_ub, self.nu)
        if u_lb is None:
            u_lb = self.dyn.u_lb
        if u_ub is None:
            u_ub = self.dyn.u_ub
        lb[self.p_slice] = np.tile(u_lb, self.spline.M)
        ub[self.p_slice] = np.tile(u_ub, self.spline.M)

        # State constraints as box bounds on nodes x_1..x_N only; x_0 stays free because the
        # initial-condition row already pins it to x_init.
        x_lb = self._bound_vec(config.x_lb, self.nx)
        x_ub = self._bound_vec(config.x_ub, self.nx)
        if x_lb is not None:
            lb[self.nx:self.n_states] = np.tile(x_lb, self.N)
        if x_ub is not None:
            ub[self.nx:self.n_states] = np.tile(x_ub, self.N)

        # cache the decision-variable bounds so a subclass that appends custom constraint
        # rows can rebuild the cyipopt problem (with a larger m) without re-deriving them.
        self._lb, self._ub = lb, ub

        # Equality constraints: g(z) = 0
        cl = np.zeros(self.n_constraints)
        cu = np.zeros(self.n_constraints)

        # build IPOPT problem
        self.problem = cyipopt.Problem(
            n=self.n_vars,        # num decision variables
            m=self.n_constraints, # num constraints
            problem_obj=self,     # problem object with objective, gradient, constraints, jacobian, (and hessian)
            lb=lb,                # lower bounds on decision variables
            ub=ub,                # upper bounds on decision variables
            cl=cl,                # lower bounds on constraints
            cu=cu,                # upper bounds on constraints
        )

        # build the options dict for IPOPT
        options = config.ipopt_options or {}
        for name, value in options.items():
            self.problem.add_option(name, value)

        # forward-pass caches keyed on z: cyipopt calls (objective, constraints) and (gradient,
        # jacobian) separately at the SAME point, so each rollout / FD pass is computed once.
        self._roll_z: Optional[np.ndarray] = None
        self._roll_Z: Optional[np.ndarray] = None
        self._fd_z: Optional[np.ndarray] = None
        self._fd_ZAB: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


    ############################################################
    # INITIAL CONDITION
    ############################################################

    def set_initial_state(self, x_init: np.ndarray) -> None:
        """Set the initial-condition constraint x_0 = x_init.

        Must be called before solve(). constraints() reads self.x_init, so
        re-solving from a new state (e.g. each MPC step) is just calling this
        again before the next solve().
        """
        self.x_init = np.asarray(x_init, dtype=float).reshape(self.nx)

    ############################################################
    # COST
    ############################################################

    @abstractmethod
    def stage_cost(self, x: np.ndarray, u: np.ndarray, k: int) -> float:
        """Running cost l(x_k, u_k, k)."""
        ...

    @abstractmethod
    def stage_cost_grad(self, x: np.ndarray, u: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gradients (dl/dx, dl/du) of the running cost; shapes (nx,), (nu,)."""
        ...

    @abstractmethod
    def terminal_cost(self, x: np.ndarray) -> float:
        """Terminal cost l_f(x_N)."""
        ...

    @abstractmethod
    def terminal_cost_grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient dl_f/dx of the terminal cost; shape (nx,)."""
        ...

    def prepare_stage_cache(self, S: np.ndarray) -> None:
        """Optional subclass hook, called at the start of objective() and
        gradient() with ALL stage states S (N_sim+1, nx): rows 0..N_sim-1 are the
        running-cost states (stage k = row k) and row N_sim is the terminal node.
        Subclasses whose costs need expensive state-dependent quantities (e.g.
        tracked-body FK) can batch-precompute them here and serve
        stage_cost/terminal_cost from the cache. Default: no-op."""
        return

    ############################################################
    # PACKING / UNPACKING DECISION VARIABLES
    ############################################################

    def unpack(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        z = [x_0, ..., x_N, p]   ->   (X, U, p)
        where p are the control parameters and U = spline.evaluate(p) are the
        per-SIM-STEP controls (N_sim, nu); X are the shooting nodes (N+1, nx).
        """
        X = z[:self.n_states].reshape(self.N + 1, self.nx)
        p = z[self.n_states:]                      # control parameters (n_params,)
        U = self.spline.evaluate(p)                # reconstructed controls (N_sim, nu)
        return X, U, p

    def pack(self, X: np.ndarray, p: np.ndarray) -> np.ndarray:
        """ Pack states X and control parameters p into a single vector z."""
        return np.concatenate([X.reshape(-1), np.asarray(p).reshape(-1)])


    @staticmethod
    def _bound_vec(v, n: int):
        """Broadcast a scalar/vector bound to length n; pass through None."""
        if v is None:
            return None
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        return arr.reshape(n)

    def x_slice(self, k: int) -> slice:
        """ get slice for x_k """
        start = k * self.nx
        stop = (k + 1) * self.nx
        return slice(start, stop)

    @property
    def p_slice(self) -> slice:
        """ slice of the control-parameter block p in the decision vector z """
        return slice(self.n_states, self.n_states + self.n_params)

    ############################################################
    # FORWARD PASSES (cached per evaluation point)
    ############################################################

    def _rollout(self, z: np.ndarray) -> np.ndarray:
        """Interval rollouts Z (N, K+1, nx) at z, cached. Z[i, 0] = X_i and
        Z[i, j+1] = f(Z[i, j], u_{iK+j}); used by objective() and constraints()."""
        if self._roll_z is not None and np.array_equal(z, self._roll_z):
            return self._roll_Z
        if self._fd_z is not None and np.array_equal(z, self._fd_z):
            return self._fd_ZAB[0]               # FD pass already rolled out here
        X, U, _ = self.unpack(z)
        Z = self.dyn.rollout_batch(X[:self.N], U.reshape(self.N, self.K, self.nu))
        self._roll_z = np.array(z, dtype=float, copy=True)
        self._roll_Z = Z
        return Z

    def _rollout_fd(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interval rollouts plus per-internal-step FD Jacobians (Z, A, B) at z,
        cached; used by gradient() and jacobian(). This threaded FD pass is the
        dominant cost of the NLP solve (see Dynamics.n_threads)."""
        if self._fd_z is not None and np.array_equal(z, self._fd_z):
            return self._fd_ZAB
        X, U, _ = self.unpack(z)
        Z, A, B = self.dyn.rollout_jacobian_batch(
            X[:self.N], U.reshape(self.N, self.K, self.nu))
        self._fd_z = np.array(z, dtype=float, copy=True)
        self._fd_ZAB = (Z, A, B)
        return self._fd_ZAB

    ############################################################
    # OBJECTIVE
    ############################################################

    def objective(self, z: np.ndarray) -> float:
        """Returns the scalar value of the objective given z."""
        # unpack decision variables (the control params p are not needed here)
        X, U, _ = self.unpack(z)

        # K = 1: every cost point is already a decision node, so no rollout is needed. Kept
        # byte-for-byte identical to the pre-K-step code (see the note in Dynamics.dynamics_batch).
        if self.K == 1:
            self.prepare_stage_cache(X)            # no-op unless a subclass hooks it
            cost = 0.0
            for k in range(self.N):
                cost += self.stage_cost(X[k], U[k], k)
            cost *= self.dt
            cost += self.terminal_cost(X[-1])
            return float(cost)

        # K > 1: cost every sim step, stage k = i*K + j at the internal state Z[i, j]. The
        # interval end Z[i, K] is skipped -- it is node X_{i+1}, costed as its own stage.
        Z = self._rollout(z)                       # (N, K+1, nx), Z[i, 0] = X_i
        S = np.concatenate([Z[:, :self.K].reshape(self.N_sim, self.nx), X[-1:]])
        self.prepare_stage_cache(S)                # no-op unless a subclass hooks it
        cost = 0.0
        for i in range(self.N):
            for j in range(self.K):
                k = i * self.K + j
                cost += self.stage_cost(Z[i, j], U[k], k)
        cost *= self.dt
        cost += self.terminal_cost(X[-1])

        return float(cost)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Returns the gradient of the objective with respect to z.

        Stage costs at interval-internal states z_j depend on (X_i, p) through
        the rollout, so their gradients are chained through the per-step
        sensitivities with a backward adjoint pass over each interval:
            w_j = v_j + A_j^T w_{j+1},   v_j = Q(z_j)^T lx_j   (tangent grad)
        giving  dJ/dp  += H_j^T (lu_j + B_j^T w_{j+1})  per step, and
                dJ/dX_i = lx_0 + Q_inv(X_i)^T (A_0^T w_1)  at the node.
        At K = 1 the chain is empty and this reduces to the per-node gradient.
        """
        X, U, p = self.unpack(z)
        H = self.spline.jacobian(p)                # (N_sim*nu, n_params)

        # K = 1: no interval interior, so no adjoint chaining -- the per-node gradient is exact
        # on its own. Skips the dynamics pass entirely, in the pre-K-step operation order.
        if self.K == 1:
            self.prepare_stage_cache(X)            # no-op unless a subclass hooks it
            grad = np.zeros(self.n_vars)
            L_u = np.zeros((self.N, self.nu))
            for k in range(self.N):
                lx, lu = self.stage_cost_grad(X[k], U[k], k)
                grad[self.x_slice(k)] += self.dt * lx
                L_u[k] = self.dt * lu
            grad[self.x_slice(self.N)] += self.terminal_cost_grad(X[-1])
            grad[self.p_slice] = H.T @ L_u.reshape(-1)
            return grad

        Z, A_all, B_all = self._rollout_fd(z)
        S = np.concatenate([Z[:, :self.K].reshape(self.N_sim, self.nx), X[-1:]])
        self.prepare_stage_cache(S)                # no-op unless a subclass hooks it

        grad = np.zeros(self.n_vars)
        g_p = np.zeros(self.n_params)
        K, nu, dt = self.K, self.nu, self.dt

        for i in range(self.N):
            c0, nc = self._p_spans[i]              # knot columns of interval i
            w = np.zeros(self.ndx)                 # adjoint w_{j+1}, w_K = 0
            for j in range(K - 1, -1, -1):
                k = i * K + j
                lx, lu = self.stage_cost_grad(Z[i, j], U[k], k)
                Hj = H[k * nu:(k + 1) * nu, c0:c0 + nc]
                g_p[c0:c0 + nc] += dt * (Hj.T @ (lu + B_all[i, j].T @ w))
                if j > 0:
                    # tangent-space cost gradient at the internal state z_j
                    v = self.dyn.full_grad_to_tangent(Z[i, j], lx)
                    w = v + A_all[i, j].T @ w
                else:
                    # node state: direct full-state term + chained internal terms
                    gx = lx + self.dyn.tangent_grad_to_full(X[i], A_all[i, 0].T @ w)
                    grad[self.x_slice(i)] += dt * gx
        grad[self.x_slice(self.N)] += self.terminal_cost_grad(X[-1])
        grad[self.p_slice] = g_p

        return grad

    ############################################################
    # CONSTRAINTS
    ############################################################

    def constraints(self, z: np.ndarray) -> np.ndarray:
        """ Constraints g(z) = 0. Residuals are tangent-space differences (state_diff),
        which reduce to plain subtraction when there is no floating base. """
        # unpack decision variables (the control params p are not needed here)
        X, U, _ = self.unpack(z)

        # initialize constraint vector
        g = np.zeros(self.n_constraints)
        nt = self.ndx

        # Initial condition constraint: x_0 ⊖ x_init = 0
        g[0:nt] = self.dyn.state_diff(X[0], self.x_init)

        # Dynamics defects x_{i+1} ⊖ Phi_K(x_i, p) = 0, Phi_K being interval i's K-step flow.
        # Intervals are independent, so the whole set is rolled out in one threaded batch.
        if self.K == 1:
            F = self.dyn.dynamics_batch(X[:self.N], U)
            for k in range(self.N):
                row = nt + k * nt
                g[row:row + nt] = self.dyn.state_diff(X[k + 1], F[k])
        else:
            Z = self._rollout(z)
            for i in range(self.N):
                row = nt + i * nt
                g[row:row + nt] = self.dyn.state_diff(X[i + 1], Z[i, self.K])

        # Terminal constraint: x_N ⊖ x_goal = 0
        if self.has_terminal:
            term = self._row_terminal
            g[term:term + nt] = self.dyn.state_diff(X[-1], self.x_goal)

        # Quaternion unit-norm constraints: ||quat(x_k)||^2 - 1 = 0 (floating base only)
        if self.is_floating:
            a = self.quat_addr
            for k in range(self.N + 1):
                quat = X[k][a:a + 4]
                g[self._row_norm + k] = quat @ quat - 1.0

        return g

    def jacobianstructure(self):
        """ Sparsity pattern (row, col) of the constraint Jacobian; see __init__.
        cyipopt calls this once; jacobian() returns the values in this same order. """
        return (self._jac_rows, self._jac_cols)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """ Nonzero values of the constraint Jacobian, in the block order declared by
        jacobianstructure(): [initial | per-k (dx_k, dx_{k+1}, dp) | terminal? | norm?].

        For the defect c_i = x_{i+1} ⊖ Phi_K(x_i, p), the flow sensitivities follow the forward
        recursion S^x_{j+1} = A_j S^x_j, S^p_{j+1} = A_j S^p_j + B_j H_j (A_j, B_j from
        mjd_transitionFD). With (J1, J2) = state_diff_jacobian(x_{i+1}, z_K), the blocks are
            dc_i/dx_i     = J2 @ S^x_K @ Q_inv(x_i)      (= -A  at K=1, fixed base)
            dc_i/dx_{i+1} = J1 @ Q_inv(x_{i+1})          (=  I  with no floating base)
            dc_i/dp       = J2 @ S^p_K                   (= -B @ H_i at K=1)
        The threaded FD pass over all internal steps dominates the solve (Dynamics.n_threads). """
        # unpack decision variables
        X, U, p = self.unpack(z)

        # control parametrization Jacobian H = dvec(U)/dvec(p), (N_sim*nu, n_params)
        H = self.spline.jacobian(p)

        # collect each block's value matrix in the SAME order as the blocks declared in jacobianstructure()
        mats = []

        # Initial condition: c_0 = x_0 ⊖ x_init,  dc_0/dx_0 = J1(x_0, x_init) @ Q_inv(x_0)
        assert self.x_init is not None
        J1_0, _ = self.dyn.state_diff_jacobian(X[0], self.x_init)
        mats.append(J1_0 @ self.dyn.tangent_jacobian(X[0]))

        # Defect blocks, one per shooting interval; see the docstring for the three formulas
        # and the sensitivity recursion behind them.
        K, nu = self.K, self.nu
        if K == 1:
            # classic per-node path: the uncached batch keeps the MjData call history (and so
            # the warm starts) byte-for-byte identical to the pre-K-step implementation
            F, A_all, B_all = self.dyn.dynamics_jacobian_batch(X[:self.N], U)
            for k in range(self.N):
                f_k = F[k]
                A, B = A_all[k], B_all[k]
                J1, J2 = self.dyn.state_diff_jacobian(X[k + 1], f_k)
                Q_inv_k = self.dyn.tangent_jacobian(X[k])
                Q_inv_k1 = self.dyn.tangent_jacobian(X[k + 1])
                c0, nc = self._p_spans[k]                   # only the knot columns
                Hk = H[k * nu:(k + 1) * nu, c0:c0 + nc]     # (nu, nc)

                mats.append(J2 @ A @ Q_inv_k)
                mats.append(J1 @ Q_inv_k1)
                mats.append(J2 @ B @ Hk)
        else:
            Z, A_all, B_all = self._rollout_fd(z)
            for i in range(self.N):
                c0, nc = self._p_spans[i]                   # only the knot columns
                Sx = np.eye(self.ndx)
                Sp = np.zeros((self.ndx, nc))
                for j in range(K):
                    k = i * K + j
                    Hj = H[k * nu:(k + 1) * nu, c0:c0 + nc]         # (nu, nc)
                    Sp = A_all[i, j] @ Sp + B_all[i, j] @ Hj
                    Sx = A_all[i, j] @ Sx
                J1, J2 = self.dyn.state_diff_jacobian(X[i + 1], Z[i, K])
                mats.append(J2 @ Sx @ self.dyn.tangent_jacobian(X[i]))
                mats.append(J1 @ self.dyn.tangent_jacobian(X[i + 1]))
                mats.append(J2 @ Sp)

        # Terminal constraint: c = x_N ⊖ x_goal,  dc/dx_N = J1 @ Q_inv(x_N)
        if self.has_terminal:
            assert self.x_goal is not None
            J1_N, _ = self.dyn.state_diff_jacobian(X[-1], self.x_goal)
            mats.append(J1_N @ self.dyn.tangent_jacobian(X[-1]))

        # Quaternion unit-norm constraints: d(||quat||^2 - 1)/d(quat) = 2 quat
        if self.is_floating:
            a = self.quat_addr
            assert a is not None
            for k in range(self.N + 1):
                mats.append((2.0 * X[k][a:a + 4]).reshape(1, 4))

        # flatten each block row-major and concatenate (matches the structure order)
        return np.concatenate([M.reshape(-1) for M in mats])

    ############################################################
    # SOLVE
    ############################################################

    def _reset_best(self) -> None:
        """Clear the per-iteration trackers at the start of a solve, and snapshot the keep-best
        settings. Read via getattr: a subclass may reassign self.cfg to a config that predates these
        fields (e.g. G1TrackingMPC swaps in its MPCTrackingConfig) -> default to off, rho=1e3."""
        self._best_merit = np.inf
        self._best_z = None
        self._best = None                 # metrics dict(iter, obj, inf_pr, inf_du) of the kept iterate
        self._last = None                 # metrics dict of the most recent iterate (for the summary)
        self._kbs = bool(getattr(self.cfg, "keep_best_sol", False))
        self._kbs_rho = float(getattr(self.cfg, "keep_best_sol_rho", 1e3))

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """IPOPT per-iteration hook: record the latest iterate's metrics (for the verbose summary)
        and, when keep_best_sol is on, stash the iterate with the smallest exact-penalty merit
        obj + rho*inf_pr (feasibility-weighted; inf_du is a diagnostic, not a selector). Guarded so
        a hiccup never aborts the solve."""
        try:
            self._last = dict(iter=int(iter_count), obj=float(obj_value),
                              inf_pr=float(inf_pr), inf_du=float(inf_du))
            if self._kbs:
                merit = float(obj_value) + self._kbs_rho * float(inf_pr)
                if merit < self._best_merit:
                    it = self.problem.get_current_iterate()      # {"x", ...} in original space
                    if it is not None and it.get("x") is not None:
                        self._best_merit = merit
                        self._best_z = np.array(it["x"], dtype=float, copy=True)
                        self._best = dict(self._last)            # metrics of the kept iterate
        except Exception:
            pass                                                 # tracking must never kill the solve
        return True

    def _print_solve_summary(self, info: dict, elapsed: float, kept: bool) -> None:
        """End-of-solve report: timing + status, the last iterate's (obj, inf_pr, inf_du), and --
        when a better iterate was returned in its place -- the kept iterate's metrics."""
        status = info.get("status_msg", b"")
        status = status.decode() if isinstance(status, (bytes, bytearray)) else str(status)
        print(f"[multi_shooting] solve done in {fmt_min_sec(elapsed)}  |  {status}")
        if self._last is not None:
            m = self._last
            print(f"[multi_shooting]   last #{m['iter']:>3}: obj={m['obj']:.4f}  "
                  f"inf_pr={m['inf_pr']:.2e}  inf_du={m['inf_du']:.2e}")
        if kept and self._best is not None:
            b = self._best
            print(f"[multi_shooting]   kept #{b['iter']:>3}: obj={b['obj']:.4f}  "
                  f"inf_pr={b['inf_pr']:.2e}  inf_du={b['inf_du']:.2e}   <- returned "
                  f"(min obj+{self._kbs_rho:g}*inf_pr)")

    def solve(self, X_guess: np.ndarray, U_guess: np.ndarray, verbose: bool = True):
        # initial condition must be set first (x_0 = x_init constraint)
        if self.x_init is None:
            raise RuntimeError(
                "Initial state not set, call set_initial_state(x0) before solve()."
            )

        # if the hard terminal constraint is on, the subclass must declare x_goal
        if self.has_terminal and self.x_goal is None:
            raise RuntimeError(
                "terminal_constraint=True but x_goal not set; declare self.x_goal "
                "in the subclass."
            )

        # fit the control guess to spline parameters, then pack z0. X_guess may be on the node
        # grid (N+1, nx) or the sim grid (N_sim+1, nx); subsample_nodes accepts either.
        p_guess = self.spline.fit(np.asarray(U_guess, dtype=float))
        z0 = self.pack(self.subsample_nodes(X_guess), p_guess)

        # solve the NLP (timed; MPC sub-solves pass verbose=False to stay quiet)
        self._reset_best()
        t0 = _time.perf_counter()
        z_sol, info = self.problem.solve(z0)
        elapsed = _time.perf_counter() - t0

        # keep-best only OVERRIDES a non-convergent exit (max-iter / wander)
        ipopt_ok = info.get("status") in (0, 1)   # Solve_Succeeded, Solved_To_Acceptable_Level
        kept = (self._kbs and not ipopt_ok and self._best_z is not None
                and not np.array_equal(self._best_z, z_sol))
        if kept:
            info["kept_best"] = dict(self._best)
            z_sol = self._best_z

        if verbose:
            self._print_solve_summary(info, elapsed, kept)

        # unpack solution (U_sol = spline.evaluate(p_sol), per sim step)
        X_sol, U_sol, _ = self.unpack(z_sol)

        return X_sol, U_sol, info

    ############################################################
    # GRID HELPERS
    ############################################################

    def subsample_nodes(self, X: np.ndarray) -> np.ndarray:
        """State trajectory -> shooting-node states (N+1, nx), from either grid:
            (N+1, nx)      already the node grid  ->  returned as-is
            (N_sim+1, nx)  the fine sim grid      ->  every K-th state
        Any other row count raises. Lets solve() and MPC warm starts accept either form."""

        # identify the grid by row count: node grid passes through, sim grid keeps every K-th
        # state (K = 1 makes the two identical, so the first test wins and nothing is dropped)
        X = np.asarray(X, dtype=float).reshape(-1, self.nx)
        if X.shape[0] == self.N + 1:
            return X
        if X.shape[0] == self.N_sim + 1:
            return X[::self.K]

        # neither grid: report both accepted lengths rather than fail deeper in the solve
        raise ValueError(
            f"state trajectory has {X.shape[0]} rows; expected N+1 = "
            f"{self.N + 1} (nodes) or N_sim+1 = {self.N_sim + 1} (sim grid)")

    def stitched_trajectory(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Fine-grid view of a solution: re-roll every interval from its node under the solved
        controls, stitch the internal states, then append the final node.
            X    (N+1, nx)      solved shooting-node states
            U    (N_sim, nu)    solved controls, one per sim step
            ->   (N_sim+1, nx)  sim-grid states (just a copy of X when K = 1)
        Used for replay and per-step tracking error. At a converged solution the defects sit at
        constraint tolerance, so the seam at each interval end is below visual level."""
        
        # K = 1: the nodes ALREADY are the sim grid, so there is nothing to stitch
        X = np.asarray(X, dtype=float).reshape(self.N + 1, self.nx)
        if self.K == 1:
            return X.copy()

        # re-roll all N intervals from their nodes, threaded (they are independent)
        U = np.asarray(U, dtype=float).reshape(self.N_sim, self.nu)
        Z = self.dyn.rollout_batch(X[:self.N], U.reshape(self.N, self.K, self.nu))

        # keep each interval's first K states, then close with node X_N; the interval ENDS are
        # dropped, since Z[i, K] and node X_{i+1} are the same point up to the defect residual
        out = np.zeros((self.N_sim + 1, self.nx))
        out[:self.N_sim] = Z[:, :self.K].reshape(self.N_sim, self.nx)
        out[self.N_sim] = X[self.N]
        return out
