##
#
# Spline Curves
#
##

# for abstract classes
from abc import ABC, abstractmethod
from dataclasses import dataclass

# standard imports
import numpy as np


########################################################################
# CONFIG
########################################################################

@dataclass
class SplineConfig:

    # number of control points / knots
    M: int = 5

    # basis: "zero" | "linear" | "cubic"
    spline_type: str = "zero"


########################################################################
# BASE SPLINE  (general:  u = h(p))
########################################################################

class Spline(ABC):
    """ Base spline: a general map h from control points p (M points, nu inputs)
    to a control trajectory U (N, nu) over the horizon, U = h(p).
    """

    def __init__(self, config: SplineConfig, N: int, nu: int):

        self.cfg = config

        # spline dimensions
        self.M = config.M                   # number of control points / knots
        self.N = N                          # number of control steps (horizon)
        self.nu = nu                        # input dimension

        # normalized per-step query times over the horizon, in [0, 1]
        self.tq = np.linspace(0.0, 1.0, self.N)

    @property
    def dim(self) -> int:
        """Number of decision parameters p (= M * nu)."""
        return self.M * self.nu

    @property
    def spline_type(self) -> str:
        """Basis name this spline was built with ('zero' | 'linear' | 'cubic'); lets a
        consumer (e.g. a plotter) render the controls as steps vs ramps."""
        return self.cfg.spline_type

    @abstractmethod
    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """h(p): control points p (M, nu) or (dim,) -> control trajectory U (N, nu)."""
        ...

    @abstractmethod
    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """d vec(U) / d vec(p) at p; shape (N*nu, dim)."""
        ...

    @abstractmethod
    def fit(self, U: np.ndarray) -> np.ndarray:
        """(Inverse of evaluate.) Control trajectory U (N, nu) -> control points (M, nu)."""
        ...

    @abstractmethod
    def jacobian_T_matvec(self, V: np.ndarray) -> np.ndarray:
        """H^T @ vec(V) for V given as (N, nu), H = d vec(U)/d vec(p); shape (dim,).
        Abstract on purpose: the generic form would build the dense H just to transpose it,
        so every basis implements this structurally instead."""
        ...


########################################################################
# ZERO-ORDER HOLD SPLINE
########################################################################

class ZeroOrderSpline(Spline):
    """ Piecewise-constant hold over M equal intervals. """

    def __init__(self, config: SplineConfig, N: int, nu: int):
        super().__init__(config, N, nu)
        # interval index of each per-step query time: floor(t * M), clipped so
        # the endpoint t = 1 falls in the last interval (closed at 1).
        self._idx = np.clip((self.tq * self.M).astype(int), 0, self.M - 1)    # (N,)
        # constant flattened Jacobian d vec(U)/d vec(p): a 0/1 selection matrix. Built ONCE
        # (it is independent of p), mirroring Linear/Cubic -- jacobian() is called twice per
        # IPOPT eval, so rebuilding this dense matrix each time is pure overhead.
        self._H = np.zeros((self.N * self.nu, self.M * self.nu))
        rows = np.arange(self.N * self.nu)
        cols = (self._idx[:, None] * self.nu + np.arange(self.nu)).reshape(-1)
        self._H[rows, cols] = 1.0

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """U[k] = p[idx[k]], the control point of the interval holding step k."""
        P = np.asarray(p, dtype=float).reshape(self.M, self.nu)
        return P[self._idx]    # (N, nu)

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """d vec(U)/d vec(p): each step copies one control point, so the
        (constant) Jacobian is a 0/1 selection matrix, shape (N*nu, M*nu)."""
        return self._H    # (N*nu, M*nu)

    def jacobian_T_matvec(self, V: np.ndarray) -> np.ndarray:
        """H^T @ vec(V): scatter each step's row to its holding control point (H is 0/1
        selection), so this is a scatter-add -- no dense matrix, no transpose. Agrees with
        jacobian(p).T @ V.reshape(-1) to floating-point rounding, not bitwise -- the same terms
        summed in a different order."""
        V = np.asarray(V, dtype=float).reshape(self.N, self.nu)
        out = np.zeros((self.M, self.nu))
        np.add.at(out, self._idx, V)
        return out.reshape(-1)    # (M*nu,)

    def fit(self, U: np.ndarray) -> np.ndarray:
        """(Inverse of evaluate.) Average the steps that fall in each interval."""
        U = np.asarray(U, dtype=float).reshape(self.N, self.nu)
        P = np.zeros((self.M, self.nu))
        for j in range(self.M):
            mask = self._idx == j
            if np.any(mask):
                P[j] = U[mask].mean(axis=0)
        return P    # (M, nu)


########################################################################
# LINEAR (PIECEWISE-LINEAR) SPLINE
########################################################################

class LinearSpline(Spline):
    """ Piecewise-linear interpolation between adjacent control points. """

    def __init__(self, config: SplineConfig, N: int, nu: int):
        super().__init__(config, N, nu)
        # knot times over the closed horizon: t = j/(M-1), j = 0..M-1 (local: only Phi is kept)
        tk = np.linspace(0.0, 1.0, self.M)              # (M,)
        # blend matrix Phi (N, M): row k holds the linear-interp weights so that
        # U[k] = sum_j Phi[k, j] p[j].  For step k in interval [tk[j], tk[j+1]],
        # Phi[k, j] = 1 - w and Phi[k, j+1] = w.
        j = np.clip(np.searchsorted(tk, self.tq, side="right") - 1,
                    0, self.M - 2)                      # (N,) interval index
        w = (self.tq - tk[j]) / (tk[j + 1] - tk[j])     # (N,) in [0, 1]
        self.Phi = np.zeros((self.N, self.M))
        rows = np.arange(self.N)
        self.Phi[rows, j] = 1.0 - w
        self.Phi[rows, j + 1] = w
        # constant flattened Jacobian d vec(U)/d vec(p) = Phi (x) I_nu
        self._H = np.kron(self.Phi, np.eye(self.nu))    # (N*nu, M*nu)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """U = Phi @ P: linear blend of adjacent control points."""
        P = np.asarray(p, dtype=float).reshape(self.M, self.nu)
        return self.Phi @ P    # (N, nu)

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """d vec(U)/d vec(p); the constant H = Phi (x) I_nu, shape (N*nu, M*nu)."""
        return self._H    # (N*nu, M*nu)

    def jacobian_T_matvec(self, V: np.ndarray) -> np.ndarray:
        """H^T @ vec(V) = vec(Phi^T @ V), since H = Phi (x) I_nu, so the dense (N*nu x M*nu)
        matmul is skipped. Agrees with jacobian(p).T @ V.reshape(-1) to floating-point rounding,
        not bitwise -- the same terms summed in a different order."""
        V = np.asarray(V, dtype=float).reshape(self.N, self.nu)
        return (self.Phi.T @ V).reshape(-1)    # (M*nu,)

    def fit(self, U: np.ndarray) -> np.ndarray:
        """(Least-squares inverse.) Recover control points P from U = Phi @ P."""
        U = np.asarray(U, dtype=float).reshape(self.N, self.nu)
        P, *_ = np.linalg.lstsq(self.Phi, U, rcond=None)
        return P    # (M, nu)


########################################################################
# CUBIC (CLAMPED CUBIC B-SPLINE) SPLINE
########################################################################

class CubicSpline(Spline):
    """ Clamped cubic B-spline (degree 3) over M control points. """

    def __init__(self, config: SplineConfig, N: int, nu: int):
        super().__init__(config, N, nu)
        p = 3
        if self.M < p + 1:
            raise ValueError(f"cubic spline needs M >= {p + 1} control points, got M={self.M}")
        # clamped (open-uniform) knot vector on [0, 1]: p+1 zeros, M-p-1 interior knots
        # uniformly in (0, 1), then p+1 ones.  Length = M + p + 1.
        interior = np.linspace(0.0, 1.0, self.M - p + 1)[1:-1]            # (M-p-1,)
        knots = np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])
        # basis matrix Phi (N, M): Phi[k, j] = N_{j,p}(tq[k]); rows sum to 1.
        self.Phi = self._basis(self.tq, knots, p)                        # (N, M)
        # constant flattened Jacobian d vec(U)/d vec(p) = Phi (x) I_nu
        self._H = np.kron(self.Phi, np.eye(self.nu))                     # (N*nu, M*nu)

    @staticmethod
    def _basis(tq: np.ndarray, knots: np.ndarray, p: int) -> np.ndarray:
        """Cox-de Boor cubic B-spline basis: returns (len(tq), M) with M = len(knots)-p-1.
        Query times at the closing knot are nudged just inside so the clamped end evaluates
        to the last control point (half-open degree-0 intervals)."""
        u = np.minimum(np.asarray(tq, dtype=float), knots[-1] - 1e-9)    # (N,)
        nk = len(knots)
        # degree 0: indicator of each half-open knot span
        N = np.array([((knots[i] <= u) & (u < knots[i + 1])).astype(float)
                      for i in range(nk - 1)])                            # (nk-1, N)
        # raise the degree via the recursion (0/0 spans -> 0)
        for d in range(1, p + 1):
            Nd = np.zeros((nk - 1 - d, len(u)))
            for i in range(nk - 1 - d):
                d1 = knots[i + d] - knots[i]
                d2 = knots[i + d + 1] - knots[i + 1]
                t1 = (u - knots[i]) / d1 * N[i] if d1 > 0 else 0.0
                t2 = (knots[i + d + 1] - u) / d2 * N[i + 1] if d2 > 0 else 0.0
                Nd[i] = t1 + t2
            N = Nd
        return N.T                                                       # (N, M)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """U = Phi @ P: cubic B-spline blend of the control points."""
        P = np.asarray(p, dtype=float).reshape(self.M, self.nu)
        return self.Phi @ P    # (N, nu)

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """d vec(U)/d vec(p); the constant H = Phi (x) I_nu, shape (N*nu, M*nu)."""
        return self._H    # (N*nu, M*nu)

    def jacobian_T_matvec(self, V: np.ndarray) -> np.ndarray:
        """H^T @ vec(V) = vec(Phi^T @ V), since H = Phi (x) I_nu, so the dense (N*nu x M*nu)
        matmul is skipped. Agrees with jacobian(p).T @ V.reshape(-1) to floating-point rounding,
        not bitwise -- the same terms summed in a different order."""
        V = np.asarray(V, dtype=float).reshape(self.N, self.nu)
        return (self.Phi.T @ V).reshape(-1)    # (M*nu,)

    def fit(self, U: np.ndarray) -> np.ndarray:
        """(Least-squares inverse.) Recover control points P from U = Phi @ P."""
        U = np.asarray(U, dtype=float).reshape(self.N, self.nu)
        P, *_ = np.linalg.lstsq(self.Phi, U, rcond=None)
        return P    # (M, nu)


########################################################################
# FACTORY
########################################################################

SPLINES = {
    "zero":    ZeroOrderSpline,
    "linear":  LinearSpline,
    "cubic":   CubicSpline,
}


def make_spline(config: SplineConfig, N: int, nu: int) -> Spline:
    """ Build the spline selected by config.spline_type. """
    key = config.spline_type.lower()
    if key not in SPLINES:
        raise ValueError(
            f"unknown spline_type '{config.spline_type}'; choose from {list(SPLINES)}"
        )
    return SPLINES[key](config, N, nu)

