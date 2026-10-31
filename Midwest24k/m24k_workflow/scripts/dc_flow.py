"""Sparse DC power flow on the solved network: signed branch flows and PTDF columns.

The solves save |p0|/s_nom only (branch_loading.parquet), which loses the flow direction. The
OPF is a lossless DC model, so the signed flows follow exactly from the hourly net injections
(dispatch - load): B theta = P + K^T b phi, f = b (K theta - phi), with one bus grounded.
`check_against_loading` confirms the reconstruction matches the saved loadings.

PTDF columns (MW of branch flow per MW injected at a bus and withdrawn across a sink
distribution) use the same factorisation, one sparse solve per bus. This avoids PyPSA's
dense PTDF (23.6k x 33.7k).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class DCFlow:
    def __init__(self, n):
        n.calculate_dependent_values()
        lines, trafos = n.lines, n.transformers
        self.branches = pd.Index([f"Line:{i}" for i in lines.index]
                                 + [f"Transformer:{i}" for i in trafos.index])
        bus0 = np.r_[lines.bus0.to_numpy(), trafos.bus0.to_numpy()]
        bus1 = np.r_[lines.bus1.to_numpy(), trafos.bus1.to_numpy()]
        self.buses = n.buses.index
        pos = pd.Series(np.arange(len(self.buses)), index=self.buses)
        i0, i1 = pos[bus0].to_numpy(), pos[bus1].to_numpy()
        m = len(self.branches)
        self.K = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)],
                                np.r_[i0, i1])), shape=(m, len(self.buses)))
        self.b = 1 / np.r_[lines.x_pu_eff.to_numpy(), trafos.x_pu_eff.to_numpy()]
        self.phi = np.r_[np.zeros(len(lines)), np.deg2rad(trafos.phase_shift.to_numpy())]
        self.s_nom = np.r_[lines.s_nom.to_numpy(), trafos.s_nom.to_numpy()]
        self.bus0, self.bus1 = bus0, bus1
        B = (self.K.T @ sp.diags(self.b) @ self.K).tocsc()
        self.slack = 0
        keep = np.arange(1, len(self.buses))
        self._lu = splu(B[keep][:, keep].tocsc())
        self._keep = keep
        self._shift_inj = self.K.T @ (self.b * self.phi)  # constant injection from phase shifters

    def _solve(self, P):
        """theta for injections P (buses x k); slack bus angle 0."""
        theta = np.zeros_like(P)
        theta[self._keep] = self._lu.solve(np.ascontiguousarray(P[self._keep]))
        return theta

    def flows(self, P):
        """Signed flows bus0->bus1 [MW], branches x hours, for net injections P (buses x hours)."""
        P = P + self._shift_inj[:, None]
        theta = self._solve(P)
        return self.b[:, None] * (self.K @ theta - self.phi[:, None])

    def ptdf(self, buses, sink_weights):
        """Branch flow change per MW injected at each bus in `buses`, withdrawn per `sink_weights`
        (Series over buses summing to 1). Returns branches x len(buses)."""
        pos = pd.Series(np.arange(len(self.buses)), index=self.buses)
        w = sink_weights.reindex(self.buses).fillna(0).to_numpy()
        P = -np.repeat(w[:, None], len(buses), axis=1)
        P[pos[list(buses)].to_numpy(), np.arange(len(buses))] += 1
        return self.b[:, None] * (self.K @ self._solve(P))


    def ptdf_rows(self, branch_idx, sink_weights):
        """Flow on each branch in `branch_idx` (positions) per MW injected at every bus and
        withdrawn per `sink_weights`. Returns len(branch_idx) x buses. B is symmetric, so each
        row is one solve with the branch's incidence vector."""
        w = sink_weights.reindex(self.buses).fillna(0).to_numpy()
        rhs = (self.K[branch_idx].T @ sp.diags(self.b[branch_idx])).toarray()
        R = self._solve(rhs).T
        return R - (R @ w)[:, None]


def decompose_lmp(lmp, R):
    """Split hourly LMPs into an energy price and one congestion term per branch:
        lmp[t, i] = energy[t] - sum_k mu[t, k] * R[k, i]
    fitted by least squares per hour (exact for a lossless DC OPF when R covers every branch
    that binds). Returns energy (T), mu (T x K) and the largest absolute residual."""
    X = np.c_[np.ones(R.shape[1]), -R.T]
    coef, *_ = np.linalg.lstsq(X, lmp.T, rcond=None)
    resid = float(np.abs(lmp.T - X @ coef).max())
    return coef[0], coef[1:].T, resid


def net_injections(n, gen_p):
    """Bus net injection [MW] (buses x hours): dispatch minus load, from the month network."""
    gens = n.generators.loc[gen_p.columns]
    gen_bus = gen_p.T.groupby(gens.bus.to_numpy()).sum()
    load = n.loads_t.p_set.loc[gen_p.index]
    load_bus = load.T.groupby(n.loads.bus.reindex(load.columns).to_numpy()).sum()
    P = gen_bus.reindex(n.buses.index, fill_value=0) - load_bus.reindex(n.buses.index, fill_value=0)
    return P.to_numpy(), load_bus


def check_against_loading(flows, s_nom, loading):
    """Largest |(|f|/s_nom) - saved loading| over all branches and hours."""
    return float(np.nanmax(np.abs(np.abs(flows) / s_nom[:, None] - loading.T.to_numpy())))
