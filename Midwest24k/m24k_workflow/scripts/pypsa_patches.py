"""Workarounds for PyPSA 1.4 scaling issues on large networks (see LOGBOOK.md)."""

import types

import pandas as pd
import pypsa
from pypsa.network.graph import NetworkGraphMixin


class _SparseFrame:
    """Just enough of a DataFrame for Network.determine_network_topology."""

    def __init__(self, matrix, index):
        self.values = matrix.tocsr()
        self.index = index


def sparse_topology(n: pypsa.Network) -> pypsa.Network:
    """Make determine_network_topology use a sparse adjacency matrix.

    PyPSA 1.4 calls adjacency_matrix(return_dataframe=True), a dense bus x bus DataFrame
    (23,643^2 int64 = 4.2 GB for Midwest24k), only to pass `.values` to
    scipy.sparse.csgraph.connected_components, which accepts sparse input. This patch
    (instance-level, affects only `n`) returns the sparse matrix instead when asked for
    the topology's DataFrame.
    """

    def adjacency_matrix(self, *args, return_dataframe=None, busorder=None, **kwargs):
        if return_dataframe and kwargs.get("weights") is None:
            order = self.c.buses.static.index if busorder is None else busorder
            m = NetworkGraphMixin.adjacency_matrix(self, *args, busorder=order,
                                                   return_dataframe=False, **kwargs)
            return _SparseFrame(m, order)
        return NetworkGraphMixin.adjacency_matrix(self, *args, busorder=busorder,
                                                  return_dataframe=return_dataframe, **kwargs)

    n.adjacency_matrix = types.MethodType(adjacency_matrix, n)
    return n


def sparse_kvl() -> None:
    """Build the cycle-based KVL constraints from the sparse cycle matrix (process-wide).

    PyPSA 1.4 densifies sub_network.C into a DataFrame (33,739 branches x 10,097 cycles,
    copied several times: ~20 GB) and then evaluates `flow @ C`, which linopy broadcasts to
    a dense snapshot x branch x cycle array (24 x 23,795 x 10,097 float64 = 43 GB). Here
    each cycle's nonzero branches become the terms of one constraint, padded to the longest
    cycle. Same constraint as PyPSA: sum_l C_lk * x_pu_eff_l * 1e5 * s_l
    = -sum_l C_lk * phase_shift_l(rad) * 1e5. Covers single-period networks with fixed
    phase shifts and all branches active (what this workflow builds).
    """
    import numpy as np
    import xarray as xr
    from linopy import LinearExpression
    from linopy.constants import TERM_DIM
    import pypsa.optimization.optimize as opt

    if getattr(opt.define_kirchhoff_voltage_constraints, "_sparse", False):
        return

    def define_kirchhoff_voltage_constraints(n, sns, cycle_basis_method="bfs-refined"):
        m = n.model
        assert not n._multi_invest, "sparse_kvl: multi-period models not supported"
        n.calculate_dependent_values()
        n.determine_network_topology(skip_isolated_buses=True,
                                     cycle_basis_method=cycle_basis_method)
        sns = pd.Index(sns)
        labels = {c: m[f"{c}-s"].labels.sel(snapshot=sns) for c in ("Line", "Transformer")
                  if f"{c}-s" in m.variables}

        rows_vars, rows_coef, consts = [], [], []
        for sub in n.c.sub_networks.static.obj:
            if not hasattr(sub, "C") or not sub.C.size:
                continue
            br = sub.branches()
            assert sub.n.c.sub_networks.static.carrier[sub.name] == "AC"
            C = sub.C.tocsc()
            weight = (br.x_pu_eff.to_numpy() * 1e5)
            shift = np.zeros(len(br))
            is_tr = br.index.get_level_values(0) == "Transformer"
            if is_tr.any():
                shift[is_tr] = br.loc[is_tr, "phase_shift"].to_numpy() * np.pi / 180 * 1e5
            # variable labels (snapshot x branch) in the row order of C
            lab = np.empty((len(sns), len(br)), dtype=np.int64)
            for c, da in labels.items():
                sel = br.index.get_level_values(0) == c
                names = br.index.get_level_values(1)[sel]
                lab[:, sel] = da.sel(name=names).transpose("snapshot", "name").to_numpy()
            for k in range(C.shape[1]):
                lo, hi = C.indptr[k], C.indptr[k + 1]
                rows, sign = C.indices[lo:hi], C.data[lo:hi]
                rows_vars.append(lab[:, rows])
                rows_coef.append(sign * weight[rows])
                consts.append(-(sign * shift[rows]).sum())

        if not rows_vars:
            return
        K, L, S = len(rows_vars), max(len(c) for c in rows_coef), len(sns)
        vars_ = np.full((S, K, L), -1, dtype=np.int64)
        coeffs = np.full((S, K, L), np.nan)
        for k, (v, c) in enumerate(zip(rows_vars, rows_coef)):
            vars_[:, k, :v.shape[1]] = v
            coeffs[:, k, :len(c)] = c
        coords = {"snapshot": sns, "cycle": np.arange(K)}
        ds = xr.Dataset({"vars": (("snapshot", "cycle", TERM_DIM), vars_),
                         "coeffs": (("snapshot", "cycle", TERM_DIM), coeffs)}, coords=coords)
        lhs = LinearExpression(ds, m)
        rhs = xr.DataArray(np.array(consts), coords={"cycle": np.arange(K)}, dims="cycle")
        m.add_constraints(lhs == rhs, name="Kirchhoff-Voltage-Law")

    define_kirchhoff_voltage_constraints._sparse = True
    opt.define_kirchhoff_voltage_constraints = define_kirchhoff_voltage_constraints


def skip_voltage_angles() -> None:
    """Skip the voltage-angle step of OptimizationAccessor.post_processing (process-wide).

    After solving, PyPSA 1.4 computes bus voltage angles with
    np.linalg.pinv(B.todense()): a dense SVD of the 23,643 x 23,643 susceptance matrix,
    which ran for 40+ CPU-minutes at 12 GB. The OPF results (dispatch, flows, LMPs) don't
    depend on it. The step only runs when sub_networks has an "obj" column, so the column
    is hidden for the duration of post_processing and restored afterwards.
    buses_t.v_ang stays empty.
    """
    from pypsa.optimization.optimize import OptimizationAccessor

    original = OptimizationAccessor.post_processing
    if getattr(original, "_skips_angles", False):
        return

    def post_processing(self):
        subs = self._n.c.sub_networks.static
        obj = subs.pop("obj") if "obj" in subs else None
        try:
            original(self)
        finally:
            if obj is not None:
                subs["obj"] = obj

    post_processing._skips_angles = True
    OptimizationAccessor.post_processing = post_processing
