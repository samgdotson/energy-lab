"""Build the static (single-snapshot) PyPSA network for the Midwest24k case.

Route: own .m parser -> ppc dict -> pypsa.Network.import_from_pypower_ppc, then the fixes
learned on the Wisconsin case (see debug/DEBUGGING_LOG.md) plus Midwest24k specifics.
Decisions are recorded in LOGBOOK.md.
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
import pypsa

from matpower_io import read_matpower, pwl_cost_table
from pypsa_patches import sparse_topology

logger = logging.getLogger(__name__)


def build(case, bus_meta, sub_meta):
    base_mva, s = read_matpower(case)
    bus, gen, branch, gencost = s["bus"], s["gen"], s["branch"], s["gencost"]
    stats = {"buses": len(bus), "generators_in_case": len(gen), "branches": len(branch)}

    # The importer ignores status, so drop out-of-service units here (all branches are in service)
    online = (gen.GEN_STATUS == 1).to_numpy()
    assert (branch.BR_STATUS == 1).all()
    gen, gencost = gen[online].reset_index(drop=True), gencost[online].reset_index(drop=True)
    fuel = pd.Series(np.array(s["genfuel"])[online])
    gtype = np.array(s["gentype"])[online]
    stats["generators_offline_dropped"] = int((~online).sum())
    stats["offline_by_type"] = pd.Series(np.array(s["gentype"])[~online]).value_counts().to_dict()

    ppc = {"version": "2", "baseMVA": base_mva,
           "bus": bus.iloc[:, :13].to_numpy(),
           "gen": gen.iloc[:, :21].to_numpy(),
           "branch": branch.iloc[:, :13].to_numpy()}
    n = sparse_topology(pypsa.Network(name="Midwest24k"))
    n.import_from_pypower_ppc(ppc)

    # Importer pins dispatch to the AC solution's PG -> infeasible in a lossless DC OPF
    n.generators["p_set"] = np.nan

    # Piecewise-linear costs -> one linear marginal cost: average slope between first and
    # last breakpoint (P0 = PMIN, Pn = PMAX). The cost at P0 (a no-load-like term) is dropped.
    P, C = pwl_cost_table(gencost)
    n.generators["marginal_cost"] = (C[:, -1] - C[:, 0]) / (P[:, -1] - P[:, 0])
    n.generators["p_min_pu"] = 0.0  # PMIN == PG for most units in this export; not a real minimum
    n.generators["carrier"] = fuel.values
    n.generators["type"] = gtype
    names = (n.generators.bus + " " + fuel.values + " "
             + (fuel.groupby(n.generators.bus.values + fuel.values).cumcount() + 1).astype(str).values)
    n.generators.index = pd.Index(names.values, name=n.generators.index.name)

    for c in ("lines", "transformers"):
        df = getattr(n, c)
        if "status" in df.columns:
            del df["status"]
        # MATPOWER ANGMIN = ANGMAX = 0 means "no limit", but PyPSA >= 1.4 enforces any finite
        # v_ang_max (degrees, symmetric) -> 0 would force every flow to zero. v_ang_min is
        # deprecated and ignored by PyPSA; reset it to its default.
        unlimited = df.v_ang_max.abs() == 0
        df.loc[unlimited, "v_ang_max"] = np.inf
        df["v_ang_min"] = -np.inf
        stats[f"{c}_with_angle_limit"] = int(np.isfinite(df.v_ang_max).sum())

    # Coordinates via bus -> substation; state from the bus "Area Name"
    bm = pd.read_csv(bus_meta, skiprows=1).set_index("Number")
    sm = pd.read_csv(sub_meta, skiprows=1).set_index("Sub Num")
    idx = n.buses.index.astype(int)
    sub = bm.loc[idx, "Sub Num"].to_numpy()
    n.buses["x"] = sm.loc[sub, "Longitude"].to_numpy()
    n.buses["y"] = sm.loc[sub, "Latitude"].to_numpy()
    n.buses["state"] = bm.loc[idx, "Area Name"].to_numpy()
    n.buses["location"] = s["bus_name"]
    n.generators["state"] = n.generators.bus.map(n.buses.state)
    n.loads["state"] = n.loads.bus.map(n.buses.state)

    # The single DC line has PMIN = PMAX = 0 -> no transfer capability; skip it
    if "dcline" in s:
        dc = s["dcline"]
        stats["dclines_skipped_zero_capacity"] = int((dc.PMAX <= 0).sum())
        if (dc.PMAX > 0).any():
            raise NotImplementedError("DC lines with capacity: add as pypsa Links")

    n.sanitize()
    n.determine_network_topology()
    stats.update({
        "generators_online": len(n.generators), "lines": len(n.lines),
        "transformers": len(n.transformers), "loads": len(n.loads),
        "sub_networks": len(n.sub_networks),
        "load_MW": round(n.loads.p_set.sum(), 1),
        "p_nom_by_carrier_MW": n.generators.groupby("carrier").p_nom.sum().round(1).to_dict(),
        "negative_x_lines": int((n.lines.x < 0).sum()),
        "negative_x_transformers": int((n.transformers.x < 0).sum()),
        "nonconvex_pwl_curves": int((np.diff(np.diff(C, axis=1) / np.diff(P, axis=1), axis=1)
                                     < -1e-6).any(axis=1).sum()),
    })
    return n, stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--bus-meta", required=True)
    p.add_argument("--sub-meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    n, stats = build(a.case, a.bus_meta, a.sub_meta)
    n.export_to_netcdf(a.out)
    with open(a.stats, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))
