"""Benchmark: build + solve one snapshot of the base network with a given HiGHS method."""

import logging
import sys
import time

import pypsa

method = sys.argv[1] if len(sys.argv) > 1 else "ipm"
logging.basicConfig(level=logging.WARNING)
n = pypsa.Network("resources/base_network.nc")
from pypsa_patches import sparse_topology
sparse_topology(n)
t = time.time(); n.determine_network_topology(); print(f">> topology {time.time() - t:.1f} s, {len(n.sub_networks)} sub-networks", flush=True)
t = time.time()
n.optimize.create_model()
print(f">> model build {time.time() - t:.1f} s", flush=True)
opts = {"solver": method, "primal_feasibility_tolerance": 1e-6, "dual_feasibility_tolerance": 1e-6}
if method == "ipm":
    opts["run_crossover"] = "off"
t = time.time()
status = n.optimize.solve_model(solver_name="highs", **opts)
print(f">> {method}: {status} solve {time.time() - t:.1f} s", flush=True)
if status[1] == "optimal":
    print(f">> objective {n.objective:,.0f}")
    print(">> dispatch", n.generators_t.p.iloc[0].groupby(n.generators.carrier).sum().round(0).to_dict())
    for c in ("Line", "Transformer"):
        st, dy = n.components[c].static, n.components[c].dynamic
        print(f">> {c} binding", int((dy.p0.iloc[0].abs() / st.s_nom > 0.999).sum()))
    print(">> LMP", n.buses_t.marginal_price.iloc[0].describe().round(2).to_dict())
