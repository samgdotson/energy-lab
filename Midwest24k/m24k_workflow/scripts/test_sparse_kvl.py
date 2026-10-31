"""Check pypsa_patches.sparse_kvl against PyPSA's own KVL on the Wisconsin 1664-bus case.

Solves the same 3-snapshot DC-OPF twice, in separate processes (the patch is
process-wide), and compares objective, dispatch and branch flows.
Usage: python scripts/test_sparse_kvl.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE = Path(__file__).parents[2] / "case_data" / "Wisconsin_1664.m"


def solve(patched: bool, out: Path):
    import pypsa
    from matpower_io import read_matpower, pwl_cost_table
    from pypsa_patches import skip_voltage_angles, sparse_kvl

    skip_voltage_angles()
    if patched:
        sparse_kvl()
    base_mva, s = read_matpower(CASE)
    ppc = {"version": "2", "baseMVA": base_mva, "bus": s["bus"].iloc[:, :13].to_numpy(),
           "gen": s["gen"].iloc[:, :21].to_numpy(), "branch": s["branch"].iloc[:, :13].to_numpy()}
    n = pypsa.Network()
    n.import_from_pypower_ppc(ppc)
    n.generators["p_set"] = np.nan
    n.generators["p_min_pu"] = 0.0
    gc = s["gencost"]
    if gc.iloc[0, 0] == 2:   # polynomial: use c1
        n.generators["marginal_cost"] = gc.iloc[:, -2].to_numpy()
    else:
        P, C = pwl_cost_table(gc)
        n.generators["marginal_cost"] = (C[:, -1] - C[:, 0]) / (P[:, -1] - P[:, 0])
    for c in ("lines", "transformers"):
        df = getattr(n, c)
        df.loc[df.v_ang_max.abs() == 0, "v_ang_max"] = np.inf
        df["v_ang_min"] = -np.inf
    n.set_snapshots(pd.RangeIndex(3, name="snapshot"))
    n.loads_t.p_set = pd.DataFrame(np.outer([0.7, 0.85, 1.0], n.loads.p_set),
                                   index=n.snapshots, columns=n.loads.index)
    n.optimize(solver_name="highs")
    res = {"objective": float(n.objective),
           "kvl_rows": int(n.model.constraints["Kirchhoff-Voltage-Law"].labels.size)}
    pd.concat([n.generators_t.p, n.lines_t.p0.add_prefix("L:"),
               n.transformers_t.p0.add_prefix("T:")], axis=1).to_parquet(out.with_suffix(".parquet"))
    out.write_text(json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        solve(sys.argv[1] == "patched", Path(sys.argv[2]))
        sys.exit()
    tmp = Path("logs/test_sparse_kvl")
    tmp.mkdir(parents=True, exist_ok=True)
    for mode in ("original", "patched"):
        subprocess.run([sys.executable, __file__, mode, str(tmp / f"{mode}.json")], check=True)
    a, b = (json.loads((tmp / f"{m}.json").read_text()) for m in ("original", "patched"))
    pa, pb = (pd.read_parquet(tmp / f"{m}.parquet") for m in ("original", "patched"))
    diff = (pa - pb[pa.columns]).abs().max().max()
    print(f"original {a}\npatched  {b}\nmax |dispatch/flow difference| = {diff:.4g} MW")
    assert abs(a["objective"] - b["objective"]) <= 1e-6 * abs(a["objective"])
    assert diff < 1e-2
    print("OK")
