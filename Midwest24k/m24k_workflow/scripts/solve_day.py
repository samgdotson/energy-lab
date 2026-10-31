"""Solve the linear DC-OPF for one day (24 hourly snapshots) of the prepared month network.

Days are independent (no storage, ramping or unit commitment), so the month is solved as
31 separate problems that Snakemake can run in parallel. Writes compact results:
  gen_p.parquet           generator dispatch [MW]          (snapshot x generator)
  lmp.parquet             nodal prices [$/MWh]              (snapshot x bus)
  branch_loading.parquet  |p0| / s_nom                      (snapshot x branch, float32)
  status.json             solver status, objective, timing, shedding
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import pypsa
import yaml

from pypsa_patches import skip_voltage_angles, sparse_kvl, sparse_topology

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--day", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("linopy").setLevel(logging.WARNING)

    cfg = yaml.safe_load(open(a.config))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    skip_voltage_angles()
    sparse_kvl()
    t0 = time.time()
    n = sparse_topology(pypsa.Network(a.network))
    day = n.snapshots[n.snapshots.normalize() == pd.Timestamp(a.day)]
    assert len(day) == 24, f"{a.day}: expected 24 snapshots, got {len(day)}"
    n.set_snapshots(day)
    t_load = time.time() - t0

    t0 = time.time()
    status = n.optimize(solver_name=cfg["solving"]["solver"], include_objective_constant=False,
                        **cfg["solving"]["solver_options"])
    t_solve = time.time() - t0
    if status[1] != "optimal":
        raise RuntimeError(f"{a.day}: solver returned {status}")

    gen_p = n.generators_t.p
    shed = gen_p.loc[:, n.generators.carrier == "load shedding"]
    gen_p.to_parquet(out / "gen_p.parquet")
    n.buses_t.marginal_price.to_parquet(out / "lmp.parquet")
    loading = pd.concat(
        [(n.components[c].dynamic.p0.abs() / n.components[c].static.s_nom).add_prefix(f"{c}:")
         for c in ("Line", "Transformer")], axis=1).astype("float32")
    loading.to_parquet(out / "branch_loading.parquet")

    info = {
        "day": a.day, "status": list(status), "objective": float(n.objective),
        "load_MWh": float(n.loads_t.p_set.sum().sum()),
        "shed_MWh": float(shed.sum().sum()), "shed_buses": int((shed.sum() > 1e-3).sum()),
        "seconds_load": round(t_load, 1), "seconds_optimize": round(t_solve, 1),
    }
    (out / "status.json").write_text(json.dumps(info, indent=2))
    logger.info(json.dumps(info))
