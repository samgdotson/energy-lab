"""Attach the analysis month's time series to the base network.

- Snapshots: hourly, hour start in UTC (stored tz-naive; PyPSA/netCDF drop the tz).
- Loads: static case p_set x state profile (fraction of annual peak).
- Wind: p_max_pu from build_wind_profiles.py.
- Hydro: constant availability factor (placeholder until inflow data exist).
- Load shedding: one generator per load bus at a high cost, so infeasible hours still solve
  and the shortfall is visible in the results.
"""

import argparse
import json

import pandas as pd
import pypsa
import yaml

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--load-profiles", required=True)
    p.add_argument("--wind-cf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    year, month = cfg["analysis"]["year"], cfg["analysis"]["month"]
    n = pypsa.Network(a.network)

    prof = pd.read_csv(a.load_profiles, index_col=0, parse_dates=True)
    prof = prof[(prof.index.year == year) & (prof.index.month == month)]
    wind = pd.read_csv(a.wind_cf, index_col=0, parse_dates=True)
    assert prof.index.equals(wind.index), "load and wind indices differ"
    snapshots = prof.index.tz_convert(None)
    n.set_snapshots(snapshots)

    missing = set(n.loads.state) - set(prof.columns)
    assert not missing, f"no load profile for states {missing}"
    p_set = prof[n.loads.state.to_numpy()].to_numpy() * n.loads.p_set.to_numpy()
    n.loads_t.p_set = pd.DataFrame(p_set, index=snapshots, columns=n.loads.index)

    wind.index = snapshots
    n.generators_t.p_max_pu = wind[n.generators.index[n.generators.carrier == "wind"]]
    hydro = n.generators.carrier == "hydro"
    n.generators.loc[hydro, "p_max_pu"] = cfg["renewables"]["hydro_availability"]

    load_buses = n.loads.bus.unique()
    n.add("Carrier", "load shedding", color="#d62728")
    n.add("Generator", [f"{b} load shedding" for b in load_buses], bus=load_buses,
          carrier="load shedding", p_nom=n.loads.groupby("bus").p_set.sum().loc[load_buses].clip(lower=1).values,
          marginal_cost=cfg["solving"]["load_shedding_cost"])
    # Shedding capacity = the bus's peak (static) load; the 1 MW floor covers the negative load

    load = n.loads_t.p_set.sum(axis=1)
    stats = {
        "snapshots": len(snapshots), "first": str(snapshots[0]), "last": str(snapshots[-1]),
        "load_MW": {"mean": round(load.mean(), 1), "min": round(load.min(), 1),
                    "max": round(load.max(), 1)},
        "energy_TWh": round(load.sum() / 1e6, 3),
        "wind_cf_mean": round(float(n.generators_t.p_max_pu.to_numpy().mean()), 3),
        "wind_energy_available_TWh": round(float(
            (n.generators_t.p_max_pu * n.generators.p_nom[n.generators_t.p_max_pu.columns]).sum().sum() / 1e6), 3),
        "load_shedding_generators": len(load_buses),
    }
    n.export_to_netcdf(a.out)
    with open(a.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
