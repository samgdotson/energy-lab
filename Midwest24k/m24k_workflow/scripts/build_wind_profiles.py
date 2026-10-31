"""Hourly capacity factors (p_max_pu) for every wind generator in the analysis month.

Sources (config: renewables.wind.source):
  atlite    - ERA5 cutout, nearest grid cell to each generator's bus, smoothed turbine power
              curve (same method as notebooks/read_matpower.ipynb)
  csv       - user file: UTC hourly index, one column per generator name (missing -> error)
  synthetic - seeded placeholder (regional AR(1) weather factor + local noise + nocturnal bias)
Output index: hour start, UTC.
"""

import argparse
import calendar
import logging

import numpy as np
import pandas as pd
import pypsa
import yaml

logger = logging.getLogger(__name__)


def month_index(year, month):
    last = calendar.monthrange(year, month)[1]
    return pd.date_range(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last} 23:00",
                         freq="h", tz="UTC")


def from_atlite(cutout_path, x, y, turbine_name):
    import atlite
    import xarray as xr
    from atlite.convert import convert_wind
    from atlite.resource import get_windturbineconfig, windturbine_smooth

    cutout = atlite.Cutout(cutout_path)
    turbine = windturbine_smooth(get_windturbineconfig(turbine_name))
    gx = xr.DataArray(x.values, dims="generator", coords={"generator": x.index.to_numpy()})
    gy = xr.DataArray(y.values, dims="generator", coords={"generator": x.index.to_numpy()})
    points = cutout.data.sel(x=gx, y=gy, method="nearest")
    cf = convert_wind(points, turbine, interpolation_method="power").transpose("time", "generator")
    df = cf.to_pandas()
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    return df.clip(0, 1)


def synthetic(index, gens, x, y, seed):
    """Placeholder: spatially correlated AR(1) wind speed -> generic power curve."""
    rng = np.random.default_rng(seed)
    T = len(index)
    regional = np.zeros((T, 3))  # three weather regimes west -> east
    for t in range(1, T):
        regional[t] = 0.95 * regional[t - 1] + rng.normal(0, 0.3, 3)
    w = np.clip((x.to_numpy() + 106) / 24, 0, 1)  # 0 at west edge, 1 at east edge
    weights = np.stack([np.clip(1 - 2 * w, 0, 1), 1 - np.abs(2 * w - 1), np.clip(2 * w - 1, 0, 1)], 1)
    local = rng.normal(0, 0.25, (T, len(gens)))
    hour_local = (index.hour.to_numpy() - 6) % 24
    nocturnal = 0.6 * np.cos((hour_local - 2) / 24 * 2 * np.pi)[:, None]
    speed = 7.5 + 1.0 * (regional @ weights.T) + local + nocturnal - 1.2 * w  # m/s at hub
    cf = np.clip((speed - 3.0) / (12.0 - 3.0), 0, 1) ** 1.8  # cut-in 3 m/s, rated 12 m/s
    return pd.DataFrame(cf, index=index, columns=gens)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--cutout", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(open(a.config))
    wcfg = cfg["renewables"]["wind"]
    idx = month_index(cfg["analysis"]["year"], cfg["analysis"]["month"])
    n = pypsa.Network(a.network)
    wind = n.generators.index[n.generators.carrier == "wind"]
    x = n.generators.loc[wind, "bus"].map(n.buses.x)
    y = n.generators.loc[wind, "bus"].map(n.buses.y)

    if wcfg["source"] == "atlite":
        cf = from_atlite(a.cutout, x, y, wcfg["turbine"]).reindex(idx)
        assert cf.notna().all().all(), "cutout does not cover the analysis month"
    elif wcfg["source"] == "csv":
        cf = pd.read_csv(wcfg["csv"], index_col=0, parse_dates=True)
        cf.index = pd.DatetimeIndex(cf.index).tz_convert("UTC") if cf.index.tz else cf.index.tz_localize("UTC")
        missing = set(wind) - set(cf.columns)
        assert not missing, f"wind CSV lacks {len(missing)} generators, e.g. {sorted(missing)[:3]}"
        cf = cf.reindex(idx)[wind]
    elif wcfg["source"] == "synthetic":
        cf = synthetic(idx, wind, x, y, cfg["renewables"]["synthetic_seed"])
    else:
        raise ValueError(wcfg["source"])

    cf.index.name = "snapshot_utc"
    cf.to_csv(a.out)
    logger.info("wind CF (%s): mean %.3f, min hourly fleet mean %.3f, max %.3f", wcfg["source"],
                cf.to_numpy().mean(), cf.mean(axis=1).min(), cf.mean(axis=1).max())
