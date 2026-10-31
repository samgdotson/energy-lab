"""Build normalised hourly load profiles (fraction of annual peak) for every state.

Priority per state: user-supplied CSV > EIA-930 balancing-authority demand > synthetic.
All profiles are indexed by the hour *start* in UTC, covering the whole year, and divided
by their maximum over the available data so the case's static load is treated as peak load.

Adding your own data: put a MISO market-day CSV (MarketDay, Hour1..Hour24, hour-ending EST)
under `load.user_profiles` in config.yaml, or write a CSV with a UTC hourly index and
one column per state and pass it with --extra-profiles.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def year_index(year):
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left",
                         tz="UTC")


def read_miso_market_csv(path):
    """MISO market-day format: hour-ending N on MarketDay, Eastern Standard Time (UTC-5)."""
    df = pd.read_csv(path)
    hours = [f"Hour{i}" for i in range(1, 25)]
    df = df.dropna(subset=["MarketDay"])[["MarketDay"] + hours]
    long = df.melt(id_vars="MarketDay", value_vars=hours, var_name="hour", value_name="MW")
    start_est = (pd.to_datetime(long.MarketDay, format="%m/%d/%Y")
                 + pd.to_timedelta(long.hour.str[4:].astype(int) - 1, unit="h"))
    long.index = (start_est + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
    return long.MW.sort_index().astype(float)


def read_eia930(files, bas):
    frames = []
    for f in files:
        d = pd.read_csv(f, thousands=",", low_memory=False)
        if d.empty:
            continue
        d = d[d["Balancing Authority"].isin(bas)]
        end = pd.to_datetime(d["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p")
        # Prefer EIA's cleaned series; fall back to reported demand where it is missing
        mw = d["Demand (MW) (Adjusted)"].fillna(d["Demand (MW)"])
        frames.append(pd.DataFrame({"ba": d["Balancing Authority"],
                                    "start": (end - pd.Timedelta(hours=1)).dt.tz_localize("UTC"),
                                    "MW": mw}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return df.pivot_table(index="start", columns="ba", values="MW", aggfunc="mean").sort_index()


def synthetic_profile(index, seed, lat=42.0):
    """Placeholder load shape: diurnal (late-afternoon peak) x weekly x seasonal + AR(1) noise."""
    rng = np.random.default_rng(seed)
    local = index.tz_convert("Etc/GMT+6")  # Central Standard Time
    h = local.hour.to_numpy()
    doy = local.day_of_year.to_numpy()
    diurnal = 0.78 + 0.22 * np.exp(-((h - 17) % 24 - 0) ** 2 / 18) + 0.08 * np.sin((h - 6) / 24 * 2 * np.pi)
    weekly = np.where(local.day_of_week.to_numpy() >= 5, 0.92, 1.0)
    # summer peak plus smaller winter peak
    seasonal = 0.82 + 0.15 * np.exp(-((doy - 205) / 35) ** 2) + 0.06 * np.exp(-((doy - 15) / 30) ** 2)
    noise = np.zeros(len(index))
    for t in range(1, len(index)):
        noise[t] = 0.97 * noise[t - 1] + rng.normal(0, 0.006)
    s = pd.Series(diurnal * weekly * seasonal * (1 + noise), index=index)
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--extra-profiles", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--sources", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    year, month = cfg["analysis"]["year"], cfg["analysis"]["month"]
    lcfg = cfg["load"]
    idx = year_index(year)
    month_mask = idx.month == month

    eia = read_eia930(json.loads(Path(a.manifest).read_text())["files"],
                      sorted(set(lcfg["state_to_ba"].values())))
    extra = (pd.read_csv(a.extra_profiles, index_col=0, parse_dates=True)
             if a.extra_profiles else pd.DataFrame())

    profiles, sources = {}, {}
    for state, ba in lcfg["state_to_ba"].items():
        series, source = None, None
        if state in extra.columns:
            series, source = extra[state], f"extra CSV {a.extra_profiles}"
        elif state in (lcfg.get("user_profiles") or {}):
            path = lcfg["user_profiles"][state]
            series, source = read_miso_market_csv(path), f"user CSV {path}"
        elif ba in eia.columns:
            series, source = eia[ba], f"EIA-930 {ba}"
        if series is not None:
            series = series.reindex(idx)
            # need full coverage of the analysis month; short gaps are interpolated
            gaps = series[month_mask].isna().sum()
            if gaps > 24:
                series, source = None, None
            else:
                series = series.interpolate(limit_direction="both")
                if gaps:
                    source += f" ({gaps} h interpolated in analysis month)"
        if series is None:
            series = synthetic_profile(idx, lcfg["synthetic_seed"] + len(profiles))
            source = "synthetic"
        profiles[state] = series / series.max()
        sources[state] = {"source": source, "peak_MW_raw": float(series.max()),
                          "month_mean_fraction_of_peak": float(profiles[state][month_mask].mean())}

    out = pd.DataFrame(profiles, index=idx)
    out.index.name = "snapshot_utc"
    out.to_csv(a.out)
    Path(a.sources).write_text(json.dumps(sources, indent=2))
    print(json.dumps(sources, indent=2))
