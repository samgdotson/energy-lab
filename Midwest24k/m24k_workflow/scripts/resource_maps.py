"""Wind and solar resource maps (hexbin of mean capacity factor per ERA5 cell) for the month.

Wind: same turbine and smoothed power curve as build_wind_profiles.py.
Solar: atlite PV model, settings in config/siting.yaml (solar). Cells further than
`mask_km` from any bus are dropped so the maps cover the network footprint only.
Writes wind_resource.png, solar_resource.png and resource_cf_grid.csv (x, y, wind_cf, solar_cf).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import atlite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

SURFACE, INK, GRID = "#fcfcfb", "#0b0b0b", "#c9c8c0"


def mean_cf_grid(da):
    """(y, x) DataArray -> long DataFrame with x, y, value."""
    df = da.to_dataframe(name="v").reset_index()
    return df[["x", "y", "v"]]


def fig_resource(n, grid, col, cmap, label, title, path, gens=None):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect(1 / np.cos(np.deg2rad(n.buses.y.mean())))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    hb = ax.hexbin(grid.x, grid.y, C=grid[col], gridsize=48, cmap=cmap, linewidths=0.15,
                   edgecolors=SURFACE, reduce_C_function=np.mean)
    hv = n.lines[n.lines.v_nom >= 230] if "v_nom" in n.lines else n.lines
    segs = [[(n.buses.at[b0, "x"], n.buses.at[b0, "y"]), (n.buses.at[b1, "x"], n.buses.at[b1, "y"])]
            for b0, b1 in zip(hv.bus0, hv.bus1)]
    ax.add_collection(LineCollection(segs, colors="#ffffff", linewidths=0.35, alpha=0.55))
    handles = []
    if gens is not None and len(gens):
        handles.append(ax.scatter(gens.x, gens.y, s=4 + gens.p_nom / 25, facecolors="none",
                                  edgecolors=INK, linewidths=0.5, label="Existing wind (size ~ MW)"))
    ax.autoscale()
    cb = fig.colorbar(hb, ax=ax, shrink=0.6)
    cb.set_label(label)
    if handles:
        ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title(title, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--siting-config", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--wind-cutout", required=True)
    p.add_argument("--solar-cutout", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    scfg = yaml.safe_load(open(a.siting_config))["solar"]
    n = pypsa.Network(a.network)
    n.lines["v_nom"] = n.lines.bus0.map(n.buses.v_nom)

    wind = mean_cf_grid(atlite.Cutout(a.wind_cutout).wind(
        turbine=cfg["renewables"]["wind"]["turbine"], smooth=True, capacity_factor=True))
    solar = mean_cf_grid(atlite.Cutout(a.solar_cutout).pv(
        panel=scfg["panel"], orientation=scfg["orientation"], tracking=scfg.get("tracking"),
        capacity_factor=True))
    grid = wind.rename(columns={"v": "wind_cf"}).merge(
        solar.rename(columns={"v": "solar_cf"}), on=["x", "y"])

    # Keep cells within mask_km of a bus (equirectangular distance is fine at this scale)
    lat0 = np.deg2rad(n.buses.y.mean())
    km = lambda x, y: np.c_[np.asarray(x) * 111.32 * np.cos(lat0), np.asarray(y) * 110.57]
    dist, _ = cKDTree(km(n.buses.x, n.buses.y)).query(km(grid.x, grid.y))
    grid = grid[dist <= scfg.get("mask_km", 40)].reset_index(drop=True)
    grid.to_csv(out / "resource_cf_grid.csv", index=False)

    wg = n.generators[n.generators.carrier == "wind"]
    wg = wg.assign(x=wg.bus.map(n.buses.x), y=wg.bus.map(n.buses.y))
    month = f"{cfg['analysis']['year']}-{cfg['analysis']['month']:02d}"
    fig_resource(n, grid, "wind_cf", "viridis", "Mean capacity factor",
                 f"Wind resource, {month} ({cfg['renewables']['wind']['turbine']}, ERA5)",
                 out / "wind_resource.png", gens=wg)
    fig_resource(n, grid, "solar_cf", "inferno",
                 "Mean capacity factor",
                 f"Solar PV resource, {month} ({scfg['panel']}, {scfg['orientation']}"
                 f"{', ' + scfg['tracking'] + ' tracking' if scfg.get('tracking') else ''}, ERA5)",
                 out / "solar_resource.png")
    print(grid[["wind_cf", "solar_cf"]].describe().round(3))
