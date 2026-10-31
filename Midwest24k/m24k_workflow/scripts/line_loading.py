"""Month line-loading maps and signed branch flows.

Rebuilds signed hourly flows from the daily dispatch (scripts/dc_flow.py), checks them against
the saved |p0|/s_nom, and writes:
  branch_flows.parquet   signed flow bus0->bus1 [MW], hours x branches (float32)
  branch_loading_stats.csv  per-branch mean/p95 loading, hours >= 90 % and at limit, mean flow
  line_loading_map.png   every branch coloured by mean loading (plasma)
  line_loading_hexbin.png  capacity-weighted mean loading, hexbin over branch midpoints
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.collections import LineCollection

from dc_flow import DCFlow, check_against_loading, net_injections

# Dark surface: plasma's low end is near-black and reads poorly on the light report surface
DARK, DARK_INK, DARK_MUTED = "#141418", "#ececea", "#9a9993"


def load_days(days_dir):
    days = sorted(p for p in Path(days_dir).iterdir() if (p / "status.json").exists())
    gen_p = pd.concat([pd.read_parquet(d / "gen_p.parquet") for d in days])
    loading = pd.concat([pd.read_parquet(d / "branch_loading.parquet") for d in days])
    return gen_p, loading


def dark_map_axes(n, figsize=(7.5, 6.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.set_aspect(1 / np.cos(np.deg2rad(n.buses.y.mean())))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def dark_colorbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6)
    cb.set_label(label, color=DARK_INK)
    cb.ax.yaxis.set_tick_params(color=DARK_MUTED, labelcolor=DARK_MUTED)
    cb.outline.set_edgecolor(DARK_MUTED)
    return cb


def branch_xy(n, dc):
    x0, y0 = n.buses.x.reindex(dc.bus0).to_numpy(), n.buses.y.reindex(dc.bus0).to_numpy()
    x1, y1 = n.buses.x.reindex(dc.bus1).to_numpy(), n.buses.y.reindex(dc.bus1).to_numpy()
    return x0, y0, x1, y1


def fig_loading_map(n, dc, mean_loading, path):
    x0, y0, x1, y1 = branch_xy(n, dc)
    order = np.argsort(mean_loading)  # most loaded drawn last (on top)
    segs = np.stack([np.c_[x0, y0], np.c_[x1, y1]], axis=1)[order]
    v = mean_loading[order]
    fig, ax = dark_map_axes(n)
    lc = LineCollection(segs, array=v, cmap="plasma", linewidths=0.25 + 1.6 * v ** 2)
    lc.set_clim(0, 1)
    ax.add_collection(lc)
    ax.autoscale()
    dark_colorbar(fig, lc, ax, "Mean loading |flow| / rating, Aug 2015")
    ax.set_title("Average branch loading (line width grows with loading)", loc="left", color=DARK_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=DARK)
    plt.close(fig)


def fig_loading_hexbin(n, dc, mean_loading, path):
    x0, y0, x1, y1 = branch_xy(n, dc)
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    # Capacity-weighted mean loading per hexagon: sum(load * s_nom) / sum(s_nom)
    fig, ax = dark_map_axes(n)
    num = ax.hexbin(xm, ym, C=mean_loading * dc.s_nom, reduce_C_function=np.sum, gridsize=55,
                    visible=False)
    den = ax.hexbin(xm, ym, C=dc.s_nom, reduce_C_function=np.sum, gridsize=55, visible=False)
    hb = ax.hexbin(xm, ym, C=mean_loading, gridsize=55, cmap="plasma", vmin=0, vmax=0.6,
                   linewidths=0.2, edgecolors=DARK)
    hb.set_array(num.get_array() / den.get_array())
    num.remove()
    den.remove()
    dark_colorbar(fig, hb, ax, "Capacity-weighted mean loading (clipped at 0.6)")
    ax.set_title("Where the grid is busy: branch loading by area", loc="left", color=DARK_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=DARK)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--days-dir", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    n = pypsa.Network(a.network)
    gen_p, loading = load_days(a.days_dir)
    dc = DCFlow(n)
    P, _ = net_injections(n, gen_p)
    F = dc.flows(P)
    err = check_against_loading(F, dc.s_nom, loading[dc.branches])
    print(f"max |reconstructed - saved loading| = {err:.2e}")
    assert err < 1e-4, "signed flow reconstruction does not match the solved loadings"

    flows = pd.DataFrame(F.T.astype("float32"), index=gen_p.index, columns=dc.branches)
    flows.to_parquet(out / "branch_flows.parquet")
    L = np.abs(F) / dc.s_nom[:, None]
    stats = pd.DataFrame({
        "bus0": dc.bus0, "bus1": dc.bus1, "s_nom": dc.s_nom,
        "v_nom": n.buses.v_nom.reindex(dc.bus0).to_numpy(),
        "mean_loading": L.mean(1), "p95_loading": np.quantile(L, 0.95, axis=1),
        "hours_ge_90pct": (L >= 0.9).sum(1), "hours_at_limit": (L > 0.999).sum(1),
        "mean_flow_MW": F.mean(1),
    }, index=dc.branches)
    stats.index.name = "branch"
    stats.to_csv(out / "branch_loading_stats.csv")
    fig_loading_map(n, dc, stats.mean_loading.to_numpy(), out / "line_loading_map.png")
    fig_loading_hexbin(n, dc, stats.mean_loading.to_numpy(), out / "line_loading_hexbin.png")
    summary = {"max_reconstruction_error": err,
               "mean_loading_capacity_weighted": float((stats.mean_loading * stats.s_nom).sum()
                                                       / stats.s_nom.sum()),
               "branches_mean_loading_ge_50pct": int((stats.mean_loading >= 0.5).sum()),
               "branches_with_hours_ge_90pct": int((stats.hours_ge_90pct > 0).sum())}
    (out / "line_loading_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
