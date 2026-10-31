"""Aggregate the daily OPF results into month tables, figures and a Markdown report."""

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
from matplotlib.colors import LinearSegmentedColormap

# Reference palette (dataviz skill): categorical slots in fixed order, sequential blue ramp
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

# Fuel -> reporting group (<= 8 groups so each gets a fixed categorical slot)
GROUPS = {"nuclear": "Nuclear", "coal": "Coal", "ng": "Gas", "wind": "Wind", "hydro": "Hydro",
          "dfo": "Oil", "jetfuel": "Oil", "wood": "Other", "wasteheat": "Other",
          "other": "Other", "load shedding": "Load shedding"}
ORDER = ["Nuclear", "Coal", "Gas", "Wind", "Hydro", "Oil", "Other", "Load shedding"]
COLOR = dict(zip(ORDER, SLOTS))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
})


def load_days(days_dir):
    days = sorted(p for p in Path(days_dir).iterdir() if (p / "status.json").exists())
    status = pd.DataFrame([json.loads((d / "status.json").read_text()) for d in days])
    gen_p = pd.concat([pd.read_parquet(d / "gen_p.parquet") for d in days])
    lmp = pd.concat([pd.read_parquet(d / "lmp.parquet") for d in days])
    loading = pd.concat([pd.read_parquet(d / "branch_loading.parquet") for d in days])
    return status, gen_p, lmp, loading


def fig_dispatch(by_group, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    groups = [g for g in ORDER if g in by_group and by_group[g].max() > 1]
    gw = by_group[groups] / 1e3
    stack = ax.stackplot(gw.index, gw.T.values, colors=[COLOR[g] for g in groups],
                         edgecolor=SURFACE, linewidth=0.4, labels=groups)
    ax.set_ylabel("Dispatch [GW]")
    ax.set_xlim(gw.index[0], gw.index[-1])
    ax.margins(y=0)
    ax.legend(stack[::-1], groups[::-1], loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title("Hourly dispatch by fuel group (UTC)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_prices(lmp, load_by_bus, path):
    w = load_by_bus.reindex(columns=lmp.columns).fillna(0)
    weighted = (lmp * w).sum(axis=1) / w.sum(axis=1)
    lo, hi = lmp.quantile(0.05, axis=1), lmp.quantile(0.95, axis=1)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(lmp.index, lo, hi, color="#b7d3f6", linewidth=0, label="5th-95th percentile of buses")
    ax.plot(weighted.index, weighted, color=SLOTS[0], linewidth=2, label="Load-weighted mean")
    ax.set_ylabel("LMP [$/MWh]")
    ax.set_xlim(lmp.index[0], lmp.index[-1])
    ax.legend(loc="upper left", frameon=False, ncols=2)
    ax.set_title("Nodal prices (UTC)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return weighted


def _map_axes(n):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_aspect(1 / np.cos(np.deg2rad(n.buses.y.mean())))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def _segments(n, names):
    out = []
    for name in names:
        c, b = name.split(":", 1)
        row = n.components[c].static.loc[b]
        out.append([(n.buses.at[row.bus0, "x"], n.buses.at[row.bus0, "y"]),
                    (n.buses.at[row.bus1, "x"], n.buses.at[row.bus1, "y"])])
    return out


def fig_lmp_map(n, mean_lmp, path):
    fig, ax = _map_axes(n)
    all_br = [f"Line:{i}" for i in n.lines.index] + [f"Transformer:{i}" for i in n.transformers.index]
    ax.add_collection(LineCollection(_segments(n, all_br), colors=GRID, linewidths=0.3, zorder=0))
    lo, hi = np.nanpercentile(mean_lmp, [2, 98])
    sc = ax.scatter(n.buses.loc[mean_lmp.index, "x"], n.buses.loc[mean_lmp.index, "y"],
                    c=mean_lmp.clip(lo, hi), cmap=SEQ, s=3, linewidths=0, zorder=1)
    ax.autoscale()
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Mean LMP [$/MWh] (2nd-98th pct. clipped)")
    ax.set_title("Month-average nodal price", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_congestion_map(n, hours_binding, path):
    fig, ax = _map_axes(n)
    all_br = [f"Line:{i}" for i in n.lines.index] + [f"Transformer:{i}" for i in n.transformers.index]
    ax.add_collection(LineCollection(_segments(n, all_br), colors=GRID, linewidths=0.3, zorder=0))
    hot = hours_binding[hours_binding > 0].sort_values()
    if len(hot):
        lc = LineCollection(_segments(n, hot.index), array=hot.values, cmap=SEQ,
                            linewidths=1.6, zorder=1)
        lc.set_clim(0, hot.max())
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, shrink=0.6, label="Hours at thermal limit")
    ax.autoscale()
    ax.set_title(f"Congested branches ({len(hot)} reach their limit at least once)", loc="left",
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(df, floatfmt="{:,.1f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--days-dir", required=True)
    p.add_argument("--load-sources", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    n = pypsa.Network(a.network)
    status, gen_p, lmp, loading = load_days(a.days_dir)
    gen_p.index = pd.DatetimeIndex(gen_p.index)
    lmp.index = pd.DatetimeIndex(lmp.index)

    group = n.generators.carrier.map(GROUPS).reindex(gen_p.columns)
    by_group = gen_p.T.groupby(group).sum().T
    by_group.to_csv(out / "dispatch_by_group_MW.csv")
    by_carrier_MWh = gen_p.T.groupby(n.generators.carrier.reindex(gen_p.columns)).sum().T.sum()

    load = n.loads_t.p_set.loc[gen_p.index]
    load_by_bus = load.T.groupby(n.loads.bus).sum().T
    fig_dispatch(by_group, out / "dispatch.png")
    weighted_lmp = fig_prices(lmp, load_by_bus, out / "lmp.png")
    mean_lmp = lmp.mean()
    fig_lmp_map(n, mean_lmp, out / "lmp_map.png")
    binding = (loading > 0.999)
    hours_binding = binding.sum()
    fig_congestion_map(n, hours_binding, out / "congestion_map.png")

    # Tables
    energy = pd.DataFrame({"TWh": by_carrier_MWh / 1e6})
    energy["share %"] = 100 * energy.TWh / energy.TWh.sum()
    cap = n.generators.groupby("carrier").p_nom.sum()
    energy["capacity GW"] = cap.reindex(energy.index) / 1e3
    energy["capacity factor %"] = 100 * by_carrier_MWh / (cap.reindex(energy.index) * len(gen_p))
    energy = energy.sort_values("TWh", ascending=False)
    energy.index.name = "carrier"
    energy.to_csv(out / "energy_by_carrier.csv")

    state_lmp = pd.DataFrame({
        "mean LMP": lmp.mean().groupby(n.buses.state).mean(),
        "load-weighted LMP": (lmp * load_by_bus.reindex(columns=lmp.columns).fillna(0)).sum().groupby(n.buses.state).sum()
        / load_by_bus.sum().groupby(n.buses.state).sum(),
        "load TWh": load.sum().groupby(n.loads.state).sum() / 1e6,
    }).sort_values("load TWh", ascending=False)
    state_lmp.index.name = "state"
    state_lmp.to_csv(out / "prices_by_state.csv")

    top = hours_binding.sort_values(ascending=False).head(15)
    top_tbl = pd.DataFrame({"hours at limit": top})
    for c in ("bus0", "bus1", "s_nom"):
        top_tbl[c] = [n.components[k.split(":")[0]].static.at[k.split(":", 1)[1], c] for k in top.index]
    top_tbl["states"] = [f"{n.buses.at[b0, 'state']}-{n.buses.at[b1, 'state']}"
                         for b0, b1 in zip(top_tbl.bus0, top_tbl.bus1)]
    top_tbl.index.name = "branch"
    hours_binding.rename("hours_at_limit").to_csv(out / "branch_hours_at_limit.csv")

    status = status.set_index("day")
    status.to_csv(out / "daily_status.csv")
    sources = json.loads(Path(a.load_sources).read_text())
    src = pd.DataFrame({k: {"source": v["source"]} for k, v in sources.items()}).T
    src.index.name = "state"

    report = f"""# Midwest24k DC-OPF results: {gen_p.index[0]:%B %Y}

Generated by `scripts/summarize.py`. Snapshots are hourly, labelled by hour start in UTC.

## Headline numbers

| Metric | Value |
|---|---|
| Days solved (optimal) | {(status.status.str[1] == 'optimal').sum()} / {len(status)} |
| Hours | {len(gen_p)} |
| Load served | {load.sum().sum() / 1e6:,.2f} TWh (peak {load.sum(axis=1).max() / 1e3:,.1f} GW) |
| Total variable cost (objective) | ${status.objective.sum() / 1e6:,.1f} M |
| Load shed | {status.shed_MWh.sum():,.1f} MWh |
| Load-weighted mean LMP | ${weighted_lmp.mean():,.2f}/MWh (hourly range {weighted_lmp.min():,.2f} – {weighted_lmp.max():,.2f}) |
| Branches reaching their limit at least once | {(hours_binding > 0).sum():,} of {len(hours_binding):,} |
| Mean solve time per day | {status.seconds_optimize.mean():,.0f} s (+ {status.seconds_load.mean():,.0f} s loading) |

## Dispatch

![dispatch](dispatch.png)

{md_table(energy)}

## Prices

![lmp](lmp.png)

![lmp map](lmp_map.png)

{md_table(state_lmp, "{:,.2f}")}

## Congestion

![congestion](congestion_map.png)

{md_table(top_tbl, "{:,.0f}")}

## Daily solves

{md_table(status[["objective", "load_MWh", "shed_MWh", "seconds_optimize"]], "{:,.0f}")}

## Load data sources

{md_table(src)}
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)
