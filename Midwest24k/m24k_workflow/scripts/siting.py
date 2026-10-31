"""Screen the solved month for a new wind farm and a new battery (sizes in config/siting.yaml).

Price-taker screening on the solved base case. No new OPF is run; see the report's limitations.

1. Price decomposition. Every hourly LMP is split into an energy price and one congestion term
   per branch that reaches its limit (dc_flow.decompose_lmp, exact for this lossless DC OPF).
   Removing the congestion term of branches listed as artefacts (LOGBOOK F12) gives
   "adjusted" prices. `price_basis` picks which prices rank the sites (raw = the solved LMPs);
   values on the other basis are reported alongside. Re-solving two days with the artefact
   branch uprated matched the spatial pattern of raw prices better than adjusted ones
   (LOGBOOK section 8), so raw is the default.
2. Deliverability. For a site at bus k with PTDF column d (injection at k, withdrawal spread
   over load) the injection that keeps every branch with |d_b| >= PTDF_CUTOFF within its rating
   in hour t is   H_t = min_b (s_nom_b - sign(d_b) F_bt) / |d_b|   (hosting capacity, >= 0),
   and the same with -d for withdrawal. Artefact branches are left out of this test.
3. Wind:    value = sum_t min(size * cf_t, H_t) * price_t; curtailment = what H_t cuts off.
   Battery: perfect-foresight arbitrage LP over the month with power bounds min(P, H_t),
            round-trip losses, a daily throughput cap and a degradation cost.
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
import scipy.sparse as sp
import yaml
from matplotlib.collections import LineCollection
from scipy.optimize import linprog

from build_wind_profiles import from_atlite
from dc_flow import DCFlow, decompose_lmp

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#d6d5cd"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PTDF_CUTOFF = 0.05  # distribution-factor cutoff, as in interconnection screening studies
LOCAL_KM = 150      # radius for "load nearby"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


# ----------------------------------------------------------------------------- helpers
def utc(idx):
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def km_xy(x, y, lat0):
    return np.c_[np.asarray(x) * 111.32 * np.cos(lat0), np.asarray(y) * 110.57]


def bus_table(n, dc, stats, load_MW, lat0):
    """Per-bus descriptors: incident branch capacity and loading, load within LOCAL_KM."""
    inc = pd.concat([pd.DataFrame({"bus": dc.bus0, "s_nom": dc.s_nom, "ml": stats.mean_loading.values}),
                     pd.DataFrame({"bus": dc.bus1, "s_nom": dc.s_nom, "ml": stats.mean_loading.values})])
    inc["mls"] = inc.ml * inc.s_nom
    g = inc.groupby("bus")
    t = n.buses[["x", "y", "v_nom", "state", "location"]].copy()
    t["incident_MVA"] = g.s_nom.sum().reindex(t.index).fillna(0)
    t["n_branches"] = g.size().reindex(t.index).fillna(0).astype(int)
    t["incident_mean_loading"] = (g.mls.sum() / g.s_nom.sum()).reindex(t.index)
    from scipy.spatial import cKDTree
    xy = km_xy(t.x, t.y, lat0)
    tree = cKDTree(xy)
    lb = load_MW.reindex(t.index).fillna(0).to_numpy()
    t["load_150km_GW"] = [lb[i].sum() / 1e3 for i in tree.query_ball_point(xy, LOCAL_KM)]
    return t


def hosting(d, F, s_nom, cutoff=PTDF_CUTOFF):
    """Hourly hosting capacity [MW] for injection along PTDF column d, and for withdrawal."""
    sel = np.flatnonzero(np.abs(d) >= cutoff)
    if len(sel) == 0:
        inf = np.full(F.shape[1], np.inf)
        return inf, inf, np.full(F.shape[1], -1)
    dd, Fs, sn = d[sel][:, None], F[sel], s_nom[sel][:, None]
    up = np.clip((sn - np.sign(dd) * Fs) / np.abs(dd), 0, None)    # injection
    down = np.clip((sn + np.sign(dd) * Fs) / np.abs(dd), 0, None)  # withdrawal
    return up.min(0), down.min(0), sel[up.argmin(0)]


def footprint(d, stats, cutoff=PTDF_CUTOFF):
    sel = np.abs(d) >= cutoff
    w = np.abs(d[sel])
    return float((stats.mean_loading.values[sel] * w).sum() / w.sum()) if w.sum() else np.nan, int(sel.sum())


def congestion_effect(d, inj, F, binding, mu_full):
    """Flow change on branch-hours at their limit for net injection inj (MW, + = injection).
    Returns relief and aggravation in MWh and valued at the hourly congestion price mu, plus
    the branch with the largest relief and the one with the largest aggravation."""
    rows = np.flatnonzero(binding.any(1) & (np.abs(d) > 1e-4))
    if len(rows) == 0:
        return 0.0, 0.0, 0.0, 0.0, None, None
    toward = np.sign(F[rows]) * d[rows, None] * inj[None, :]  # > 0 pushes further into the limit
    b = binding[rows]
    rel = np.where(b, np.clip(-toward, 0, None), 0)
    wor = np.where(b, np.clip(toward, 0, None), 0)
    mu = np.abs(mu_full[rows])
    best = rows[rel.sum(1).argmax()] if rel.sum() > 0 else None
    worst = rows[wor.sum(1).argmax()] if wor.sum() > 0 else None
    return float(rel.sum()), float(wor.sum()), float((rel * mu).sum()), float((wor * mu).sum()), best, worst


def pick_diverse(df, score, n_sites, spacing_km, max_per_state, lat0, flag_col="artefact_flag"):
    chosen = []
    xy = km_xy(df.x, df.y, lat0)
    pos = {b: i for i, b in enumerate(df.index)}
    for b in df.sort_values(score, ascending=False).index:
        if df.at[b, flag_col]:
            continue
        if sum(df.at[c, "state"] == df.at[b, "state"] for c in chosen) >= max_per_state:
            continue
        if all(np.hypot(*(xy[pos[b]] - xy[pos[c]])) >= spacing_km for c in chosen):
            chosen.append(b)
        if len(chosen) == n_sites:
            break
    return chosen


def price_drivers(bus_pos, R, mu, branches, art_mask, top=2):
    """Mean congestion contribution [$/MWh] of each branch to this bus's price; the largest
    non-artefact ones, formatted for the report."""
    contrib = -(mu * R[:, bus_pos][None, :]).mean(0)  # mean over hours, per branch
    order = np.argsort(-np.abs(np.where(art_mask, 0, contrib)))[:top]
    return "; ".join(f"{branches[k].split(':')[1]} {contrib[k]:+.2f}" for k in order
                     if abs(contrib[k]) >= 0.05 and not art_mask[k]) or "none > $0.05"


# ----------------------------------------------------------------------------- battery LP
def battery_lp(price, pmax_dis, pmax_ch, cfg, days):
    """Perfect-foresight arbitrage. Variables [c, d, s] per hour. Returns (value, c, d)."""
    T = len(price)
    P, E = cfg["power_MW"], cfg["energy_MWh"]
    eta = np.sqrt(cfg["round_trip_eff"])
    deg = cfg["degradation_cost"]
    # minimise  sum p c - sum (p - deg) d
    cost = np.r_[price, -(price - deg), np.zeros(T)]
    I = sp.identity(T, format="csr")
    shift = sp.eye(T, k=-1, format="csr")
    # s_t - s_{t-1} - eta c_t + d_t / eta = 0 (s_0 = E/2 enters the RHS of t=0)
    A_eq = sp.hstack([-eta * I, I / eta, I - shift])
    b_eq = np.zeros(T)
    b_eq[0] = E / 2
    # c_t + d_t <= P (no simultaneous full charge and discharge); daily discharge cap;
    # end state >= start state
    day_rows = sp.csr_matrix((np.ones(T), (days, np.arange(T))))
    A_ub = sp.vstack([
        sp.hstack([I, I, sp.csr_matrix((T, T))]),
        sp.hstack([sp.csr_matrix((day_rows.shape[0], T)), day_rows,
                   sp.csr_matrix((day_rows.shape[0], T))]),
        sp.hstack([sp.csr_matrix((1, 2 * T)), sp.csr_matrix(([-1.0], ([0], [T - 1])), shape=(1, T))]),
    ])
    b_ub = np.r_[np.full(T, P), np.full(day_rows.shape[0], cfg["max_cycles_per_day"] * E), -E / 2]
    bounds = ([(0, min(P, h)) for h in pmax_ch] + [(0, min(P, h)) for h in pmax_dis]
              + [(0, E)] * T)
    r = linprog(cost, A_ub=A_ub.tocsc(), b_ub=b_ub, A_eq=A_eq.tocsc(), b_eq=b_eq, bounds=bounds,
                method="highs")
    assert r.status == 0, r.message
    c, d = r.x[:T], r.x[T:2 * T]
    return float(price @ (d - c) - deg * d.sum()), c, d


# ----------------------------------------------------------------------------- figures
def base_map(n, ax):
    ax.set_aspect(1 / np.cos(np.deg2rad(n.buses.y.mean())))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_sites(ax, sites, marker, color, prefix):
    for i, (b, r) in enumerate(sites.iterrows(), 1):
        ax.scatter(r.x, r.y, s=110, marker=marker, facecolor=color, edgecolor=INK, linewidth=0.9, zorder=5)
        ax.annotate(f"{prefix}{i}", (r.x, r.y), xytext=(6, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))


def branch_segments(n, dc):
    x0, y0 = n.buses.x.reindex(dc.bus0).to_numpy(), n.buses.y.reindex(dc.bus0).to_numpy()
    x1, y1 = n.buses.x.reindex(dc.bus1).to_numpy(), n.buses.y.reindex(dc.bus1).to_numpy()
    return np.stack([np.c_[x0, y0], np.c_[x1, y1]], axis=1)


def fig_sites(n, dc, stats, grid, bus_metric, wind_sites, batt_sites, binding_hours, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    segs = branch_segments(n, dc)
    hot = stats.mean_loading.to_numpy() >= 0.5
    congested = binding_hours > 0

    ax = axes[0]
    base_map(n, ax)
    hb = ax.hexbin(grid.x, grid.y, C=grid.wind_cf, gridsize=48, cmap="viridis", linewidths=0.15,
                   edgecolors=SURFACE)
    ax.add_collection(LineCollection(segs[stats.v_nom.to_numpy() >= 230], colors="#ffffff",
                                     linewidths=0.35, alpha=0.6))
    ax.add_collection(LineCollection(segs[hot], colors=SLOTS[7], linewidths=1.2))
    draw_sites(ax, wind_sites, "^", SLOTS[3], "W")
    ax.autoscale()
    fig.colorbar(hb, ax=ax, shrink=0.55, label="Mean wind capacity factor, Aug 2015")
    ax.set_title("500 MW wind sites W1-W5\nwhite: lines >= 230 kV; red: mean loading >= 50 %",
                 loc="left", fontsize=9.5)

    ax = axes[1]
    base_map(n, ax)
    ax.add_collection(LineCollection(segs, colors=GRID, linewidths=0.25))
    m = bus_metric.dropna().sort_values()
    sc = ax.scatter(n.buses.x.reindex(m.index), n.buses.y.reindex(m.index), c=m.clip(0, 1.5), s=4,
                    cmap="magma_r", vmin=0, vmax=1.5, linewidths=0)
    ax.add_collection(LineCollection(segs[congested], colors=SLOTS[0], linewidths=2.2))
    mid = segs[congested].mean(1)
    ax.scatter(mid[:, 0], mid[:, 1], s=40, facecolor="none", edgecolor=SLOTS[0], linewidth=1.4,
               zorder=4)
    draw_sites(ax, batt_sites, "s", SLOTS[2], "B")
    ax.autoscale()
    fig.colorbar(sc, ax=ax, shrink=0.55, extend="max",
                 label="Local congestion premium: mean |adjusted LMP - energy price| [$/MWh]")
    ax.set_title("200 MW / 800 MWh battery sites B1-B5\nblue circles: branches that reach their limit "
                 "(excl. L4698)", loc="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_price_duration(price, batt_sites, system, path, basis):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    pct = np.linspace(0, 100, len(price))
    ax.plot(pct, np.sort(system)[::-1], color=MUTED, lw=2.4, label="System energy price")
    for i, b in enumerate(batt_sites.index):
        ax.plot(pct, np.sort(price[b].to_numpy())[::-1], color=SLOTS[i], lw=1.3,
                label=f"B{i + 1} {batt_sites.at[b, 'location']} ({batt_sites.at[b, 'state']})")
    ax.set_xlabel("Share of hours [%]")
    ax.set_ylabel(f"LMP, {basis} [$/MWh]")
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("Price duration curves at the battery sites, Aug 2015", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_wind_value(wind_sites, path, basis):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels = [f"W{i + 1} {r.location} ({r.state})" for i, (_, r) in enumerate(wind_sites.iterrows())]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, wind_sites.value_k_per_MW, color=SLOTS[3], label="Deliverable market value")
    ax.barh(y, wind_sites.curtail_value_k_per_MW, left=wind_sites.value_k_per_MW, color=SLOTS[7],
            label="Lost to grid limits")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel(f"$k per MW installed, Aug 2015 ({basis} prices)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2)
    ax.set_title("Wind sites: one-month energy market value", loc="left", pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_artefact_component(n, comp, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    base_map(n, ax)
    v = np.nanpercentile(np.abs(comp), 98)
    sc = ax.scatter(n.buses.x.reindex(comp.index), n.buses.y.reindex(comp.index), c=comp.clip(-v, v),
                    cmap="RdBu_r", vmin=-v, vmax=v, s=3, linewidths=0)
    ax.autoscale()
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Mean price effect [$/MWh] (2-98 pct. clipped)")
    ax.set_title("How much L4698 congestion adds to each bus's price, Aug 2015", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--siting-config", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--days-dir", required=True)
    p.add_argument("--flows", required=True)
    p.add_argument("--branch-stats", required=True)
    p.add_argument("--wind-cutout", required=True)
    p.add_argument("--resource-grid", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    scfg = yaml.safe_load(open(a.siting_config))
    wcfg, bcfg = scfg["wind_site"], scfg["battery"]

    n = pypsa.Network(a.network)
    dc = DCFlow(n)
    flows = pd.read_parquet(a.flows)
    assert (flows.columns == dc.branches).all()
    F = flows.to_numpy(dtype=float).T
    idx = utc(flows.index)
    T = len(idx)
    day_of = (idx.normalize() - idx.normalize()[0]).days.to_numpy()
    stats = pd.read_csv(a.branch_stats, index_col=0, dtype={"bus0": str, "bus1": str}).loc[dc.branches]
    binding = np.abs(F) / dc.s_nom[:, None] > 0.999
    days_dirs = sorted(p for p in Path(a.days_dir).iterdir() if (p / "status.json").exists())
    lmp = pd.concat([pd.read_parquet(d / "lmp.parquet") for d in days_dirs])
    lmp.index = utc(lmp.index)
    lmp = lmp.loc[idx, dc.buses]
    load = n.loads_t.p_set.loc[flows.index]
    load_bus = load.T.groupby(n.loads.bus.reindex(load.columns).to_numpy()).sum().T
    load_bus.index = idx
    sink = load_bus.sum() / load_bus.sum().sum()
    lat0 = np.deg2rad(n.buses.y.mean())
    buses = bus_table(n, dc, stats, load_bus.mean(), lat0)
    bpos = pd.Series(np.arange(len(dc.buses)), index=dc.buses)

    # ------------------------------------------------------------------ price decomposition
    cong = np.flatnonzero(binding.any(1))
    R = dc.ptdf_rows(cong, sink)
    energy, mu, resid = decompose_lmp(lmp.to_numpy(), R)
    print(f"LMP decomposition over {len(cong)} congested branches: max residual {resid:.2e} $/MWh")
    cong_names = dc.branches[cong]
    art_mask = np.isin(cong_names, scfg["artefact_branches"])
    art_comp = -(mu[:, art_mask] @ R[art_mask])  # T x buses, $/MWh added by artefact branches
    adjusted = lmp - art_comp
    price, alt = (lmp, adjusted) if scfg["price_basis"] == "raw" else (adjusted, lmp)
    mu_full = np.zeros_like(F)                      # congestion price per branch-hour
    mu_full[cong] = mu.T
    art_rows = dc.branches.get_indexer(scfg["artefact_branches"])
    s_nom_test = dc.s_nom.copy()
    s_nom_test[art_rows] = 1e9                      # artefact branches don't limit hosting
    binding_test = binding.copy()
    binding_test[art_rows] = False

    mean_comp = pd.DataFrame(-(mu.mean(0)[:, None] * R).T, index=dc.buses, columns=cong_names)
    decomp = pd.DataFrame({"mean_lmp": lmp.mean(), "energy": energy.mean(),
                           "artefact_component": art_comp.mean(0),
                           "other_congestion": mean_comp.loc[:, ~art_mask].sum(1)})
    decomp = decomp.join(mean_comp).join(n.buses[["state", "location"]])
    decomp.to_csv(out / "lmp_decomposition_by_bus.csv")
    pd.DataFrame(mu, index=idx, columns=cong_names).to_csv(out / "congestion_prices_hourly.csv")
    by_state = decomp.groupby("state")[["mean_lmp", "energy", "artefact_component",
                                        "other_congestion"]].mean().sort_values("mean_lmp")
    by_state.to_csv(out / "lmp_decomposition_by_state.csv")
    fig_artefact_component(n, decomp.artefact_component, out / "artefact_price_effect.png")
    flag_bus = (decomp.artefact_component.abs() > scfg["artefact_price_threshold"])

    # ------------------------------------------------------------------ wind
    wc = buses[(buses.v_nom >= wcfg["min_kv"])].index
    cf = from_atlite(a.wind_cutout, buses.x[wc], buses.y[wc], cfg["renewables"]["wind"]["turbine"])
    cf = cf.reindex(idx)
    assert cf.notna().all().all()
    L, Lalt = price[wc], alt[wc]
    w = pd.DataFrame(index=wc)
    w["cf"] = cf.mean()
    w["value_raw_k_per_MW"] = (cf * L).sum() / 1e3
    w["value_alt_basis_k_per_MW"] = (cf * Lalt).sum() / 1e3
    w["capture_price"] = (cf * L).sum() / cf.sum()
    w["mean_price"] = L.mean()
    w["value_factor"] = w.capture_price / w.mean_price
    w["neg_price_hours"] = (L < 0).sum()
    w = w.join(buses)
    short = w.sort_values("value_raw_k_per_MW", ascending=False).head(wcfg["shortlist"]).index
    ptdf = dc.ptdf(short, sink)
    size = wcfg["size_MW"]
    rows = {}
    for j, b in enumerate(short):
        d = ptdf[:, j]
        h_up, _, lim_up = hosting(d, F, s_nom_test)
        gen = size * cf[b].to_numpy()
        deliv = np.minimum(gen, h_up)
        curt = gen - deliv
        rel, wor, rel_usd, wor_usd, best, worst = congestion_effect(d, deliv, F, binding_test, mu_full)
        lim = pd.Series(lim_up[curt > 1e-6])
        fl, nfoot = footprint(d, stats)
        rows[b] = {
            "hosting_p5_MW": float(np.percentile(h_up, 5)), "hosting_median_MW": float(np.median(h_up)),
            "curtail_pct": 100 * curt.sum() / gen.sum(),
            "value_k_per_MW": float(deliv @ L[b].to_numpy()) / size / 1e3,
            "curtail_value_k_per_MW": float(curt @ np.clip(L[b].to_numpy(), 0, None)) / size / 1e3,
            "limiting_branch": dc.branches[lim.mode()[0]] if len(lim) else "",
            "congestion_relief_MWh": rel, "congestion_added_MWh": wor,
            "congestion_relief_usd": rel_usd, "congestion_added_usd": wor_usd,
            "most_loaded_congested_branch": dc.branches[worst] if worst is not None else "",
            "footprint_mean_loading": fl, "footprint_branches": nfoot,
            "price_drivers": price_drivers(bpos[b], R, mu, cong_names, art_mask),
        }
    ws = w.loc[short].join(pd.DataFrame(rows).T.infer_objects())
    ws["artefact_component"] = decomp.artefact_component.reindex(ws.index)
    ws["artefact_flag"] = flag_bus.reindex(ws.index)
    wg = n.generators[n.generators.carrier == "wind"]
    wxy = km_xy(wg.bus.map(n.buses.x), wg.bus.map(n.buses.y), lat0)
    sxy = km_xy(ws.x, ws.y, lat0)
    dist = np.hypot(sxy[:, None, 0] - wxy[None, :, 0], sxy[:, None, 1] - wxy[None, :, 1])
    ws["existing_wind_50km_MW"] = (dist <= 50) @ wg.p_nom.to_numpy()
    ws.sort_values("value_k_per_MW", ascending=False).to_csv(out / "wind_screen.csv")
    w_pick = pick_diverse(ws, "value_k_per_MW", wcfg["n_sites"], wcfg["min_spacing_km"],
                          wcfg["max_per_state"], lat0)
    wind_sites = ws.loc[w_pick]

    # ------------------------------------------------------------------ battery
    bc = buses[(buses.v_nom >= bcfg["min_kv"])].index
    Lb = price[bc].to_numpy()
    nd = T // 24
    daily = np.sort(Lb[: nd * 24].reshape(nd, 24, -1), axis=1)
    eta = np.sqrt(bcfg["round_trip_eff"])
    spread = daily[:, -4:].mean(1) * eta - daily[:, :4].mean(1) / eta - bcfg["degradation_cost"]
    b = pd.DataFrame(index=bc)
    b["daily_spread"] = (daily[:, -4:].mean(1) - daily[:, :4].mean(1)).mean(0)
    b["fast_value_k"] = np.clip(spread, 0, None).sum(0) * bcfg["energy_MWh"] / 1e3
    b["mean_price"] = Lb.mean(0)
    b["min_price"] = Lb.min(0)
    b["max_price"] = Lb.max(0)
    b = b.join(buses)
    # one representative bus per distinct price series (many buses share a series)
    key = pd.Series([hash(np.round(Lb[:, i], 1).tobytes()) for i in range(Lb.shape[1])], index=bc)
    b["series_group_size"] = key.map(key.value_counts())
    rep = (b.assign(key=key).sort_values(["fast_value_k", "incident_MVA"], ascending=False)
           .drop_duplicates("key"))
    short_b = rep.head(bcfg["shortlist"]).index
    ptdf_b = dc.ptdf(short_b, sink)
    rows = {}
    for j, bus in enumerate(short_b):
        d = ptdf_b[:, j]
        h_up, h_dn, _ = hosting(d, F, s_nom_test)
        pr = price[bus].to_numpy()
        val, c, dis = battery_lp(pr, h_up, h_dn, bcfg, day_of)
        val_free, _, _ = battery_lp(pr, np.full(T, np.inf), np.full(T, np.inf), bcfg, day_of)
        val_alt, _, _ = battery_lp(alt[bus].to_numpy(), h_up, h_dn, bcfg, day_of)
        rel, wor, rel_usd, wor_usd, best, worst = congestion_effect(d, dis - c, F, binding_test, mu_full)
        rows[bus] = {
            "arbitrage_k": val / 1e3, "arbitrage_no_grid_limit_k": val_free / 1e3,
            "arbitrage_alt_basis_k": val_alt / 1e3,
            "usd_per_kW_month": val / (bcfg["power_MW"] * 1e3),
            "discharge_MWh": dis.sum(), "cycles": dis.sum() / bcfg["energy_MWh"],
            "avg_charge_price": float(pr @ c / c.sum()) if c.sum() else np.nan,
            "avg_discharge_price": float(pr @ dis / dis.sum()) if dis.sum() else np.nan,
            "hosting_dis_p5_MW": float(np.percentile(h_up, 5)),
            "hosting_ch_p5_MW": float(np.percentile(h_dn, 5)),
            "congestion_relief_MWh": rel, "congestion_added_MWh": wor,
            "congestion_relief_usd": rel_usd, "congestion_added_usd": wor_usd,
            "relieved_branch": dc.branches[best] if best is not None else "",
            "relieved_branch_hours_binding": int(binding[best].sum()) if best is not None else 0,
            "simultaneous_MWh": float(np.minimum(c, dis).sum()),
            "price_drivers": price_drivers(bpos[bus], R, mu, cong_names, art_mask),
        }
    bs = b.loc[short_b].join(pd.DataFrame(rows).T.infer_objects())
    bs["artefact_component"] = decomp.artefact_component.reindex(bs.index)
    bs["artefact_flag"] = flag_bus.reindex(bs.index)
    bs.sort_values("arbitrage_k", ascending=False).to_csv(out / "battery_screen.csv")
    b_pick = pick_diverse(bs, "arbitrage_k", bcfg["n_sites"], bcfg["min_spacing_km"],
                          bcfg["max_per_state"], lat0)
    batt_sites = bs.loc[b_pick]
    b.to_csv(out / "battery_fast_screen_all_buses.csv")

    # ------------------------------------------------------------------ outputs
    grid = pd.read_csv(a.resource_grid)
    local_premium = pd.Series(np.abs(adjusted.to_numpy() - energy[:, None]).mean(0), index=dc.buses)
    local_premium.rename("local_congestion_premium").to_csv(out / "local_congestion_premium.csv")
    fig_sites(n, dc, stats, grid, local_premium, wind_sites, batt_sites,
              binding_test.sum(1), out / "siting_map.png")
    fig_price_duration(price, batt_sites, energy, out / "battery_price_duration.png", scfg["price_basis"])
    fig_wind_value(wind_sites, out / "wind_value.png", scfg["price_basis"])

    wcols = ["location", "state", "v_nom", "x", "y", "cf", "capture_price", "value_factor",
             "value_k_per_MW", "value_alt_basis_k_per_MW", "curtail_pct", "hosting_p5_MW",
             "hosting_median_MW", "incident_MVA", "incident_mean_loading", "footprint_mean_loading",
             "load_150km_GW", "existing_wind_50km_MW", "congestion_relief_usd", "congestion_added_usd",
             "most_loaded_congested_branch", "limiting_branch", "price_drivers", "artefact_component"]
    bcols = ["location", "state", "v_nom", "x", "y", "arbitrage_k", "arbitrage_no_grid_limit_k",
             "arbitrage_alt_basis_k", "usd_per_kW_month", "cycles", "avg_charge_price",
             "avg_discharge_price", "mean_price", "min_price", "max_price", "daily_spread",
             "hosting_dis_p5_MW", "hosting_ch_p5_MW", "incident_MVA", "load_150km_GW",
             "congestion_relief_MWh", "congestion_relief_usd", "congestion_added_usd",
             "relieved_branch", "relieved_branch_hours_binding", "price_drivers",
             "series_group_size", "simultaneous_MWh", "artefact_component"]
    wind_sites[wcols].to_csv(out / "wind_sites.csv")
    batt_sites[bcols].to_csv(out / "battery_sites.csv")
    raw_rank_b = bs.sort_values("arbitrage_alt_basis_k", ascending=False).head(10)
    summary = {
        "decomposition_max_residual": resid,
        "congested_branches": list(cong_names),
        "mean_congestion_price_by_branch": dict(zip(cong_names, np.abs(mu).mean(0).round(2))),
        "state_decomposition": by_state.round(3).reset_index().to_dict("records"),
        "wind_candidates": int(len(wc)), "wind_shortlist": int(len(short)),
        "battery_candidates": int(len(bc)), "battery_distinct_series": int(key.nunique()),
        "battery_shortlist": int(len(short_b)),
        "system_energy_price_mean": float(energy.mean()),
        "price_basis": scfg["price_basis"],
        "wind_sites": wind_sites[wcols].round(3).reset_index().to_dict("records"),
        "battery_sites": batt_sites[bcols].round(3).reset_index().to_dict("records"),
        "top_battery": bs.sort_values("arbitrage_k", ascending=False).head(12)[
            ["location", "state", "arbitrage_k", "arbitrage_alt_basis_k", "artefact_component",
             "artefact_flag", "relieved_branch"]].round(2).reset_index().to_dict("records"),
        "top_battery_alt_basis": raw_rank_b[["location", "state", "arbitrage_alt_basis_k",
                                              "arbitrage_k", "artefact_component", "artefact_flag"]]
            .round(2).reset_index().to_dict("records"),
        "top_wind": ws.sort_values("value_k_per_MW", ascending=False).head(12)[
            ["location", "state", "cf", "value_k_per_MW", "value_alt_basis_k_per_MW", "curtail_pct",
             "artefact_flag"]].round(3).reset_index().to_dict("records"),
        "top_wind_alt_basis": ws.sort_values("value_alt_basis_k_per_MW", ascending=False).head(8)[
            ["location", "state", "cf", "value_alt_basis_k_per_MW", "value_k_per_MW",
             "artefact_component"]].round(3).reset_index().to_dict("records"),
        "wind_curtailment_any": ws[ws.curtail_pct > 0.01][["location", "state", "curtail_pct",
                                                           "limiting_branch"]].round(2)
            .reset_index().to_dict("records"),
    }
    (out / "siting_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(by_state.round(2).to_string())
    print(wind_sites[wcols].round(2).to_string())
    print(batt_sites[bcols].round(2).to_string())
