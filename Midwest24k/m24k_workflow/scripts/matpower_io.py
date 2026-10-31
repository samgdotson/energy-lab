"""Minimal MATPOWER .m reader that keeps every row, extra columns, and cell arrays.

Shared by the workflow scripts. Generalises debug/diagnose_case.py::read_m_sections to cases
with extra columns (PowerWorld exports add columns to bus/gen/branch) and to cell-array
entries containing escaped quotes ('').
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

BUS_COLS = ["BUS_I", "BUS_TYPE", "PD", "QD", "GS", "BS", "BUS_AREA", "VM", "VA",
            "BASE_KV", "ZONE", "VMAX", "VMIN", "LAM_P", "LAM_Q", "MU_VMAX", "MU_VMIN"]
GEN_COLS = ["GEN_BUS", "PG", "QG", "QMAX", "QMIN", "VG", "MBASE", "GEN_STATUS", "PMAX",
            "PMIN", "PC1", "PC2", "QC1MIN", "QC1MAX", "QC2MIN", "QC2MAX", "RAMP_AGC",
            "RAMP_10", "RAMP_30", "RAMP_Q", "APF", "MU_PMAX", "MU_PMIN", "MU_QMAX", "MU_QMIN"]
BRANCH_COLS = ["F_BUS", "T_BUS", "BR_R", "BR_X", "BR_B", "RATE_A", "RATE_B", "RATE_C",
               "TAP", "SHIFT", "BR_STATUS", "ANGMIN", "ANGMAX", "PF", "QF", "PT", "QT",
               "MU_SF", "MU_ST", "MU_ANGMIN", "MU_ANGMAX"]
DCLINE_COLS = ["F_BUS", "T_BUS", "BR_STATUS", "PF", "PT", "QF", "QT", "VF", "VT", "PMIN",
               "PMAX", "QMINF", "QMAXF", "QMINT", "QMAXT", "LOSS0", "LOSS1"]


def _numeric_block(body):
    rows = [r.split("%")[0].strip().rstrip(";").split() for r in body.strip().splitlines()]
    rows = [r for r in rows if r]
    return pd.DataFrame(np.array(rows, dtype=float))


def _cell_block(body):
    out = []
    for r in body.strip().splitlines():
        r = r.split("%")[0].strip().rstrip(";").strip()
        if r:
            out.append(r[1:-1].replace("''", "'"))
    return out


def _label(df, cols):
    k = min(len(cols), df.shape[1])
    return df.rename(columns=dict(zip(range(k), cols[:k])))


def read_matpower(case_path):
    """Return (baseMVA, dict of sections). Numeric blocks -> DataFrames, cells -> lists."""
    text = Path(case_path).read_text()
    base_mva = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text).group(1))
    s = {name: _numeric_block(body)
         for name, body in re.findall(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", text, re.S)}
    s.update({name: _cell_block(body)
              for name, body in re.findall(r"mpc\.(\w+)\s*=\s*\{(.*?)\};", text, re.S)})
    s["bus"] = _label(s["bus"], BUS_COLS)
    s["gen"] = _label(s["gen"], GEN_COLS)
    s["branch"] = _label(s["branch"], BRANCH_COLS)
    if "dcline" in s:
        s["dcline"] = _label(s["dcline"], DCLINE_COLS)
    for df, cols in ((s["bus"], ["BUS_I"]), (s["gen"], ["GEN_BUS"]),
                     (s["branch"], ["F_BUS", "T_BUS"])):
        df[cols] = df[cols].astype(int)
    return base_mva, s


def pwl_cost_table(gencost):
    """Split MODEL=1 (piecewise-linear) gencost rows into breakpoint arrays P, C."""
    assert (gencost[0] == 1).all(), "expected piecewise-linear gencost (MODEL=1)"
    ncost = int(gencost[3].iloc[0])
    assert (gencost[3] == ncost).all()
    vals = gencost.iloc[:, 4:4 + 2 * ncost].to_numpy()
    return vals[:, 0::2], vals[:, 1::2]
