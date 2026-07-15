import numpy as np

# Fields that belong in the ppc; everything else is a non-standard extension.
_STANDARD = {"baseMVA", "bus", "gen", "branch", "gencost", "version", "areas"}


def load_matpower_case(path):
    """
    Load a MATPOWER .mat case into a PYPOWER-style ppc dict + a dict of the
    non-standard fields (genfuel, genid, branchid, ...).

    Verified against PyPSA 1.2.4 / PYPOWER 5.1 / SciPy 1.17:
        ppc, extras = load_matpower_case("case.mat")
        n = pypsa.Network()
        n.import_from_pypower_ppc(ppc, overwrite_zero_s_nom=<big number>)
        n.generators["carrier"] = extras["genfuel"]   # row order is preserved

    Why not just PYPOWER's loadcase()? It works for v<=7 files and even keeps the
    custom fields, but (a) it wraps scipy.io.loadmat, so it raises on v7.3/HDF5
    files, and (b) it returns genfuel as a nested object array you have to decode
    anyway. This does both cleanly and is format-agnostic.


    Co-authored by: Claude Opus 4.8
    """
    # --- read the container, regardless of MAT version -----------------------
    try:
        from scipy.io import loadmat
        raw = loadmat(path, simplify_cells=True)
        top = {k: v for k, v in raw.items() if not k.startswith("__")}
    except (NotImplementedError, ValueError):
        from pymatreader import read_mat
        top = read_mat(path)

    # --- locate the case struct (stored as 'mpc', 'mpc_orig', ... ) ----------
    struct = next((v for v in top.values() if isinstance(v, dict)), top)

    # --- split standard ppc fields from the extensions -----------------------
    ppc, extras = {}, {}
    for k, v in struct.items():
        (ppc if k in _STANDARD else extras)[k] = v

    for key in ("bus", "gen", "branch", "gencost"):
        if key in ppc:
            ppc[key] = np.atleast_2d(np.asarray(ppc[key], dtype=float))
    if "baseMVA" in ppc:
        ppc["baseMVA"] = float(np.asarray(ppc["baseMVA"]).squeeze())
    ppc.setdefault("version", "2")

    # normalise genfuel/genid/... into flat, usable arrays (row order == gen row)
    for k, v in list(extras.items()):
        arr = np.asarray(v).squeeze()
        if arr.dtype.kind in ("U", "S", "O"):
            extras[k] = np.array([str(np.asarray(x).ravel()[0]) if np.asarray(x).size
                                  else str(x) for x in np.atleast_1d(arr)])
        else:
            extras[k] = np.atleast_1d(arr)
    return ppc, extras