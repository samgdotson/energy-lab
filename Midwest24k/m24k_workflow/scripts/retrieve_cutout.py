"""Download an ERA5 cutout for the analysis month with atlite.

Wind only by default (the dispatch model); `--features influx temperature` gives the solar
cutout used for the resource maps.

Requires CDS API credentials in ~/.cdsapirc. On any failure an empty marker is not written;
the rule fails and `renewables.wind.source: synthetic` can be used instead.
"""

import argparse
import calendar
import logging

import atlite

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--bounds", type=float, nargs=4, required=True, help="x0 y0 x1 y1")
    p.add_argument("--features", nargs="+", default=["wind"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    last = calendar.monthrange(a.year, a.month)[1]
    x0, y0, x1, y1 = a.bounds
    cutout = atlite.Cutout(
        path=a.out, module="era5", x=slice(x0, x1), y=slice(y0, y1),
        time=slice(f"{a.year}-{a.month:02d}-01", f"{a.year}-{a.month:02d}-{last}"),
    )
    # NetCDF instead of atlite's default GRIB: GRIB needs cfgrib + the ecCodes C library,
    # which is not available in this environment
    cutout.prepare(features=a.features, data_format="netcdf")
