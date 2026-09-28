# Midwest 24k Power System Analysis

The initial goal for this project is to run a power system analysis for the Midwest using the TAMU Midwest 24k synthetic grid system.

The synthetic grid system was downloaded from TAMU, [here](https://electricgrids.engr.tamu.edu/texas-am-perform-cases/).

## Learnings

1. It turns out that the MATPOWER versions of the TAMU datasets are saved and downloaded as a `.m` file rather than a `.mat` file.
Common tools such as
* `scipy.io.loadmat`
* `pymatreader.read_mat`
* `np.loadtxt`

are unable to read `.m` files. However, from TAMU documentation on [file formats](https://electricgrids.engr.tamu.edu/file-formats/),
> m – text file containing only power flow data with generator cost model data

these files are only text files. So, they can be read in manually via `pandas` and `readlines`.

2. `.m` files cannot hold coordinate data. But it is possible to extract the coordinates from [PowerWorld Viewer](https://www.powerworld.com/download-purchase/demo-software/powerworld-viewer-download), a free a demo software for viewing PWB files. So, I downloaded the corresponding PWB files and saved the coordinates found under `Tools and Add Ons > Geography > Buses`. (The PWB for Midwest24k is not included in this repository due to its size). 