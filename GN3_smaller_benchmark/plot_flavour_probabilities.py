"""Plot script for flavour probability comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import glob

from puma import Histogram, HistogramPlot
from puma.utils import logger
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader

def load_df(file_pattern, model_name, num_jets=None, batch_size=500_000):
    """Loads and filters data from HDF5 files matching a pattern using H5Reader.

    Parameters
    ----------
    file_pattern : str
        Glob pattern (e.g., "/data/*.h5") for HDF5 files
    num_jets : int, optional
        Max number of jets to load
    batch_size : int, optional
        Batch size to read at a time

    Returns
    -------
    pd.DataFrame
        Filtered jet data as a pandas DataFrame
    """
    # Expand wildcard pattern into list of file paths
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    # Define cuts 
    cuts = Cuts.from_list([
        ("pt", ">", 250000),
        ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
        # ("eventNumber", "%10==", 9),
    ])

    # Define required variables
    jet_vars = [
        "pt", "eta", "eventNumber",
        "HadronGhostTruthLabelID",
        "HadronConeExclTruthLabelID",
        "flavour_label",
        "GN3V01_pb", "GN3V01_pc", "GN3V01_ptau", "GN3V01_pud", "GN3V01_ps", "GN3V01_pg",
        f"{model_name}_pb", f"{model_name}_pc", f"{model_name}_ptau", f"{model_name}_ps",
        f"{model_name}_pud", f"{model_name}_pg"
    ]

    variables = {"jets": jet_vars}

    reader = H5Reader(
        fname=file_paths,
        jets_name="jets",
        batch_size=batch_size,
        do_remove_inf=True,
        shuffle=True
    )

    data = reader.load(variables=variables, cuts=cuts, num_jets=num_jets)

    df = pd.DataFrame(data["jets"])

    return df

# Open test file and and convert to pandas dataframe
fname='/home/xzcapfed/tmp/ckpts/epoch=003-val_loss=1.01886__test_zprime.h5'
logger.info("Loading h5 files")
model_name = "GN3V01_smaller"
df = load_df(fname, model_name)

# Initialise histogram plot
logger.info("Initializing histogram")
plot_histo = HistogramPlot(
    n_ratio_panels=0,
    ylabel="Normalised number of jets",
    xlabel="$b$-jets probability",
    logy=True,
    leg_ncol=1,
    atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV, $Z'$ events",
    atlas_second_tag="250 GeV < $p_T$ < 6 TeV, $|\eta| < 2.5$",
    atlas_brand=None,  # Deactivate ATLAS branding
    draw_errors=False,
)

# Add the ttbar histograms
u_jets = df[df["HadronConeExclTruthLabelID"] == 0]
c_jets = df[df["HadronConeExclTruthLabelID"] == 4]
b_jets = df[df["HadronConeExclTruthLabelID"] == 5]

# the "flavour" argument will add a "light-flavour jets" (or other) prefix to the label
# + set the colour to the one that is defined in puma.utils.global_config
logger.info("Plotting histogram")
plot_histo.add(
    Histogram(
        u_jets[f"{model_name}_pb"],
        bins=np.linspace(0, 1, 30),
        flavour="ujets",
        linestyle="dashed",
    )
)
plot_histo.add(
    Histogram(
        c_jets[f"{model_name}_pb"],
        bins=np.linspace(0, 1, 30),
        flavour="cjets",
        linestyle="dashdot",
    )
)
plot_histo.add(
    Histogram(
        b_jets[f"{model_name}_pb"],
        bins=np.linspace(0, 1, 30),
        flavour="bjets",
    )
)

plot_histo.draw()
plot_histo.savefig("plots/flavour_prob_zprime.png", transparent=False)