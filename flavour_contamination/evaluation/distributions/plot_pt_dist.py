"""Produce histogram of pt distribution for true vs. contaminated b-jets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import glob

from ftag import Flavours
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader

from puma import Histogram, HistogramPlot
from puma.utils import logger

def get_details(df, label):
    '''Defines a boolean array to select the different flavour classes'''
    is_light = df[label] == 0
    is_c = df[label] == 4
    is_tau = df[label] == 15
    # is_trueb = (df[label] == 5) & (df['is_contaminated_bjet'] == False)
    # is_contaminatedb = (df[label] == 5) & (df['is_contaminated_bjet'] == True)
    is_trueb = (df[label] == 5) & (df['flavour_label'] == 0)
    is_contaminatedb = (df[label] == 5) & (df['flavour_label'] == 1)

    n_jets_trueb = sum(is_trueb)
    n_jets_contaminatedb = sum(is_contaminatedb)
    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)
    n_jets_tau = sum(is_tau)

    return is_light, is_c, is_trueb, is_contaminatedb, is_tau, n_jets_light, n_jets_c, n_jets_trueb, n_jets_contaminatedb, n_jets_tau

def load_df(file_pattern, num_jets=None, batch_size=500_000):
    """Loads and filters data from HDF5 files matching a pattern using H5Reader."""
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    # Ensure cuts match the histogram binning range
    cuts = Cuts.from_list([
        ("pt", ">", 20000),
        ("pt", "<", 250000),
        # ("pt", ">", 250000),
        # ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
    ])

    # We only need basic variables for this plot
    jet_vars = [
        "pt", "eta", "eventNumber",
        "HadronGhostTruthLabelID",
        "flavour_label",
        "pt_btagJes",
        'HadronGhostExtendedTruthLabelID'
        # "is_contaminated_bjet"
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

# ----------- Plotting ---------------
fname_ttbar = '/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_ttbar.h5'
fname_zprime = '/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_zprime.h5'
fname_qcd = '/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_qcd.h5'

fname_train='/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-contaminated-bjet/output/pp_output_train.h5'
fname_train_highstat='/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-contaminated-bjet-morestats/output/pp_output_train_vds.h5'

logger.info("Loading h5 files")
df_ttbar = load_df(fname_ttbar)
df_zprime = load_df(fname_zprime)
df_qcd = load_df(fname_qcd)
# df = pd.concat([df_zprime, df_qcd])
df = load_df(fname_train)


logger.info(f"Loaded {len(df)} jets.")
is_light, is_c, is_trueb, is_contaminatedb, is_tau, n_jets_light, n_jets_c, n_jets_trueb, n_jets_contaminatedb, n_jets_tau = get_details(df, "HadronGhostTruthLabelID")
print(f"Number light-jets : {sum(is_light)}")
print(f"Number tau-jets : {sum(is_tau)}")
print(f"Number true b-jets : {sum(is_trueb)}")
print(f"Number contaminated b-jets : {sum(is_contaminatedb)}")
print(f"Number c-jets : {sum(is_c)}")
print(f"Total : {sum(is_light) + sum(is_trueb) + sum(is_c) + sum(is_tau)}")

# Extract pt arrays and convert from MeV to GeV
pt_trueb = df["pt_btagJes"][is_trueb] / 1000.0
pt_contaminatedb = df["pt_btagJes"][is_contaminatedb] / 1000.0
pt_c = df["pt_btagJes"][is_c] / 1000.0
pt_light = df["pt_btagJes"][is_light] / 1000.0

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=0,
    ylabel="Normalised number of jets",
    # ylabel_ratio=["Ratio"],
    xlabel="$p_T$ [GeV]",
    logy=False,
    leg_ncol=1,
    figsize=(5.5, 4.5),
    atlas_second_tag="$\\sqrt{s}=13.6$ TeV, $t\\overline{t}$ & QCD events \n20 GeV < $p_T$ < 250 GeV",
    # atlas_second_tag="$\sqrt{s} = 13.6$ TeV, $Z'$ & QCD events\n250 GeV < $p_T$ < 2.5 TeV"
    # atlas_second_tag="$\sqrt{s} = 13.6$ TeV, $t\\overline{t}$, $Z'$ & QCD events"
)

pt_bins = np.arange(20, 260, 10)
# pt_bins = np.arange(250, 3000, 50)
print(pt_bins)

# Add true b-jets histogram
plot_histo.add(
    Histogram(
        values=pt_trueb,
        bins=pt_bins,
        label="true $b$-jets",
        colour=Flavours["trueghostbjets"].colour,
        norm=True,
    ),
    # reference=True # Use true b-jets as the denominator for the ratio panel
)

# contaminated b-jets histogram
plot_histo.add(
    Histogram(
        values=pt_contaminatedb,
        bins=pt_bins,
        label="contaminated $b$-jets",
        colour=Flavours["contaminatedghostbjets"].colour,
        norm=True,
    ),
)

# c-jets
plot_histo.add(
    Histogram(
        values=pt_c,
        bins=pt_bins,
        label="$c$-jets",
        colour=Flavours["cjets"].colour,
        norm=True,
    ),
)

# light-jets
plot_histo.add(
    Histogram(
        values=pt_light,
        bins=pt_bins,
        label="light-jets",
        colour=Flavours["ujets"].colour,
        norm=True,
    ),
)

plot_histo.draw()

# Save the plot
output_path = "/home/xzcapfed/MSci/flavour_contamination/evaluation/distributions/TEST.png"
plot_histo.savefig(output_path, transparent=False)