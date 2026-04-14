"""Produce roc curves from tagger output and labels."""

from __future__ import annotations

import numpy as np

from puma import Roc, RocPlot
from ftag.utils import calculate_rejection
from puma.utils import logger
import h5py
import pandas as pd
from puma.utils import get_good_colours
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader
from ftag import Flavours
import glob

def get_details(df, label):
    '''Defines a boolean array to select the different flavour classes'''
    is_light = df[label] == 0
    is_c = df[label] == 4
    is_b = df[label] == 5
    is_tau = df[label] == 15
    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)
    n_jets_tau = sum(is_tau)

    return is_light, is_c, is_b, is_tau, n_jets_light, n_jets_c, n_jets_tau


def calc_discriminant(df, name, fc, ftau, gn2=False, split_simple=False, split=False):
    if gn2:
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * df[f'{name}_pu']) + 1e-10
        disc = nom/denom
    elif split_simple:
        uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pg'] # prob of light jet is the sum of these components
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
        disc = nom/denom  
    elif split:
        uscore = df[f'{name}_pghostsjets'] + df[f'{name}_pghostudjets'] + df[f'{name}_pghostgjets']
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
        disc = nom/denom  
    return np.log(disc)


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

    # Define cuts as before
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
        "flavour_label",
        "GN3V01_smaller_official_pb", "GN3V01_smaller_official_pc", "GN3V01_smaller_official_ptau", 
        "GN3V01_smaller_official_pghostudjets", "GN3V01_smaller_official_pghostsjets", "GN3V01_smaller_official_pghostgjets",
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
    # df = df[df["HadronConeExclExtendedTruthLabelID"] == df["HadronGhostExtendedTruthLabelID"]]

    return df

def describe_df(df, column=None):
    """Prints a summary of the given dataframe to the terminal
    
    Parameters:
    df : dataframe table (pd.dataframe)
    column : column to produce statsitics for (str)
    """
    pd.set_option('display.max_columns', None) # show all columns

    print("---SHAPE INFO----")
    print(df.info())
    print("---SEE FIRST FEW ROWS ---")
    print(df.head(5))

    if column:
        print("---LOOK at desired COLUMN---")
        print(df[column].describe())

    return

fname='/home/xzcapfed/tmp/ckpts/epoch=003-val_loss=1.01886__test_zprime_with_wei_scores.h5'
logger.info("Loading h5 files")
model_name = "GN3V01_smaller"
df_gn3 = load_df(fname, model_name)
describe_df(df_gn3)
logger.info(f"plotting with {len(df_gn3)} jets")

logger.info("caclulate tagger discriminants")
discs_baseline = calc_discriminant(df_gn3, "GN3V01_smaller_official", fc=0.2, ftau=0.1, split=True) # benchmark GN3 model
discs_smaller_gn3 = calc_discriminant(df_gn3, "GN3V01_smaller", fc=0.2, ftau=0.1, split_simple=True) # GN3 smaller model we trained

# defining target efficiency
sig_eff = np.linspace(0.6, 1, 100)

# defining boolean arrays to select the different flavour classes
is_light_gn3, is_c_gn3, is_b_gn3, is_tau_gn3, n_jets_light_gn3, n_jets_c_gn3, n_jets_tau_gn3 = get_details(df_gn3, "HadronGhostTruthLabelID")


logger.info("Calculate rejection")
cjets_rej_baseline = calculate_rejection(discs_baseline[is_b_gn3].values, discs_baseline[is_c_gn3].values, sig_eff)
taujets_rej_baseline = calculate_rejection(discs_baseline[is_b_gn3].values, discs_baseline[is_tau_gn3].values, sig_eff)
light_jets_rej_baseline = calculate_rejection(discs_baseline[is_b_gn3].values, discs_baseline[is_light_gn3].values, sig_eff)

# rejection for smaller GN3 trained model
cjets_rej_smaller_gn3 = calculate_rejection(discs_smaller_gn3[is_b_gn3].values, discs_smaller_gn3[is_c_gn3].values, sig_eff)
taujets_rej_smaller_gn3 = calculate_rejection(discs_smaller_gn3[is_b_gn3].values, discs_smaller_gn3[is_tau_gn3].values, sig_eff)
light_jets_rej_smaller_gn3 = calculate_rejection(discs_smaller_gn3[is_b_gn3].values, discs_smaller_gn3[is_light_gn3].values, sig_eff)

# here the plotting of the roc starts
logger.info("Plotting ROC curves.")
plot_roc = RocPlot(
    n_ratio_panels=3,
    ylabel="Background Rejection",
    xlabel="$b$-jet Efficiency",
    atlas_second_tag="$\sqrt{s} = 13.6$ TeV, $Z'$ events\n250 GeV < $p_T$ < 6 TeV, $|\eta| < 2.5$",
    figsize=(5.5, 6),
    y_scale=1.4,
)

# Now manually assign label colours if needed
plot_roc.label_colours = {
    "GN3V01 benchmark": get_good_colours()[0],
    "GN3V01 retrain": get_good_colours()[2]
}

# plot light jet rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        light_jets_rej_baseline,
        n_test=n_jets_light_gn3,
        rej_class="ujets",
        signal_class="bjets",
        label="GN3V01 benchmark"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        light_jets_rej_smaller_gn3,
        n_test=n_jets_light_gn3,
        rej_class="ujets",
        signal_class="bjets",
        label="GN3V01 retrain"
    ),
)


# plot c jet rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        cjets_rej_baseline,
        n_test=n_jets_c_gn3,
        rej_class="cjets",
        signal_class="bjets",
        label="GN3V01 benchmark"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        cjets_rej_smaller_gn3,
        n_test=n_jets_c_gn3,
        rej_class="cjets",
        signal_class="bjets",
        label="GN3V01 retrain"
    ),
)


# plot tau rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        taujets_rej_baseline,
        n_test=n_jets_tau_gn3,
        rej_class="taujets",
        signal_class="bjets",
        label="GN3V01 benchmark"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        taujets_rej_smaller_gn3,
        n_test=n_jets_tau_gn3,
        rej_class="taujets",
        signal_class="bjets",
        label="GN3V01 retrain"
    ),
)


# setting which flavour rejection ratio is drawn in which ratio panel
plot_roc.set_ratio_class(1, Flavours["cjets"])
plot_roc.set_ratio_class(2, Flavours["ujets"])
plot_roc.set_ratio_class(3, Flavours["taujets"])


plot_roc.draw()
plot_roc.savefig("/home/xzcapfed/MSci/GN3_smaller_benchmark/plots_official/roc_zprime.png", transparent=False)



