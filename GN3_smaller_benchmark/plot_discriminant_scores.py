"""Produce histogram of discriminant from tagger output and labels."""

from __future__ import annotations

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import h5py
import glob

from ftag import Flavours
from ftag.utils import get_discriminant
from ftag.cuts import Cuts

from puma import Histogram, HistogramPlot
from puma.utils import get_good_linestyles, logger, get_dummy_2_taggers
from ftag.hdf5.h5reader import H5Reader

def get_details(df, label1, label2):
    '''Defines a boolean array to select the different flavour classes'''
    is_light = df[label1] == 0
    is_c = df[label1] == 4
    is_b = df[label1] == 5

    is_ud = (df[label1] ==0) & (df[label2] <= 2)
    is_s = (df[label1] == 0) & (df[label2] == 3)
    is_g = (df[label1] ==0) & (df[label2] == 21)

    return is_light, is_c, is_b, is_ud, is_s, is_g

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
        ("pt", ">", 20000),
        ("pt", "<", 250000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
    ])

    # Define required variables
    jet_vars = [
        "pt", "eta", "eventNumber",
        "PartonTruthLabelID",
        "HadronGhostTruthLabelID",
        "flavour_label",
        "GN3V01_pb", "GN3V01_pc", "GN3V01_ptau", "GN3V01_pud", "GN3V01_ps", "GN3V01_pg", "GN3V01_pu",
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

    df = data["jets"] # we are NOT using a pandas df

    return df

def calc_discriminant(df, name, fc, ftau, gn2=False, split_simple=False, sjet_disc=False):
    if gn2:
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * df[f'{name}_pu']) + 1e-10
        disc = nom/denom
    elif split_simple:
        uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pg'] # light flavour jets
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
        disc = nom/denom  
    elif sjet_disc:
        fb = fc
        uscore = df[f'{name}_pud'] + df[f'{name}_pg']
        nom = df[f'{name}_ps'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + (fb * df[f'{name}_pb']) + ((1-fc-fb-ftau) * uscore) + 1e-10
        disc = nom/denom  

    return np.log(disc)

fname='/home/xzcapfed/tmp/ckpts/epoch=003-val_loss=1.01886__test_ttbar.h5'
logger.info("Loading h5 files")
model_name = "GN3V01_smaller"
df = load_df(fname, model_name)

logger.info("Caclulate tagger discriminants")

disc_gn3_explicit = calc_discriminant(df, "GN3V01", fc=0.018, ftau=0.1, split_simple=True) # benchmark GN3 model
disc_gn3_explicit_sjet = calc_discriminant(df, "GN3V01", fc=0.018, ftau=0.1, sjet_disc=True) 
disc_gn3_smaller_explicit = calc_discriminant(df, "GN3V01_smaller", fc=0.018, ftau=0.1, split_simple=True) # GN3 smaller model we trained
disc_gn3_smaller_explicit_sjet = calc_discriminant(df, "GN3V01_smaller", fc=0.018, ftau=0.1, sjet_disc=True)


# Create GN3V01_smaller_pu label for get_discriminant function
GN3V01_smaller_pu = df['GN3V01_smaller_ps'] + df['GN3V01_smaller_pud'] + df['GN3V01_smaller_pg']
# create NEW df with extra column
df_new = rfn.append_fields(
    base=df,
    names='GN3V01_smaller_pu',
    data=GN3V01_smaller_pu,
    dtypes=np.dtype(df['GN3V01_smaller_ps'][0]),
    usemask=False,
    asrecarray=True
)

# Calculate discriminant scores using puma library and add them to the dataframe
# disc_gn3_smaller = get_discriminant(
#     jets=df_new,
#     tagger="GN3V01_smaller",
#     signal=Flavours["bjets"],
#     flavours=Flavours.by_category("single-btag"),
#     fraction_values={
#         "fc": 0.018,
#         "fu": 0.982,
#         "ftau": 0.1,
#     },
# )
# disc_gn3 = get_discriminant(
#     jets=df_new,
#     tagger="GN3V01",
#     signal=Flavours["bjets"],
#     flavours=Flavours.by_category("single-btag"),
#     fraction_values={
#         "fc": 0.018,
#         "fu": 0.982,
#         "ftau": 0.1,
#     },
# )

# defining boolean arrays to select the different flavour classes
is_light, is_c, is_b, is_ud, is_s, is_g = get_details(df, "HadronGhostTruthLabelID", "PartonTruthLabelID")


taggers = ["GN3V01_smaller", "GN3V01"]
discs = {"GN3V01_smaller": disc_gn3_smaller_explicit_sjet, "GN3V01": disc_gn3_explicit_sjet}
linestyles = get_good_linestyles()[:2]

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=1,
    ylabel="Normalised number of jets",
    ylabel_ratio=["Ratio to GN3V01"],
    xlabel="$s$-jet discriminant",
    logy=False,
    leg_ncol=1,
    figsize=(5.5, 4.5),
    y_scale=1.5,
    ymax_ratio=[1.5],
    ymin_ratio=[0.5],
    atlas_second_tag="$\\sqrt{s}=13$ TeV $t\overline{t}$ events \n20 GeV < $p_T$ < 250 GeV \n$f_{c}=f_{b}=0.018$, $f_{\\tau}=0.1$",
)

# Add the histograms
for tagger, linestyle in zip(taggers, linestyles):
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_ud],
            bins=np.linspace(-10, 10, 50),
            label="$ud$-jets" if tagger == "GN3V01" else None,
            colour=Flavours["lquarkjets"].colour,
            ratio_group="lquarkjets",
            linestyle=linestyle,
        ),
        reference=tagger == "GN3V01",
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_s],
            bins=np.linspace(-10, 10, 50),
            label="$s$-jets" if tagger == "GN3V01" else None,
            colour=Flavours["strangejets"].colour,
            ratio_group="strangejets",
            linestyle=linestyle,
        ),
        reference=tagger == "GN3V01",
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_g],
            bins=np.linspace(-10, 10, 50),
            label="$gluon$-jets" if tagger == "GN3V01" else None,
            colour=Flavours["gluonjets"].colour,
            ratio_group="gluonjets",
            linestyle=linestyle,
        ),
        reference=tagger == "GN3V01",
    )

plot_histo.draw()
plot_histo.make_linestyle_legend(
    linestyles=linestyles, labels=["GN3V01", "GN3_lowstat"], bbox_to_anchor=(0.55, 1)
)
plot_histo.savefig("plots/discriminants/Ds_ttbar.png", transparent=False)