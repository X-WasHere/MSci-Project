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

fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_ttbar.h5'
fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_ttbar.h5'
model_name = "GN3_calo"
benchmark_name = "GN3_calo_benchmark"

logger.info("Loading h5 files")
df = load_df(fname, model_name)
df_benchmark = load_df(fname_benchmark, benchmark_name)
logger.info(f"plotting with {len(df)} jets")

logger.info("Caclulate tagger discriminants")
discs_benchmark = calc_discriminant(df_benchmark, benchmark_name, fc=0.2, ftau=0.1, split_simple=True) # benchmark GN3 model
discs_calo = calc_discriminant(df, model_name, fc=0.2, ftau=0.1, split_simple=True) # new GN3 model

# defining boolean arrays to select the different flavour classes
is_light, is_c, is_b, is_ud, is_s, is_g = get_details(df, "HadronGhostTruthLabelID", "PartonTruthLabelID")


taggers = ["GN3V01_calo", "GN3V01_benchmark"]
discs = {"GN3V01_calo": discs_calo, "GN3V01_benchmark": discs_benchmark}
linestyles = get_good_linestyles()[:2]

bjet_discs = discs_calo[is_b]
working_point = 0.70
wp_cut = np.percentile(bjet_discs, (1-working_point) * 100) 

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=1,
    ylabel="Normalised number of jets",
    ylabel_ratio=["Ratio to GN3 benchmark"],
    xlabel="$b$-jet discriminant",
    logy=True,
    leg_ncol=1,
    figsize=(5.5, 4.5),
    y_scale=1.5,
    ymax_ratio=[1.5],
    ymin_ratio=[0.5],
    atlas_second_tag="$\\sqrt{s}=13$ TeV $t\overline{t}$ events \n20 GeV < $p_T$ < 250 GeV \n$f_{c}=f_{b}=0.2$, $f_{\\tau}=0.1$",
)

# Add the histograms
for tagger, linestyle in zip(taggers, linestyles):
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_ud],
            bins=np.linspace(-10, 10, 50),
            label="$ud$-jets" if tagger == "GN3V01_calo" else None,
            colour=Flavours["ghostudjets"].colour,
            ratio_group="ghostudjets",
            linestyle=linestyle,
        ),
        reference=tagger == "GN3V01_benchmark",
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_s],
            bins=np.linspace(-10, 10, 50),
            label="$s$-jets" if tagger == "GN3V01_calo" else None,
            colour=Flavours["strangejets"].colour,
            ratio_group="strangejets",
            linestyle=linestyle,
        ),
        reference=tagger == "GN3V01_benchmark",
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_g],
            bins=np.linspace(-10, 10, 50),
            label="$g$-jets" if tagger == "GN3V01_calo" else None,
            colour=Flavours["gluonjets"].colour,
            ratio_group="gluonjets",
            linestyle=linestyle,
        ),
        reference=tagger == "GN3V01_benchmark",
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_b],
            bins=np.linspace(-10, 10, 50),
            label="$b$-jets" if tagger == "GN3V01_calo" else None,
            colour=Flavours["bjets"].colour,
            ratio_group="bjets",
            linestyle=linestyle,
        ),
        reference=tagger == "GN3V01_benchmark",
    )

plot_histo.draw_vlines(
    xs=[wp_cut],
    labels=[f"{working_point:.0%} WP"], 
    linestyle="dashed",              
    colour="black"                    
)

plot_histo.draw()
plot_histo.make_linestyle_legend(
    linestyles=linestyles, labels=["GN3 calo", "GN3 benchmark"], bbox_to_anchor=(0.55, 1)
)
plot_histo.savefig("/home/xzcapfed/MSci/GN3_calo/plots/discriminants/ud-s-g-ttbar.png", transparent=False)