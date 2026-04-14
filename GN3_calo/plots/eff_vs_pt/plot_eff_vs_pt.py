"""Produce a range of plots from tagger output and labels."""

from __future__ import annotations

from puma.hlplots import Results, Tagger
from puma.utils import logger
from ftag.cuts import Cuts

import h5py
import numpy as np

# fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_ttbar.h5'
# fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_ttbar.h5'
fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_zprime.h5'
fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_zprime.h5'

file_calo = h5py.File(fname,'r')
file_benchmark = h5py.File(fname_benchmark,'r')

# define jet selections
cuts = Cuts.from_list([
    # ("pt", ">", 20000),
    # ("pt", "<", 250000),
    ("pt", ">", 250000),
    ("pt", "<", 6000000),
    ("eta", "<", 2.5),
    ("eta", ">", -2.5),
    ])

# define the taggers
gn3_calo = Tagger(
    name="GN3_calo",
    output_flavours=["cjets", "bjets"],
    label="GN3 calo",
    # fxs={"fc": 0.07},
    fxs={"fc": 0.2},
    colour="#AA3377",
)
gn3_benchmark = Tagger(
    name="GN3_calo_benchmark",
    output_flavours=["cjets", "bjets"],
    label="GN3 benchmark",
    # fxs={"fc": 0.07},
    fxs={"fc": 0.2},
    colour="#4477AA",
    reference=True,
)

# create the Results object
# for c-tagging use signal="cjets"
# for Xbb/cc-tagging use signal="hbb"/"hcc"
results = Results(signal="bjets", sample="zprime_smallerbins")

# load taggers from the file object
logger.info("Loading calo tagger.")
results.load_taggers_from_file(
    [gn3_calo],
    file_calo.filename,
    cuts=cuts,
    num_jets=len(file_calo["jets"]),
)
logger.info("Loading benchmark tagger.")
results.load_taggers_from_file(
    [gn3_benchmark],
    file_benchmark.filename,
    cuts=cuts,
    num_jets=len(file_benchmark["jets"]),
)

if 'ttbar' in fname:
    results.atlas_second_tag = (
        "$\\sqrt{s} = 13.6$ TeV, $t\\overline{t}$ events \n20 GeV < $p_T$ < 250 GeV"
    )    
    bins = np.linspace(20, 250, 20)
    wp = 0.70
else:
    results.atlas_second_tag = (
        "$\\sqrt{s}=13$ TeV, Z' events \n250 GeV $< p_{T} <6$ TeV"
    )
    bins = [250, 260, 270, 300, 350, 400, 600, 700, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    wp = 0.3


# tagger probability distribution plots
results.plot_probs(logy=True, 
                   bins=40,
                   figsize=(5.5,4.5))

# eff/rej vs. variable plots
logger.info("Plotting efficiency/rejection vs pT curves.")
results.plot_var_perf(
    working_point=wp,
    bins=bins,
    flat_per_bin=False,
    figsize=(5.5,4.5)
)


