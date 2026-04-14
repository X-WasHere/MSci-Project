"""Produce a range of plots from tagger output and labels."""

from __future__ import annotations

from puma.hlplots import Results, Tagger
from puma.utils import logger
from ftag.cuts import Cuts

import h5py


# The line below generates dummy data which is similar to a NN output
fname = "/home/xzcapfed/tmp/ckpts/epoch=003-val_loss=1.01886__test_zprime_general.h5"
file = h5py.File(fname,'r')

# define jet selections
cuts = Cuts.from_list([
    ("pt", ">", 250000),
    ("pt", "<", 6000000),
    ("eta", "<", 2.5),
    ("eta", ">", -2.5),
    ])

# define the taggers
gn3_smaller = Tagger(
    name="GN3V01_smaller",
    output_flavours=["cjets", "bjets"],
    label="GN3 lowstat ($f_{c}=0.07$)",
    fxs={"fc": 0.07},
    colour="#AA3377",
)
gn3_official = Tagger(
    name="GN3V01",
    output_flavours=["cjets", "bjets"],
    label="GN3V01 official ($f_{c}=0.07$)",
    fxs={"fc": 0.07},
    colour="#4477AA",
    reference=True,
)

# create the Results object
# for c-tagging use signal="cjets"
# for Xbb/cc-tagging use signal="hbb"/"hcc"
results = Results(signal="bjets", sample="zprime")

# load taggers from the file object
logger.info("Loading taggers.")
results.load_taggers_from_file(
    [gn3_smaller, gn3_official],
    file.filename,
    cuts=cuts,
    num_jets=len(file["jets"]),
)

results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, \nZ' events, $250$ GeV $< p_{T} <6$ TeV"
)


# ---PLOTTING---

# ROC curves
logger.info("Plotting ROC curves.")
results.plot_rocs()

exit()

# tagger probability distribution plots
results.plot_probs(logy=True, bins=40)

# eff/rej vs. variable plots
logger.info("Plotting efficiency/rejection vs pT curves.")
results.plot_var_perf(
    working_point=0.3,
    bins=[250, 260, 270, 300, 350, 400, 600, 700, 900, 1000, 1500, 2000, 3000, 4000, 5000, 6000],
    flat_per_bin=False,
)


