from __future__ import annotations


from puma.utils.precision_recall_scores import precision_recall_scores_per_class

# ... [Keep your existing imports] ..
import numpy as np

from puma import Roc, RocPlot
from ftag.utils import calculate_rejection
from puma.utils import get_dummy_2_taggers, logger
import h5py
import pandas as pd
from puma.utils import get_good_colours
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader
import glob

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
        # ("eventNumber", "%10==", 9),
    ])

    # Define required variables
    jet_vars = [
        "pt", "eta", "eventNumber",
        "HadronGhostTruthLabelID",
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
    # df = df[df["HadronConeExclExtendedTruthLabelID"] == df["HadronGhostExtendedTruthLabelID"]]

    return df

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from puma.utils.precision_recall_scores import precision_recall_scores_per_class

def get_pr_scores(df, model_name):
    """Calculates and returns precision and recall arrays for a given model."""
    # 1. Get Probabilities (Summing light jets)
    p_light = df[f'{model_name}_pud'] + df[f'{model_name}_ps'] + df[f'{model_name}_pg']
    p_c     = df[f'{model_name}_pc']
    p_b     = df[f'{model_name}_pb']
    p_tau   = df[f'{model_name}_ptau']
    
    # 2. Get Prediction (Argmax)
    probs = np.column_stack((p_light, p_c, p_b, p_tau))
    predictions = np.argmax(probs, axis=1)

    # 3. Get Truth Labels (Mapped to 0-3)
    label_map = {0: 0, 4: 1, 5: 2, 15: 3}
    targets = df["HadronGhostTruthLabelID"].map(label_map)
    
    # Filter valid
    valid_mask = targets.notna()
    targets = targets[valid_mask].astype(int).values
    predictions = predictions[valid_mask]

    # 4. Calculate
    precision, recall = precision_recall_scores_per_class(targets, predictions)
    return precision, recall

fname='/home/xzcapfed/tmp/ckpts/epoch=003-val_loss=1.01886__test_zprime.h5'
logger.info("Loading h5 files")
model_name = "GN3V01_smaller"
df_gn3 = load_df(fname, model_name)

# --- 1. CALCULATE SCORES FOR BOTH MODELS ---
logger.info("Calculating Precision/Recall for Baseline...")
prec_base, rec_base = get_pr_scores(df_gn3, "GN3V01")

logger.info("Calculating Precision/Recall for Smaller Model...")
prec_small, rec_small = get_pr_scores(df_gn3, "GN3V01_smaller")

# --- 2. SETUP PLOTTING ---
classes = ["Light", "Charm", "Bottom", "Tau"]
x = np.arange(len(classes))
width = 0.35  # Width of the bars

# --- PLOT 1: PRECISION COMPARISON ---
fig_p, ax_p = plt.subplots(figsize=(8, 6))
rects1 = ax_p.bar(x - width/2, prec_base, width, label='GN3V01', color='#1f77b4')
rects2 = ax_p.bar(x + width/2, prec_small, width, label='GN3V01_smaller', color='#ff7f0e')

ax_p.set_ylabel('Precision')
ax_p.set_title("Precision Comparison $Z'$")
ax_p.set_xticks(x)
ax_p.set_xticklabels(classes)
ax_p.legend()
ax_p.grid(axis='y', linestyle='--', alpha=0.3)
ax_p.bar_label(rects1, padding=3, fmt='%.2f')
ax_p.bar_label(rects2, padding=3, fmt='%.2f')
ax_p.set_ylim(0, 1.15)

plt.tight_layout()
fig_p.savefig("plots/comparison_precision_zprime.png")
plt.close(fig_p)

# --- PLOT 2: RECALL COMPARISON ---
fig_r, ax_r = plt.subplots(figsize=(8, 6))
rects1 = ax_r.bar(x - width/2, rec_base, width, label='GN3V01', color='#1f77b4')
rects2 = ax_r.bar(x + width/2, rec_small, width, label='GN3V01_smaller', color='#ff7f0e')

ax_r.set_ylabel('Recall')
ax_r.set_title("Recall Comparison $Z'$")
ax_r.set_xticks(x)
ax_r.set_xticklabels(classes)
ax_r.legend()
ax_r.grid(axis='y', linestyle='--', alpha=0.3)
ax_r.bar_label(rects1, padding=3, fmt='%.2f')
ax_r.bar_label(rects2, padding=3, fmt='%.2f')
ax_r.set_ylim(0, 1.15)

plt.tight_layout()
fig_r.savefig("plots/comparison_recall_zprime.png")
plt.close(fig_r)