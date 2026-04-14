import numpy as np

from puma import Roc, RocPlot
from puma.metrics import calc_rej


def get_rej_for_hbb(jets_df, sig_eff = np.linspace(0.4, 1, 40), from_score = None):
    #is_hbb = jets_df["R10TruthLabel_R22v1"] == 11
    #is_hcc = jets_df["R10TruthLabel_R22v1"] == 12
    #is_top = np.isin(jets_df["R10TruthLabel_R22v1"], [1]) # add or conditions for 6,7
    #is_qcd = jets_df["R10TruthLabel_R22v1"] == 10

    is_hbb = jets_df["flavour_label"] == 0
    is_hcc = jets_df["flavour_label"] == 1
    is_top = jets_df["flavour_label"] == 2
    is_qcd = jets_df["flavour_label"] == 3

    # is_qddbb is true when jets_df["R10TruthLabel_R22v1"] == 10 AND jets_df["doubleB"] = True
    #is_qcdbb = is_qcd & (jets_df["doubleB"] == True)

    f_hcc = 0.02
    f_top = 0.25
    f_qcd = 1.0 - f_hcc - f_top

    if from_score:
        p_hbb = jets_df[from_score+"_phbb"]
        p_hcc = jets_df[from_score+"_phcc"]
        if from_score+"_pinclusive_top" in jets_df.columns:
            p_top = jets_df[from_score+"_pinclusive_top"]
        else: 
            p_top = jets_df[from_score+"_ptop"]
        p_qcd = jets_df[from_score+"_pqcd"]
    else:
        p_hbb = jets_df["p_phbb"]
        p_hcc = jets_df["p_phcc"]
        if "p_pinclusive_top" in jets_df.columns:
            p_top = jets_df["p_pinclusive_top"]
        else:
            p_top = jets_df["p_ptop"]
        p_qcd = jets_df["p_pqcd"]


    disc = np.log(p_hbb /  (f_hcc * p_hcc + f_top * p_top + f_qcd * p_qcd))
    disc_hcc_rej = calc_rej(disc[is_hbb], disc[is_hcc], sig_eff)
    disc_qcd_rej = calc_rej(disc[is_hbb], disc[is_qcd], sig_eff)
    disc_top_rej = calc_rej(disc[is_hbb], disc[is_top], sig_eff)
    #disc_qcdbb_rej = calc_rej(disc[is_hbb], disc[is_qcdbb], sig_eff)

    return disc_hcc_rej, disc_qcd_rej, disc_top_rej #, disc_qcdbb_rej


if __name__ == "__main__":
    
    
    models = {
        #"Baseline GN2X" : "/share/lustre/avaitkus/salt/salt/logs/GN2XAux-fullData-50Epoch-flow_20231025-T100442/ckpts/epoch=037-val_loss=0.48245__test_pp_output_test.h5",
        "GN2XAux-flow" : "/share/lustre/avaitkus/salt/salt/logs/GN2XAux-fullData-50Epoch-flow_20231025-T100442/ckpts/epoch=037-val_loss=0.48245__test_pp_output_test.h5",        
    }