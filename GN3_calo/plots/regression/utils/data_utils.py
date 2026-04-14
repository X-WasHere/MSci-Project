import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from puma.utils import logger


def create_output_folders(parent_folder_name, base_dir="/share/rcifdata/avaitkus/saltPlotTools/plots/"): # FIXME: hardcoded!
    parent_folder_name = base_dir+parent_folder_name
    if parent_folder_name[-1] != "/":
        parent_folder_name += "/"
    if base_dir[-1] != "/":
        base_dir += "/"
    def create_folder(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    create_folder(parent_folder_name)
    create_folder(parent_folder_name+"roc_plots")
    create_folder(parent_folder_name+"regression_tasks")
    create_folder(parent_folder_name+"regression_tasks/variable_hists")
    create_folder(parent_folder_name+"regression_tasks/regression_output")
    create_folder(parent_folder_name+"regression_tasks/residuals")
    create_folder(parent_folder_name+"regression_tasks/response_plots")
    return parent_folder_name


def create_folder(folder_name, base_dir="/share/rcifdata/avaitkus/saltPlotTools/plots/"):
    full_folder_name = base_dir+folder_name
    if full_folder_name[-1] != "/":
        full_folder_name += "/"
    if not os.path.exists(full_folder_name):
        os.makedirs(full_folder_name)
    return full_folder_name


def read_jets_from_test_file(
    filepath, 
    max_jets = None, 
    add_muon_column = False,
):
    '''
    Reads jets from a test file and returns a pandas dataframe.
    '''
    with h5py.File(filepath, "r") as f:
        if max_jets:
            jets_df = pd.DataFrame(f["jets"][:max_jets])
        else:
            jets_df = pd.DataFrame(f["jets"][:])

        for key in f.keys():
            if key == "jets":
                continue
            constituent = f[key]["valid"]
            jets_df[f"nConstituens_{key}"] = np.sum(constituent, axis=1)

        if add_muon_column:
            charged_muon_quality_FatJet = f["charged"]["muon_quality_FatJet"]
            tracks_loose_muon_quality_FatJet = f["tracks_loose"]["muon_quality_FatJet"]
            print(f"muon_quality_FatJet shape: {charged_muon_quality_FatJet.shape}")
            for i in tqdm(range(len(jets_df))):
                # Selecting the highest pT muon
                muon_idx_in_charged =  np.argwhere( \
                    (charged_muon_quality_FatJet[i,:] == 1) | \
                    (charged_muon_quality_FatJet[i,:] == 2)).flatten()
                muon_idx_in_charged = muon_idx_in_charged[0] if len(muon_idx_in_charged)>0 else None
                muon_idx_in_tracks_loose =  np.argwhere( \
                    (tracks_loose_muon_quality_FatJet[i, :] == 1) | \
                    (tracks_loose_muon_quality_FatJet[i, :] == 2)).flatten()
                muon_idx_in_tracks_loose = muon_idx_in_tracks_loose[0] if len(muon_idx_in_tracks_loose)>0 else None
                muon_in_jet = True if (muon_idx_in_charged or muon_idx_in_tracks_loose) else False
                jets_df.at[i, "muonInJet"] = muon_in_jet
                if muon_in_jet:
                    muon_in_charged = False if (muon_idx_in_charged is None) else True
                    if muon_in_charged and (not muon_idx_in_tracks_loose is None):
                        muon_in_charged = muon_idx_in_charged < muon_idx_in_tracks_loose
                    muon_idx = muon_idx_in_charged if muon_in_charged else muon_idx_in_tracks_loose
                    # take highest pT muon (if its idx)
                    # extract pT, eta, phi, M
                    # create muon correction 4 vector
                    from pylorentz import Momentum4
                    selected_container = "charged" if muon_in_charged else "tracks_loose"
                    muon_correction_4vec = Momentum4.m_eta_phi_pt(
                        f[selected_container]["muon_muonCorrM"][i, muon_idx],
                        f[selected_container]["muon_muonCorrEta"][i, muon_idx],
                        f[selected_container]["muon_muonCorrPhi"][i, muon_idx],
                        f[selected_container]["muon_muonCorrPt"][i, muon_idx]
                    )

                    # create jet 4 vector (do it outside the loop or smth)
                        # 1 for reco, 1 for pred, 1 for truth
                    jet_reco_4vec = Momentum4.m_eta_phi_pt(
                        jets_df.at[i, "mass"],
                        jets_df.at[i, "eta"],
                        jets_df.at[i, "phi"],
                        jets_df.at[i, "pt"],
                    )

                    jet_pred_4vec = Momentum4.m_eta_phi_pt(
                        jets_df.at[i, "jet_mass_regression_R10TruthLabel_R22v1_TruthGroomedJetMass"],
                        jets_df.at[i, "eta"],
                        jets_df.at[i, "phi"],
                        jets_df.at[i, "jet_pt_regression_R10TruthLabel_R22v1_TruthGroomedJetPt"],
                    )

                    jet_truth_4vec = Momentum4.m_eta_phi_pt(
                        jets_df.at[i, "R10TruthLabel_R22v1_TruthGroomedJetMass"],
                        jets_df.at[i, "eta"],
                        jets_df.at[i, "phi"],
                        jets_df.at[i, "R10TruthLabel_R22v1_TruthGroomedJetPt"],
                    )

                    # jet 4 vector += muon 4 vector
                    corr_jet_reco_4vec = jet_reco_4vec + muon_correction_4vec
                    corr_jet_pred_4vec = jet_pred_4vec + muon_correction_4vec
                    corr_jet_truth_4vec = jet_truth_4vec + muon_correction_4vec

                    # extract new mass, pT
                    corr_mass_reco = corr_jet_reco_4vec.m
                    corr_pt_reco = corr_jet_reco_4vec.p_t

                    corr_mass_pred = corr_jet_pred_4vec.m
                    corr_pt_pred = corr_jet_pred_4vec.p_t

                    corr_mass_truth = corr_jet_truth_4vec.m
                    corr_pt_truth = corr_jet_truth_4vec.p_t

                    #print(f"Reco pt, was: {jets_df.at[i, 'pt']/1e3:.1f}, now: {corr_pt_reco/1e3:.1f}")
                    #print(f"Reco mass, was: {jets_df.at[i, 'mass']/1e3:.1f}, now: {corr_mass_reco/1e3:.1f}")
                    #print(f"Pred mass, was: {jets_df.at[i, 'jet_mass_regression_R10TruthLabel_R22v1_TruthGroomedJetMass']}, now: {corr_mass_pred}")
                    #print(f"Pred pt, was: {jets_df.at[i, 'jet_pt_regression_R10TruthLabel_R22v1_TruthGroomedJetPt']}, now: {corr_pt_pred}")
                    #print(f"Truth mass, was: {jets_df.at[i, 'R10TruthLabel_R22v1_TruthGroomedJetMass']}, now: {corr_mass_truth}")
                    #print(f"Truth pt, was: {jets_df.at[i, 'R10TruthLabel_R22v1_TruthGroomedJetPt']}, now: {corr_pt_truth}")
                else:
                    corr_mass_reco = jets_df.at[i, "mass"]
                    corr_pt_reco = jets_df.at[i, "pt"]
                    corr_mass_pred = jets_df.at[i, "jet_mass_regression_R10TruthLabel_R22v1_TruthGroomedJetMass"]
                    corr_pt_pred = jets_df.at[i, "jet_pt_regression_R10TruthLabel_R22v1_TruthGroomedJetPt"]
                    corr_mass_truth = jets_df.at[i, "R10TruthLabel_R22v1_TruthGroomedJetMass"]
                    corr_pt_truth = jets_df.at[i, "R10TruthLabel_R22v1_TruthGroomedJetPt"]

                jets_df.at[i, "muonCorr_mass"] = corr_mass_reco / 1e3
                jets_df.at[i, "muonCorr_pt"] = corr_pt_reco / 1e3
                jets_df.at[i, "muonCorr_p_truthMass"] = corr_mass_pred / 1e3
                jets_df.at[i, "muonCorr_p_truthPt"] = corr_pt_pred / 1e3
                jets_df.at[i, "muonCorr_R10TruthLabel_R22v1_TruthGroomedJetMass"] = corr_mass_truth / 1e3
                jets_df.at[i, "muonCorr_R10TruthLabel_R22v1_TruthGroomedJetPt"] = corr_pt_truth / 1e3

    return jets_df


def process_jets_data(jets_df, mask=None, print_stats=False):

    # Converting mass, pt from MeV to GeV
    jets_df["mass"] /= 1e3
    jets_df["pt"] /= 1e3
    if "R10TruthLabel_R22v1_TruthJetMass" in jets_df.columns:
        jets_df["R10TruthLabel_R22v1_TruthJetMass"] /= 1e3
    if "R10TruthLabel_R22v1_TruthJetPt" in jets_df.columns:
        jets_df["R10TruthLabel_R22v1_TruthJetPt"] /= 1e3
    if "R10TruthLabel_R22v1_TruthGroomedJetMass" in jets_df.columns:
        jets_df["R10TruthLabel_R22v1_TruthGroomedJetMass"] /= 1e3
    if "R10TruthLabel_R22v1_TruthGroomedJetPt" in jets_df.columns:
        jets_df["R10TruthLabel_R22v1_TruthGroomedJetPt"] /= 1e3
    if "p_truthMass" in jets_df.columns:
        jets_df["p_truthMass"] /= 1e3
    if "p_truthPt" in jets_df.columns:
        jets_df["p_truthPt"] /= 1e3

    if mask is not None:
        jets_df = jets_df[mask]

    if print_stats:
        print(f"Total: {len(jets_df):,} test jets")
        print("reco:")
        print(f'\tmass: \t[{np.amin(jets_df["mass"]):.2f}, {np.amax(jets_df["mass"]):.2f}]')
        print(f'\tpt: \t[{np.amin(jets_df["pt"]):.2f}, {np.amax(jets_df["pt"]):.2f}]')
        print("truth (ungroomed):")
        print(f'\tmass: \t[{np.amin(jets_df["R10TruthLabel_R22v1_TruthJetMass"]):.2f}, {np.amax(jets_df["R10TruthLabel_R22v1_TruthJetMass"]):.2f}]')
        print(f'\tpt: \t[{np.amin(jets_df["R10TruthLabel_R22v1_TruthJetPt"]):.2f}, {np.amax(jets_df["R10TruthLabel_R22v1_TruthJetPt"]):.2f}]')
        print("truth (groomed):")
        print(f'\tmass: \t[{np.amin(jets_df["R10TruthLabel_R22v1_TruthGroomedJetMass"]):.2f}, {np.amax(jets_df["R10TruthLabel_R22v1_TruthGroomedJetMass"]):.2f}]')
        print(f'\tpt: \t[{np.amin(jets_df["R10TruthLabel_R22v1_TruthGroomedJetPt"]):.2f}, {np.amax(jets_df["R10TruthLabel_R22v1_TruthGroomedJetPt"]):.2f}]')
        print("pred:")
        print(f'\tmass: \t[{np.amin(jets_df["p_truthMass"]):.2f}, {np.amax(jets_df["p_truthMass"]):.2f}]')
        print(f'\tpt: \t[{np.amin(jets_df["p_truthPt"]):.2f}, {np.amax(jets_df["p_truthPt"]):.2f}]')


        lower_bound = 115
        upper_bound = 135
        reco_slice = jets_df[(jets_df["mass"] >= lower_bound) & (jets_df["mass"] <= upper_bound)]
        reco_purity = len(reco_slice[(reco_slice["R10TruthLabel_R22v1_TruthJetMass"] >= lower_bound) & (reco_slice["R10TruthLabel_R22v1_TruthJetMass"] <= upper_bound)])/len(reco_slice)
        print(f"reco purity ({lower_bound}-{upper_bound} GeV): {100*reco_purity:.2f}%")
        pred_slice = jets_df[(jets_df["p_truthMass"] >= lower_bound) & (jets_df["p_truthMass"] <= upper_bound)]
        pred_purity = len(pred_slice[(pred_slice["R10TruthLabel_R22v1_TruthJetMass"] >= lower_bound) & (pred_slice["R10TruthLabel_R22v1_TruthJetMass"] <= upper_bound)])/len(pred_slice)
        print(f"pred purity ({lower_bound}-{upper_bound} GeV): {100*pred_purity:.2f}%")

        improvement = (pred_purity - reco_purity)/reco_purity
        print(f"improvement: {100*improvement:.2f}%")
        print()

    return jets_df


def calculate_discriminant_hbb(jets_df, model_name, f_hcc=0.02, f_top=0.25):
    f_qcd = 1.0 - f_hcc - f_top
    p_hbb = jets_df[f"{model_name}_phbb"]
    p_hcc = jets_df[f"{model_name}_phcc"]
    p_top = jets_df[f"{model_name}_ptop"]
    p_qcd = jets_df[f"{model_name}_pqcd"]
    disc = np.log(p_hbb /  (f_hcc * p_hcc + f_top * p_top + f_qcd * p_qcd))
    jets_df[f"D_{model_name}_hbb"] = disc
    return jets_df


def find_WP_cut(jets_df, model_name, wp, target="hbb"):
    '''
    Finds the discriminant cut value for a given working point.
    '''
    disc_column_name = f"D_{model_name}_{target}"
    jets_df = jets_df.sort_values(by=disc_column_name, ascending=False)
    cut_value = jets_df.iloc[int(len(jets_df)*wp)][disc_column_name]
    return cut_value


def find_tag_from_filepath(filepath):
    '''
    Finds tag from filepath.
    Works only with standard log folders of Salt
    '''
    if not "logs" in filepath:
        raise ValueError("Filepath does not contain 'logs' folder.")
    tag = filepath.split("logs")[-1].split("/")[1] # choose the stadard log folder name
    tag = tag[:-17] # remove datetime from the tag
    return tag


def find_tag_from_phbb(jets_df, saved_scores = ["GN2Xv00", "GN2XWithMassv00"]):
    phbb_columns = [col for col in jets_df.columns if col.endswith("_phbb")]
    phbb_columns = [col for col in phbb_columns if not any(score+"_phbb" in col for score in saved_scores)]
    if len(phbb_columns) == 0:
        raise ValueError("No column ending with _phbb found in the dataframe.")
    elif len(phbb_columns) > 1:
        raise ValueError("Multiple columns ending with _phbb found in the dataframe.")
    else:
        tag = phbb_columns[0][:-5]
    return tag


def standardise_jet_classification_labels(jets_df, tag):

    # rename all colums that have tag to standard names
    for col in jets_df.columns:
        if col.startswith(tag):  
            new_col = "p"+col[len(tag):]
            #print(f"renaming {col} to {new_col}")
            jets_df.rename(columns={col:new_col}, inplace=True)

    return jets_df


def standardise_jet_regression_labels(jets_df, model_name=None):
    # TODO: make more general, not only for mass and pT

    # if old file format, convert predicted ratio to truth values
    if "salt_pTruthJetMass_mean" in jets_df.columns:
        logger.warning('old standard column "salt_pTruthJetMass_mean" found, converting to truth value')
        jets_df["p_truthMass"] = jets_df["mass"]*jets_df["salt_pTruthJetMass_mean"] 
        jets_df.drop(columns=["salt_pTruthJetMass_mean"], inplace=True) # drop old column
    
    if "salt_pTruthJetPt_mean" in jets_df.columns:
        logger.warning('old standard column "salt_pTruthJetPt_mean" found, converting to truth value')
        jets_df["p_truthPt"] = jets_df["pt"]*jets_df["salt_pTruthJetPt_mean"] 
        jets_df.drop(columns=["salt_pTruthJetPt_mean"], inplace=True) # drop old column
    
    if "salt_pTruthJetMass_sigma" in jets_df.columns:
        logger.warning('old standard column "salt_pTruthJetMass_sigma" found, converting to truth value')
        jets_df["p_truthMass_sigma"] = jets_df["mass"]*jets_df["salt_pTruthJetMass_sigma"] 
        jets_df.drop(columns=["salt_pTruthJetMass_sigma"], inplace=True) # drop old column

    if "salt_pTruthJetPt_sigma" in jets_df.columns:
        logger.warning('old standard column "salt_pTruthJetPt_sigma" found, converting to truth value')
        jets_df["p_truthPt_sigma"] = jets_df["pt"]*jets_df["salt_pTruthJetPt_sigma"] 
        jets_df.drop(columns=["salt_pTruthJetPt_sigma"], inplace=True) # drop old column
    
    # if new file format, rename columns to standard names
    if "jet_mass_regression_R10TruthLabel_R22v1_TruthJetMass" in jets_df.columns:
        logger.warning('new standard column "jet_mass_regression_R10TruthLabel_R22v1_TruthJetMass" found, renaming to "p_truthMass"')
        jets_df.rename(columns={"jet_mass_regression_R10TruthLabel_R22v1_TruthJetMass":"p_truthMass"}, inplace=True)

    if "jet_pt_regression_R10TruthLabel_R22v1_TruthJetPt" in jets_df.columns:
        logger.warning('new standard column "jet_pt_regression_R10TruthLabel_R22v1_TruthJetPt" found, renaming to "p_truthPt"')
        jets_df.rename(columns={"jet_pt_regression_R10TruthLabel_R22v1_TruthJetPt":"p_truthPt"}, inplace=True)

    # if new groomed variables names, rename columns to standard names
    if "jet_mass_regression_R10TruthLabel_R22v1_TruthGroomedJetMass" in jets_df.columns:
        logger.warning('new standard column "jet_mass_regression_R10TruthLabel_R22v1_TruthGroomedJetMass" found, renaming to "p_truthMass"')
        jets_df.rename(columns={"jet_mass_regression_R10TruthLabel_R22v1_TruthGroomedJetMass":"p_truthMass"}, inplace=True)

    if "jet_pt_regression_R10TruthLabel_R22v1_TruthGroomedJetPt" in jets_df.columns:
        logger.warning('new standard column "jet_pt_regression_R10TruthLabel_R22v1_TruthGroomedJetPt" found, renaming to "p_truthPt"')
        jets_df.rename(columns={"jet_pt_regression_R10TruthLabel_R22v1_TruthGroomedJetPt":"p_truthPt"}, inplace=True)

    if model_name is not None:
        for col in jets_df.columns:
            if col.startswith(model_name) and col.endswith("TruthGroomedJetMass"):
                logger.warning(f'column "{col}" found, renaming to "p_truthMass"')
                new_col = "p_truthMass"
                jets_df.rename(columns={col:new_col}, inplace=True)
            if col.startswith(model_name) and col.endswith("TruthGroomedJetPt"):
                logger.warning(f'column "{col}" found, renaming to "p_truthPt"')
                new_col = "p_truthPt"
                jets_df.rename(columns={col:new_col}, inplace=True)

    return jets_df


if __name__ == "__main__":
    filepath = "/share/lustre/avaitkus/salt/salt/logs/GN2XAux-fullData-50Epoch-flow_20231025-T100442/ckpts/epoch=037-val_loss=0.48245__test_pp_output_test.h5"
    create_output_folders("test_folder")

    jets_df = read_jets_from_test_file(filepath)
    
    tag1 = find_tag_from_filepath(filepath)
    tag2 = find_tag_from_phbb(jets_df)

    standardise_jet_classification_labels(jets_df, tag2)

    print(list(jets_df.columns))