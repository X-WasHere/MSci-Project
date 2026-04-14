import h5py
import numpy as np
import os

def process_directories(base_dir, folder_list):
    print(f"{'Folder Name':<80} | {'First .h5 File':<35} | {'Average pt':<15}")
    print("-" * 135)

    for folder in folder_list:
        dir_path = os.path.join(base_dir, folder)

        if not os.path.exists(dir_path):
            print(f"{folder:<80} | {'[Directory Not Found]':<35} | {'N/A'}")
            continue

        found_file = False
        try:
            files = os.listdir(dir_path)
            files.sort()  
            
            for filename in files:
                if filename.endswith('.h5'):
                    file_path = os.path.join(dir_path, filename)
                    
                    avg_pt = get_avg_pt_from_file(file_path)
                    
                    display_folder = (folder[:75] + '..') if len(folder) > 75 else folder
                    display_file = (filename[:30] + '..') if len(filename) > 30 else filename
                    
                    if avg_pt is not None:
                        print(f"{display_folder:<80} | {display_file:<35} | {avg_pt:.4f}")
                    else:
                        print(f"{display_folder:<80} | {display_file:<35} | {'Error/No pt'}")
                    
                    found_file = True
                    break
            
            if not found_file:
                 print(f"{folder[:75]:<80} | {'[No .h5 files]':<35} | {'N/A'}")

        except OSError as e:
             print(f"{folder[:75]:<80} | {'[Error accessing dir]':<35} | {'N/A'}")

def get_avg_pt_from_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            if 'jets' in f:
                try:
                    pt_values = f['jets']['pt'][:]
                    return np.mean(pt_values)
                except (KeyError, ValueError):
                    try:
                        jets_data = f['jets'][:]
                        return np.mean(jets_data['pt'])
                    except:
                        return None
            else:
                return None
    except Exception:
        return None

if __name__ == "__main__":
    base_directory = "/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-gluon-split-label"
    
    directories_to_check = [
        "user.npond.364703.e7142_s3681_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.364704.e7142_s3681_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.364705.e7142_s3681_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.364706.e7142_s3681_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.364707.e7142_s3681_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800286.e8547_s3797_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800286.e8547_s3797_r13145_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800286.e8547_s3797_r13167_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800286.e8564_s4159_r15530_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800287.e8547_s3797_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800287.e8547_s3797_r13145_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800287.e8547_s3797_r13167_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.800287.e8564_s4159_r15530_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801168.e8514_s4159_r15224_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801169.e8514_s4159_r15224_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801170.e8514_s4159_r15224_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801171.e8514_s4159_r15224_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801172.e8514_s4159_r15224_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801471.e8441_s3681_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801472.e8441_s3681_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801972.e8514_s4159_r15224_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.801973.e8514_s4159_r15224_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802068.e8547_s3797_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802068.e8547_s3797_r13145_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802068.e8547_s3797_r13167_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802068.e8564_s4159_r15530_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802069.e8547_s3797_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802069.e8547_s3797_r13145_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802069.e8547_s3797_r13167_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802069.e8564_s4159_r15530_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802070.e8547_s3797_r13144_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802070.e8547_s3797_r13145_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802070.e8547_s3797_r13167_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5",
        "user.npond.802070.e8564_s4159_r15530_p6453.tdd.GN3_dev_for_norm.25_2_76.Haloween2025-28-g51dab6e_output.h5"
    ]
    
    process_directories(base_directory, directories_to_check)