'''
Creates .h5 datasets for particular samples i.e. ttbar, Higgs sample, Z prime etc.
Can create a dataset containing train, evaluation or test data.
Sample datasets used contain the HFShowerLabel column, allowing us to investigate HF contamination.
'''

import h5py
import glob
import os

from puma.utils import logger

def flatten(lst):
    '''
    Flatten a list of lists into a single list
    '''
    flat_list = []
    for sub_lst in lst:
        for element in sub_lst:
            flat_list.append(element)

    return flat_list

def filter_directories(directories, filters, dataset_type="test"):
    '''
    Filters directories using a list of filters. Uses any and or for filtering so for exclusive 
    filtering call function several times.
    '''
    filtered_directories = [f for f in directories 
                        if dataset_type in f and any(fltr in f for fltr in filters)]

    return filtered_directories

def concatenate_dataset(h5files, output_path):
    '''
    Using a list of h5 file directories concateanates into a single h5 file saved in the 
    output path.
    '''
    # concatenate files
    with h5py.File(output_path, "w") as h5fw:

        for i, h5file in enumerate(h5files):
            logger.info(f"Processing {i+1} / {len(h5files)}")

            with h5py.File(h5file, "r") as h5fr:
                # loop through every dataset key (i.e. jets or electrons)
                for key in h5fr.keys():
                    # ignore any groups, only continue with Datasets
                    if not isinstance(h5fr[key], h5py.Dataset):
                        continue

                    new_data = h5fr[key]
                    maxshape = (None, ) + new_data.shape[1:] # data may have shape (N, 50) for example

                    if key not in h5fw:
                        h5fw.create_dataset(
                            key,
                            data=new_data,
                            maxshape=maxshape,
                            chunks=True
                        )
                    
                    else:
                        old_data = h5fw[key]

                        old_rows = old_data.shape[0] 
                        new_rows = new_data.shape[0]
                        tot_rows = old_rows + new_rows

                        # resize with new number of rows and insert new data
                        old_data.resize(tot_rows, axis=0)
                        old_data[old_rows:] = new_data

def virtual_dataset(h5files, output_path):
    '''
    Creates a virtual dataset to access h5 files to avoid copying and merging many h5 files. Returns 
    a mapping which can be used as a regular h5 file.
    '''
    # h5files = sorted(files) # ensures predictable behaviour

    if not h5files:
        raise FileNotFoundError("No files found to merge.")
    
    keys = ["jets"] # target keys

    # dictionary containing virtual layouts for each dataset (i.e. jets, electrons...)
    layouts = {}

    # assuming same structure across h5 files, use first to create VirtualLayout object
    with h5py.File(h5files[1], "r") as h5fr:
        # loop through datasets
        for key in keys:
            columns = h5fr[key].shape[1:]
            dtype = h5fr[key].dtype

            rows = 0
            for h5file in h5files:
                with h5py.File(h5file, "r") as file:
                    rows += file[key].shape[0]
            
            # create VirtualLayout for each datasedt
            layouts[key] = h5py.VirtualLayout(shape=(rows,) + columns, dtype=dtype)

    # create mapping 
    for key in keys:
        current_index = 0
        for h5file in h5files:
            with h5py.File(h5file, 'r') as h5fr:
                dset_shape = h5fr[key].shape
                length = dset_shape[0]

                vsource = h5py.VirtualSource(h5file, key, shape=dset_shape)
                
                # Map this file to the correct slice in the virtual dataset
                layouts[key][current_index : current_index + length] = vsource
                
                current_index += length

    # write virtual file
    with h5py.File(output_path, 'w', libver='latest') as h5fw:
        for key in keys:
            h5fw.create_virtual_dataset(key, layouts[key])

    print(f"Created virtual dataset at: {output_path}")
    return


# --------------- Building Dataset ------------------
datasets_path = ""
jz_datasets_path = ""
output_path = "/home/xzcapfed/MSci/flavour_contamination/sample_datasets/JZ_output_801172.h5"

logger.info("Grabbing and filtering files")
# Get directories containing .h5 files
search_pattern = os.path.join(jz_datasets_path, "*") # * means join anything after
directories = glob.glob(search_pattern)

# filtered_directories = filter_directories(directories,["top"], dataset_type="test")
# filtered_directories = filter_directories(filtered_directories, ["410470", "410471"], dataset_type="test")
# filtered_directories = filter_directories(filtered_directories, ["ghostbjets", "ghostcjets"], dataset_type="test")
filtered_directories = filter_directories(directories, 
                                          filters=["801172"], 
                                          dataset_type="output")

# Get h5 files from each directory
files = [glob.glob(os.path.join(directory, "*.h5")) for directory in filtered_directories]
h5files = flatten(files)

concatenate_dataset(h5files, output_path)

    


                    

