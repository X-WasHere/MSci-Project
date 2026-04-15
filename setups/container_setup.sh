#!/bin/bash
#### BASH SCRIPT TO ACTIVATE CONTAINER INSIDE CONDA ENV

### 1. Navigate to salt directory and setup conda
cd /home/xzcapfed/salt
source setup/setup_conda.sh

### 2. Navigate to home directory and acivate container
cd /home/xzcapfed/
source activate_container.sh

### 3. Navigate back to salt and setup conda again
cd /home/xzcapfed/salt
source setup/setup_conda.sh