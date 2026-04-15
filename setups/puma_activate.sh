#!/bin/bash
#### BASH SCRIPT TO ACTIVATE PUMA CONTAINER FOR PLOTTING
#### Pulls latest image from puma repository

DATA_DIR="/home/xzcappon/phd/tools/umami-preprocessing/super-stats-base-dir/output/test/pp_output_test-full_0.h5:/home/xzcappon/phd/tools/umami-preprocessing/super-stats-base-dir/output/test/pp_output_test-full_0.h5"
TRAINING_DATA_DIR="/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-6class/output/pp_output_train.h5:/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-6class/output/pp_output_train.h5"
SUPERSET_DATA_DIR="/share/data1/xzcappon/datasets/ftag/super-stats/split-components/"
JZ_DATA_DIR="/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-gluon-split-label/:/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-gluon-split-label/"
CALO_DATA_DIR="/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-calo-info/:/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-calo-info/"
HF_CONTAM_DATA_DIR="/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-contaminated-bjet/output/:/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-contaminated-bjet/output/"
SIX_CLASS="/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-6class-calo-info/output/:/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-6class-calo-info/output/"

singularity shell \
    -B $PWD \
    -B $DATA_DIR \
    -B $TRAINING_DATA_DIR \
    -B $SUPERSET_DATA_DIR \
    -B $JZ_DATA_DIR \
    -B $CALO_DATA_DIR \
    -B $HF_CONTAM_DATA_DIR \
    -B $SIX_CLASS \
    docker://gitlab-registry.cern.ch/aft/training-images/puma-images/puma:latest