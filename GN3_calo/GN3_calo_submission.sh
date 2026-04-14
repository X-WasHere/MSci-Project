#!/bin/bash
#SBATCH --job-name=salt-train-GN3V01_smaller
#SBATCH -p LIGHTGPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=/home/xzcapfed/salt/salt/logs/slurm-%j.%x.%a.out

set -euo pipefail

############################
# User settings
############################
SALT_DIR="/home/xzcapfed/salt" # change to your salt dir
cd "${SALT_DIR}"
LIGHT_SETUP=""
IMAGE="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:latest/"
EXTRA_BINDS="${EXTRA_BINDS:-}"   # e.g. "/scratch /data/shared:/mnt/shared"

CONFIG_PATH="/home/xzcapfed/MSci/GN3_calo/configs/GN3_benchmark_tauregress.yaml" # training config path

# Save dir and trainer options (your requested style)
OPTIONS="--trainer.devices 1 --data.num_workers 10" # additional salt training settings
PYTORCH_CMD="salt fit --config ${CONFIG_PATH} ${OPTIONS}"

############################
# Host-side prep
############################

# Source site light env
if [[ -f "${LIGHT_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${LIGHT_SETUP}"
fi

# Singularity helper dirs
SING_DIR="${SALT_DIR}/.sing"
mkdir -p "${SING_DIR}/MSci/GN3_calo/logs" "${SING_DIR}/pipcache" "${SING_DIR}/home"

# Source setup_conda.sh on the HOST
source "${SALT_DIR}/setup/setup_conda.sh"

# COMET prep (add your comet keys and project details here)
export COMET_API_KEY=''
export COMET_WORKSPACE=''
export COMET_PROJECT_NAME='salt'
export OMP_NUM_THREADS=1
export COMET_OFFLINE_DIRECTORY='/home/xzcapfed/MSci/GN3_calo/logs/tauregress'


############################
# Build binds
############################

BIND_ARGS=(
  --bind "${SALT_DIR}:/work"
  --bind "${SING_DIR}/MSci/GN3_calo/logs:/MSci/GN3_calo/logs"
  --bind "${SING_DIR}/pipcache:/root/.cache/pip"
  --home "${SING_DIR}/home"
  --bind "/home/xzcapfed/MSci/GN3_calo/"
)

if [[ -n "${EXTRA_BINDS}" ]]; then
  read -r -a EXTRA_ARRAY <<< "${EXTRA_BINDS}"
  for b in "${EXTRA_ARRAY[@]}"; do
    BIND_ARGS+=( --bind "$b" )
  done
fi

############################
# Container command
############################

# Pass CONFIG_PATH / OPTIONS in env to keep the command small & readable
export CONFIG_PATH OPTIONS PYTORCH_CMD

CONTAINER_CMD='
set -euo pipefail
cd /work

# Source setup_conda.sh AGAIN inside container
source setup/setup_conda.sh

# Activate env (if not auto-activated)
conda activate salt || true

# Run
echo "Running: ${PYTORCH_CMD}"
eval "${PYTORCH_CMD}"
'

singularity exec --nv \
  "${BIND_ARGS[@]}" \
  "${IMAGE}" \
  bash -lc "${CONTAINER_CMD}"
