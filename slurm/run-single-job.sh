#!/bin/bash
source ~/.bashrc
module load gcc arrow python/3.12.4 cuda/12.6 httpproxy/1.0
cd /project/aip-sreddy/tvergara/sah
unset PYTHONPATH
export HYDRA_AUTO_SCHEMA=0
source .venv/bin/activate
eval "$@"
