#!/bin/bash
source ~/.bashrc
module load gcc arrow python/3.12.4 cuda/12.6 httpproxy/1.0
cd $SCRATCH/sah
unset PYTHONPATH
source .venv/bin/activate
eval "$@"
