#!/bin/bash
source /afs/cern.ch/work/a/aguzel/private/bamboo_105/pytorch_env/bin/activate
cd /afs/cern.ch/work/a/aguzel/private/bamboo_105/HHtoWWbb_Run3/XGBoost
python train.py --output_dir $1 --max_depth $2